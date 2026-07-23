import os
import tkinter as tk
from tkinter import Label, filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import processing
from ultralytics import YOLO

phase_names = processing.load_class_names("./classes.txt")
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
DATA_OUT_DIR = "output_data"
VIS_OUT_DIR = "output_detections"

# Initialise global variables
model = None
img_path = None
batch_dir = None
last_detections_class = None
last_vis_frame = None


def require_model():
    if model is None:
        messagebox.showerror("No model loaded", "Please load a model (.pt file) before running detection.")
        return False
    return True


def get_settings():
    # Reads and validates the conf / iou / n_vials tuning fields
    try:
        conf = float(conf_var.get())
        iou = float(iou_var.get())
        n_vials = int(n_vials_var.get())
    except ValueError:
        raise ValueError("conf and iou must be numbers, n_vials must be an integer")
    if not (0.0 <= conf <= 1.0):
        raise ValueError("conf must be between 0 and 1")
    if not (0.0 <= iou <= 1.0):
        raise ValueError("iou must be between 0 and 1")
    return conf, iou, n_vials


def log_message(text):
    log_panel.configure(state="normal")
    log_panel.insert("end", text + "\n")
    log_panel.configure(state="disabled")
    log_panel.see("end")


def log_warnings_for_result(detections_class, n_vials):
    for msg in processing.flag_warnings(detections_class, n_vials):
        log_message(f"[WARNING] {msg}")


def fit_to_display(img, max_w=1180, max_h=350):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def show_on_panel(panel, img_bgr):
    img_rgb = cv2.cvtColor(fit_to_display(img_bgr), cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
    panel.imgtk = img_tk  # keep a reference to avoid garbage collection
    panel.configure(image=img_tk)


# ======== Model ========

def choose_model():
    global model
    filetypes = [("PyTorch model", "*.pt")]
    path = filedialog.askopenfilename(title="Select model file", initialdir="./trained_models", filetypes=filetypes)
    if not path:
        return

    model = YOLO(path)
    model_label.configure(text=os.path.basename(path))
    log_message(f"Loaded model: {path}")


# ======== Single-image tuning ========

def choose_input():
    global img_path
    filetypes = [("Image Files", "*.jpg *.jpeg *.png *.tiff *.bmp")]
    path = filedialog.askopenfilename(title="Select image", initialdir="./images", filetypes=filetypes)
    if not path:
        return
    img_path = path

    show_on_panel(img_panel, cv2.imread(img_path))
    detect_button.configure(state="normal")
    save_button.configure(state="disabled")


def run_single_detect():
    global last_detections_class, last_vis_frame
    if not require_model():
        return
    try:
        conf, iou, n_vials = get_settings()
    except ValueError as e:
        messagebox.showerror("Invalid settings", str(e))
        return

    frame = cv2.imread(img_path)
    frame, detections, vials = processing.detect_phases(frame, model, conf, iou, phase_names)
    detections_class = processing.process_detections(vials, os.path.basename(img_path), phase_names)

    log_warnings_for_result(detections_class, n_vials)

    vis_frame = processing.visualise_phases(frame, detections, phase_names)
    show_on_panel(detected_panel, vis_frame)

    last_detections_class = detections_class
    last_vis_frame = vis_frame
    save_button.configure(state="normal")


def save_single_data():
    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    os.makedirs(VIS_OUT_DIR, exist_ok=True)

    name, ext = os.path.splitext(os.path.basename(img_path))
    file_out = os.path.join(DATA_OUT_DIR, name)
    processing.save_data(last_detections_class, file_out)
    cv2.imwrite(os.path.join(VIS_OUT_DIR, f"{name}_vis{ext}"), last_vis_frame)

    log_message(f"Saved results for {os.path.basename(img_path)} to {DATA_OUT_DIR} and {VIS_OUT_DIR}")
    messagebox.showinfo("Saved", f"Results saved to {DATA_OUT_DIR}\nVisualisation saved to {VIS_OUT_DIR}")


# ======== Batch processing ========

def choose_batch_dir():
    global batch_dir
    path = filedialog.askdirectory(title="Select image directory", initialdir="./images")
    if not path:
        return
    batch_dir = path
    batch_dir_label.configure(text=batch_dir)
    run_batch_button.configure(state="normal")


def run_batch():
    if not require_model():
        return
    try:
        conf, iou, n_vials = get_settings()
    except ValueError as e:
        messagebox.showerror("Invalid settings", str(e))
        return

    filenames = sorted(f for f in os.listdir(batch_dir) if f.lower().endswith(image_exts))
    if not filenames:
        messagebox.showwarning("No images found", f"No image files found in {batch_dir}")
        return

    os.makedirs(DATA_OUT_DIR, exist_ok=True)
    os.makedirs(VIS_OUT_DIR, exist_ok=True)

    detections_classes = []
    for filename in filenames:
        frame = cv2.imread(os.path.join(batch_dir, filename))
        frame, detections, vials = processing.detect_phases(frame, model, conf, iou, phase_names)
        detections_class = processing.process_detections(vials, filename, phase_names)
        detections_classes.append(detections_class)

        log_warnings_for_result(detections_class, n_vials)

        name, ext = os.path.splitext(filename)
        file_out = os.path.join(DATA_OUT_DIR, name)
        processing.save_data(detections_class, file_out)

        vis_frame = processing.visualise_phases(frame, detections, phase_names)
        cv2.imwrite(os.path.join(VIS_OUT_DIR, f"{name}_vis{ext}"), vis_frame)

        log_message(f"Processed {filename}")
        root.update_idletasks()

    processing.save_batch_data(detections_classes, os.path.join(DATA_OUT_DIR, "batch"))
    log_message(f"Batch complete: {len(filenames)} image(s) processed, data saved to {DATA_OUT_DIR}, "
                f"visualisations saved to {VIS_OUT_DIR}")
    messagebox.showinfo("Batch complete", f"{len(filenames)} image(s) processed.\n"
                         f"Data saved to {DATA_OUT_DIR}\nVisualisations saved to {VIS_OUT_DIR}")


# ======== Main window ========

def main():
    global root, conf_var, iou_var, n_vials_var
    global img_panel, detected_panel, detect_button, save_button
    global batch_dir_label, run_batch_button, log_panel, model_label

    root = tk.Tk()
    root.title("Phase Detection")
    root.geometry("1200x900")

    # --- Model ---
    model_frame = ttk.LabelFrame(root, text="Model")
    model_frame.pack(side="top", fill="x", padx=10, pady=5)

    model_button = ttk.Button(model_frame, text="Load Model", command=choose_model)
    model_button.pack(side="left", padx=5, pady=5)

    model_label = ttk.Label(model_frame, text="(no model loaded)")
    model_label.pack(side="left", padx=5, pady=5)

    # --- Settings ---
    settings_frame = ttk.LabelFrame(root, text="Settings")
    settings_frame.pack(side="top", fill="x", padx=10, pady=5)

    conf_var = tk.StringVar(value="0.5")
    iou_var = tk.StringVar(value="0.7")
    n_vials_var = tk.StringVar(value="6")

    ttk.Label(settings_frame, text="conf").grid(row=0, column=0, padx=5, pady=5)
    ttk.Entry(settings_frame, textvariable=conf_var, width=8).grid(row=0, column=1, padx=5, pady=5)
    ttk.Label(settings_frame, text="iou").grid(row=0, column=2, padx=5, pady=5)
    ttk.Entry(settings_frame, textvariable=iou_var, width=8).grid(row=0, column=3, padx=5, pady=5)
    ttk.Label(settings_frame, text="n_vials").grid(row=0, column=4, padx=5, pady=5)
    ttk.Entry(settings_frame, textvariable=n_vials_var, width=8).grid(row=0, column=5, padx=5, pady=5)

    # --- Single image tuning ---
    single_frame = ttk.LabelFrame(root, text="Single Image (tune settings)")
    single_frame.pack(side="top", fill="x", padx=10, pady=5)

    source_button = ttk.Button(single_frame, text="Choose Input Image", command=choose_input)
    source_button.pack(side="left", padx=5, pady=5)

    detect_button = ttk.Button(single_frame, text="Detect Vials and Phases", command=run_single_detect, state="disabled")
    detect_button.pack(side="left", padx=5, pady=5)

    save_button = ttk.Button(single_frame, text="Save Data", command=save_single_data, state="disabled")
    save_button.pack(side="left", padx=5, pady=5)

    images_frame = ttk.Frame(root)
    images_frame.pack(side="top", fill="x", padx=10, pady=5)

    img_panel = Label(images_frame)
    img_panel.pack(side="left", padx=5, pady=5, expand=True)

    detected_panel = Label(images_frame)
    detected_panel.pack(side="left", padx=5, pady=5, expand=True)

    # --- Batch processing ---
    batch_frame = ttk.LabelFrame(root, text="Batch Processing")
    batch_frame.pack(side="top", fill="x", padx=10, pady=5)

    batch_dir_button = ttk.Button(batch_frame, text="Choose Directory", command=choose_batch_dir)
    batch_dir_button.pack(side="left", padx=5, pady=5)

    batch_dir_label = ttk.Label(batch_frame, text="(no directory chosen)")
    batch_dir_label.pack(side="left", padx=5, pady=5)

    run_batch_button = ttk.Button(batch_frame, text="Run Batch", command=run_batch, state="disabled")
    run_batch_button.pack(side="left", padx=5, pady=5)

    # --- Log / warnings panel ---
    log_frame = ttk.LabelFrame(root, text="Log / Warnings")
    log_frame.pack(side="top", fill="both", expand=True, padx=10, pady=5)

    log_scrollbar = ttk.Scrollbar(log_frame)
    log_scrollbar.pack(side="right", fill="y")

    log_panel = tk.Text(log_frame, height=10, state="disabled", yscrollcommand=log_scrollbar.set)
    log_panel.pack(side="left", fill="both", expand=True)
    log_scrollbar.configure(command=log_panel.yview)

    root.mainloop()


if __name__ == "__main__":
    main()
