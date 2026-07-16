import csv
import os
from datetime import datetime
import tkinter as tk
from tkinter import Label, filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import processing
from ultralytics import YOLO

model = YOLO("./models/yolo26l_pt.pt")  # initialize model
phase_names = ["Vial", "Broken cream", "Broken coalescence", "20% St", "Tt", "NS"]

# Initialise global variables
img_path = None
last_vial_count = 0
last_detected_phases = []

def choose_input():
    global img_path
    # ======== File input =========
    filetypes = [
        ("Image Files", "*.jpg *.jpeg *.png *.tiff *.bmp"),
    ]

    img_path = filedialog.askopenfilename(title="Select media file", initialdir="./images", filetypes=filetypes)
    if not img_path:
        return

    update_feed()

def fit_to_display(img, max_w=1180, max_h=350):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

def update_feed():
    img = cv2.imread(img_path)
    img = fit_to_display(img)

    # Display a single image in the video panel
    # Convert from BGR (OpenCV) to RGB (Pillow/Tkinter)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Convert to Tkinter-compatible PhotoImage
    imgtk = ImageTk.PhotoImage(image=img_pil)

    # Update the Tkinter label (same widget as used for video)
    img_panel.imgtk = imgtk       # keep a reference to avoid garbage collection
    img_panel.configure(image=imgtk)

def capture_and_detect():
    global last_vial_count, last_detected_phases
    frame = cv2.imread(img_path)
    frame, vial_count, detected_classes = processing.ProcessImage(frame, model, phase_names)
    frame = fit_to_display(frame)
    # Convert processed frame to RGB and update the detection panel
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = Image.fromarray(frame_rgb)
    frame_tk = ImageTk.PhotoImage(image=frame_rgb)
    detected_panel.imgtk = frame_tk
    detected_panel.configure(image=frame_tk)

    last_vial_count = vial_count
    last_detected_phases = [phase_names[c] for c in sorted(detected_classes) if c < len(phase_names)]
    capture_button.configure(state="normal")

def save_data():
    results_path = "results.csv"
    file_exists = os.path.exists(results_path)

    with open(results_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "image", "vial_count", "phases_detected"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            os.path.basename(img_path),
            last_vial_count,
            "; ".join(last_detected_phases),
        ])

    messagebox.showinfo("Saved", f"Results appended to {results_path}")

# Set up the main window
root = tk.Tk()
root.title("Phase Detection")
root.geometry("1200x800")

source_button = ttk.Button(root, text="Choose Input File", command=choose_input)
source_button.pack(side="top", fill="x", padx=10, pady=10)

# Create a panel for the image
img_panel = Label(root)
img_panel.pack(side="top", padx=10, pady=10)

# Create a panel for the detection result
detected_panel = Label(root)
detected_panel.pack(side="top", padx=10, pady=10)

set_up = ttk.Button(root, text="Detect Vials and Phases", command=capture_and_detect)
set_up.pack(side="bottom", fill="both", padx=10, pady=10)

capture_button = ttk.Button(root, text="Save Data", command=save_data, state="disabled")
capture_button.pack(side="bottom", fill="both", padx=10, pady=10)

root.mainloop()


