import cv2
import processing
import os
from ultralytics import YOLO

# User-defined settings
conf = 0.5      # Confidence threshold below which we ignore detected objects
iou = 0.7       # Amount of overlap between detected objects above which we ignore them - this can be very high for vial detection!
n_vials = 6     # Minimum number of vials in each image - triggers a warning if we detect fewer than this
img_w, img_h = 1180, 500    # Maximum dimensions of the image shown
filedir = './images/'
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
phase_names = processing.load_class_names('./classes.txt')
data_out_dir = 'output_data'
vis_out_dir = 'output_detections'

model = YOLO("./trained_models/yolo26l_pt.pt")  # initialize model

filenames = sorted(f for f in os.listdir(filedir) if f.lower().endswith(image_exts))

os.makedirs(data_out_dir, exist_ok=True)
os.makedirs(vis_out_dir, exist_ok=True)

detections_classes = []
for filename in filenames:
    frame = cv2.imread(os.path.join(filedir, filename))
    # Detections of the form: (class_id, x1, y1, x2, y2, xc, yc, conf)
    frame, detections, vials = processing.detect_phases(frame, model, conf, iou, phase_names)
    detections_class = processing.process_detections(vials, filename, phase_names)
    detections_classes.append(detections_class)

    for msg in processing.flag_warnings(detections_class, n_vials):
        print(f"[WARNING] {msg}")

    file_out = os.path.join(data_out_dir, os.path.splitext(filename)[0])
    processing.save_data(detections_class, file_out)

    vis_frame = processing.visualise_phases(frame, detections, phase_names)
    name, ext = os.path.splitext(filename)
    vis_path = os.path.join(vis_out_dir, f"{name}_vis{ext}")
    cv2.imwrite(vis_path, vis_frame)

processing.save_batch_data(detections_classes, os.path.join(data_out_dir, 'batch'))
