import cv2
import processing
from ultralytics import YOLO

def fit_to_display(img, max_w=1180, max_h=350):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img

model = YOLO("./trained_models/yolo26l_pt.pt")  # initialize model
phase_names = ["Vial", "Broken cream", "Broken coalescence", "20% St", "Tt", "NS"]

frame = cv2.imread('./images/Aracel 165 (RB) emulsions day 1 Squalane only samples 7-12.jpg')
frame, detections, vials = processing.DetectPhases(frame, model, phase_names)
frame = processing.VisualisePhases(frame, detections, phase_names)

frame = fit_to_display(frame)
cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()