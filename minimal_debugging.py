import cv2
import processing
from ultralytics import YOLO

# Some user-defined settings
conf = 0.5      # Confidence threshold below which we ignore detected objects
iou = 0.7       # Amount of overlap between detected objects above which we ignore them - this can be very high for vial detection!
n_vials = 6     # Minimum number of vials in each image - triggers a warning if we detect fewer than this
img_w, img_h = 1180, 500    # Maximum dimensions of the image shown
filedir = './images/'
filename = 'Aracel 165 (RB) emulsions day 1 Squalane only samples 7-12.jpg'
phase_names = ["Vial", 
               "Broken cream", 
               "Broken coalescence", 
               "20% St", 
               "Tt", 
               "NS"]

model = YOLO("./trained_models/yolo26l_pt.pt")  # initialize model

frame = cv2.imread(os.path.join(filedir, filename))
frame, detections, vials = processing.DetectPhases(frame, model, conf, iou, phase_names)
frame = processing.VisualisePhases(frame, detections, phase_names)
print('vials is', vials)
detections_dict = FilterDetections(vials, filename, n_vials) 
processing.SaveData(frame, filename, vials, phase_names)

frame = processing.fit_to_display(frame, img_w, img_h)
cv2.imshow("Image", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()