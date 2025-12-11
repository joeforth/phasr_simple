import csv
import os
import cv2
from ultralytics import YOLO


img_path = './training_data/phasr2.v2-2025_11_18.yolov11/valid/images'
model_path = './trained_models'

img = 'frame32_jpg.rf.b2db85add839890a080f07cdc3894f5c.jpg'
model = 'smalldataset_run1.pt'

img = os.path.join(img_path, img)
model = os.path.join(model_path, model)
model = YOLO(model)

phase_name = ['foam', 'high_turbidity', 'low_turbidity', 'transparent', 'transparent_gel', 'vial']

img = cv2.imread(img)

# Documentation here - https://docs.ultralytics.com/modes/predict/#inference-arguments
# Use guide here - https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
results = model(img)
# results = model.predict(source=img, conf=0.65, iou=0.3, show_labels=False, device="cpu")

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk



