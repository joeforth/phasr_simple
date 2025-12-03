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

results = model.predict(source=img, conf=0.65, iou=0.3, show_labels=False, device="cpu")

# def ProcessImage(image, vial_size):
#     x_arr = []
#     results = model.predict(source=image, conf=0.65, iou=0.3, show_labels=False,device="cpu")

#     # Extract bounding boxes
#     img = results[0].plot(labels=False, conf=False)  # returns a numpy array

#     for result in results:
#         for box in result.boxes:
#             class_id = box.cls[0].item()  # Class label
#             boxes = box.xywh  # Alternative format

#             for boxe in boxes:
#                 x, y, w, h = boxe
#                 x_arr.append((class_id, x.item(), y.item(), h.item()))

#     vials = cluster_x_values(x_arr)
#     finalVials = []
#     #measure the phase size and display it
#     for i in range(len(vials)):
#         try:
#             size = getSize(vials[i], vial_size[i])
#             finalVials.append(size)
#             for j in range(len(size)):
#                 text = f"{size[j][3]:.2f} cm"
#                 position = (int(size[j][1]), int(size[j][2]))  # (x, y) coordinates
#                 # Define font, font scale, color (BGR), and thickness
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 0.5
#                 color = (0, 255, 255)
#                 thickness = 1
#                 cv2.putText(img, text, position, font, font_scale, color, thickness)
#         except ValueError as e:
#             print(f"Skipping group — no reference found: {e}")
#             continue
#         except IndexError:
#             # No provided size for this vial — skip it
#             continue

#     return img, len(vials), finalVials


