import csv
import glob
import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pygrabber.dshow_graph import FilterGraph
import processing
import matplotlib.pyplot as plt

# Initialise global variables
img_path = ''
model = ''

size_cm = []
folder=''
capture_count=0
phase_name = ['foam', 'high_turbidity', 'low_turbidity', 'transparent', 'transparent_gel', 'vial']
description_entries = []

img = cv2.imread(img_path)
detected_frame = cv2.imread(img_path)
processed,new_count,finalVials = ProcessImage(detected_frame,size_cm)

def ProcessImage(image, vial_size):
    x_arr = []
    results = model.predict(source=image, conf=0.65, iou=0.3, show_labels=False,device="cpu")

    # Extract bounding boxes
    img = results[0].plot(labels=False, conf=False)  # returns a numpy array

    for result in results:
        for box in result.boxes:
            class_id = box.cls[0].item()  # Class label
            boxes = box.xywh  # Alternative format

            for boxe in boxes:
                x, y, w, h = boxe
                x_arr.append((class_id, x.item(), y.item(), h.item()))

    vials = cluster_x_values(x_arr)
    finalVials = []
    #measure the phase size and display it
    for i in range(len(vials)):
        try:
            size = getSize(vials[i], vial_size[i])
            finalVials.append(size)
            for j in range(len(size)):
                text = f"{size[j][3]:.2f} cm"
                position = (int(size[j][1]), int(size[j][2]))  # (x, y) coordinates
                # Define font, font scale, color (BGR), and thickness
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 255, 255)
                thickness = 1
                cv2.putText(img, text, position, font, font_scale, color, thickness)
        except ValueError as e:
            print(f"Skipping group — no reference found: {e}")
            continue
        except IndexError:
            # No provided size for this vial — skip it
            continue

    return img, len(vials), finalVials


