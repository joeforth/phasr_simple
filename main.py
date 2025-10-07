import csv
import glob
import os
import tkinter as tk
from tkinter import Label, Button, Entry, simpledialog, filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pygrabber.dshow_graph import FilterGraph
import processing
import matplotlib.pyplot as plt

# Initialize the camera
cap = cv2.VideoCapture("noFeed.jpeg")
# Global variables to hold schedule intervals (in milliseconds) and current index
intervals = []
current_interval_index = 0
count=0
size_cm = []
folder=''
capture_count=0
phase_name = ["Vials","foams","slightly turbids","transparent","very turbids"]
description_entries = []
timer_job = None



def display_image(img):
    # Display a single image in the video panel
    # Convert from BGR (OpenCV) to RGB (Pillow/Tkinter)
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)

    # Convert to Tkinter-compatible PhotoImage
    imgtk = ImageTk.PhotoImage(image=img_pil)

    # Update the Tkinter label (same widget as used for video)
    video_panel.imgtk = imgtk       # keep a reference to avoid garbage collection
    video_panel.configure(image=imgtk)

processed,new_count,finalVials = processing.ProcessImage(detected_frame,size_cm)

# Set up the main window
root = tk.Tk()
root.title("Camera Feed with Object Detection")

source_button = ttk.Button(root, text="Choose Input Source", command=choose_input)
source_button.pack(side="top", fill="x", padx=10, pady=10)

# Create a panel for the live video feed
video_panel = Label(root)
video_panel.pack(side="left", padx=10, pady=10)

# Create a panel for the detection result
detected_panel = Label(root)
detected_panel.pack(side="right", padx=10, pady=10)

set_up = ttk.Button(root, text="Set Up Experiment", command=lambda: (size_cm.clear(),capture_and_detect()))
set_up.pack(side="bottom", fill="both", padx=10, pady=10)

# Button to manually capture an image and run object detection
capture_button = ttk.Button(root, text="Detect", command=lambda: capture_and_detect(count),state="disabled")
capture_button.pack(side="bottom", fill="both", padx=10, pady=10)

entry_container = tk.Frame(root)
entry_container.pack(side="bottom", fill="x", padx=10, pady=10)

update_feed()
root.mainloop()
cap.release()


