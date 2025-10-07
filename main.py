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

# Initialise global variables
img_path = None

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

def choose_input():
    global img_path
    # ======== File input =========
    filetypes = [
        ("Image Files", "*.jpg *.jpeg *.png *.tiff *.bmp"),
    ]

    img_path = filedialog.askopenfilename(title="Select media file", filetypes=filetypes)
    if not img_path:
        return

    img = cv2.imread(img_path)
    if img is None:
        return messagebox.showerror("Error", "Could not read image file.")
    
    update_feed()

def update_feed():
    img = cv2.imread(img_path)

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
    global capture_count
    global size_cm
    global count
    global folder

    detected_frame = cv2.imread(img_path)

    processed,new_count,finalVials = processing.ProcessImage(detected_frame,size_cm)
    # Convert processed frame to RGB and update the detection panel
    detected_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    img_detected = Image.fromarray(detected_rgb)
    imgtk_detected = ImageTk.PhotoImage(image=img_detected)
    detected_panel.imgtk = imgtk_detected
    detected_panel.configure(image=imgtk_detected)
    capture_button.configure(state="normal")

# Set up the main window
root = tk.Tk()
root.title("Object Detection")
root.geometry("1200x800")

source_button = ttk.Button(root, text="Choose Input File", command=choose_input)
source_button.pack(side="top", fill="x", padx=10, pady=10)

# Create a panel for the image
img_panel = Label(root)
img_panel.pack(side="top", padx=10, pady=10)

# Create a panel for the detection result
detected_panel = Label(root)
detected_panel.pack(side="top", padx=10, pady=10)

set_up = ttk.Button(root, text="Detect Vials and Phases", command=lambda: (size_cm.clear(),capture_and_detect()))
set_up.pack(side="bottom", fill="both", padx=10, pady=10)

# Button to run object detection
capture_button = ttk.Button(root, text="Save Data", command=lambda: capture_and_detect(),state="disabled")
capture_button.pack(side="bottom", fill="both", padx=10, pady=10)

entry_container = tk.Frame(root)
entry_container.pack(side="bottom", fill="x", padx=10, pady=10)

root.mainloop()


