import cv2
import numpy as np

class_colours = [
    (255,  80,   0),  # 0 Vial               - blue
    ( 60, 200,  60),  # 1 Broken cream        - green
    (  0,  60, 220),  # 2 Broken coalescence  - red
    (  0, 140, 255),  # 3 20% St              - orange
    (180,   0, 180),  # 4 Tt                  - purple
    (  0, 200, 200),  # 5 NS                  - yellow
]

def cluster_x_values(x_values, threshold=30):
    # Sorts and groups detected items - so you can associate a phase with a vial
    # Each entry in x-values is tuple - (class_id, x1, y1)
    if not x_values:
        return []
    # Sort all entries by tup[1] - x1 coordinate
    x_values.sort(key=lambda tup: tup[1])

    groups = []
    current_group = [x_values[0]]
    # Run pair-wise comparison that groups vials that are within the threshold distance
    for current in x_values[1:]:
        if abs(current[1] - current_group[-1][1]) <= threshold:
            current_group.append(current)
        else:
            groups.append(current_group)
            current_group = [current]
    groups.append(current_group)
    return groups


def DetectPhases(image, model, phase_names=None):
    # Takes messy YOLO predict object and extracts bounding box coords and class labels
    # Returns:
    # Detections - list of identified objects with xy coords and label
    # Vials - list of clustered objects linking vials and phases
    results = model.predict(source=image, conf=0.65, iou=0.3, show_labels=False, device="cpu")

    detections = []
    for result in results:
        for box in result.boxes:
            print('box is', box)
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append((class_id, x1, y1, x2, y2))
    # Draw Vial (index 0) first so phase boxes render on top
    detections.sort(key=lambda d: (0 if d[0] == 0 else 1))
    print('detections is', detections)

    x_arr = [(d[0], d[1], d[2]) for d in detections]
    vials = cluster_x_values(x_arr)
    print('vials is', vials)

    return image, detections, vials


def VisualisePhases(image, detections, phase_names):
    # Copy image so opencv can manipulate directly
    img = image.copy()
    # Constructs a set of detected classes for building legend for visualisation
    detected_classes = set()
    for class_id, x1, y1, x2, y2 in detections:
        color = class_colours[class_id] if class_id < len(class_colours) else (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 15)
        detected_classes.add(class_id)

    if phase_names and detected_classes:
        draw_legend(img, phase_names, detected_classes)

    return img


def draw_legend(img, phase_names, detected_classes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 3
    pad = 12

    entries = [(cls, phase_names[cls] if cls < len(phase_names) else str(cls))
               for cls in sorted(detected_classes)]

    text_sizes = [cv2.getTextSize(label, font, font_scale, thickness)[0] for _, label in entries]
    max_tw = max(s[0] for s in text_sizes)
    max_th = max(s[1] for s in text_sizes)
    swatch = max_th
    row_h = swatch + pad * 2

    legend_w = pad + swatch + pad + max_tw + pad
    legend_h = row_h * len(entries) + pad

    ih, iw = img.shape[:2]
    lx = iw - legend_w - pad
    ly = max(0, (ih - legend_h) // 2)

    # Semi-transparent dark background
    y2 = min(ly + legend_h, ih)
    x2 = min(lx + legend_w, iw)
    roi = img[ly:y2, lx:x2]
    img[ly:y2, lx:x2] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)

    for i, (cls, label) in enumerate(entries):
        color = class_colours[cls] if cls < len(class_colours) else (255, 255, 255)
        y = ly + pad + i * row_h
        cv2.rectangle(img, (lx + pad, y), (lx + pad + swatch, y + swatch), color, -1)
        _, th = text_sizes[i]
        cv2.putText(img, label, (lx + pad + swatch + pad, y + swatch // 2 + th // 2),
                    font, font_scale, (255, 255, 255), thickness)