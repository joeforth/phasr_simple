from ultralytics import YOLO
import cv2
import numpy as np

# model = YOLO("YOLOV11modelv9.pt")  # initialize model
model = YOLO("best.pt")  # initialize model

CLASS_COLORS = [
    (255,  80,   0),  # 0 Vial               - blue
    ( 60, 200,  60),  # 1 Broken cream        - green
    (  0,  60, 220),  # 2 Broken coalescence  - red
    (  0, 140, 255),  # 3 20% St              - orange
    (180,   0, 180),  # 4 Tt                  - purple
    (  0, 200, 200),  # 5 NS                  - yellow
]

def cluster_x_values(x_values, threshold=30):
    if not x_values:
        return []
    x_values.sort(key=lambda tup: tup[1])

    groups = []
    current_group = [x_values[0]]

    for current in x_values[1:]:
        if abs(current[1] - current_group[-1][1]) <= threshold:
            current_group.append(current)
        else:
            groups.append(current_group)
            current_group = [current]
    groups.append(current_group)
    return groups


def _draw_legend(img, phase_names, detected_classes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 3
    pad = 12
    swatch = 35

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
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (255, 255, 255)
        y = ly + pad + i * row_h
        cv2.rectangle(img, (lx + pad, y), (lx + pad + swatch, y + swatch), color, -1)
        _, th = text_sizes[i]
        cv2.putText(img, label, (lx + pad + swatch + pad, y + swatch // 2 + th // 2),
                    font, font_scale, (255, 255, 255), thickness)


def ProcessImage(image, phase_names=None):
    results = model.predict(source=image, conf=0.65, iou=0.3, show_labels=False, device="cpu")

    img = image.copy()

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            detections.append((class_id, x1, y1, x2, y2))

    # Draw Vial (index 0) first so phase boxes render on top
    detections.sort(key=lambda d: (0 if d[0] == 0 else 1))

    detected_classes = set()
    for class_id, x1, y1, x2, y2 in detections:
        color = CLASS_COLORS[class_id] if class_id < len(CLASS_COLORS) else (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 15)
        detected_classes.add(class_id)

    if phase_names and detected_classes:
        _draw_legend(img, phase_names, detected_classes)

    x_arr = [(d[0], d[1], d[2]) for d in detections]
    vials = cluster_x_values(x_arr)
    return img, len(vials)
