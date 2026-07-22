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


def DetectPhases(image, model, conf=0.4, iou=0.5, phase_names=None):
    # Takes messy YOLO predict object and extracts bounding box coords and class labels
    # Returns:
    # Detections - list of identified objects with xy coords and label
    # Vials - list of clustered objects linking vials and phases
    results = model.predict(source=image, conf=conf, iou=iou, show_labels=False, agnostic_nms=True, device="cpu")

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            xc, yc = 0.5*(x1 + x2), 0.5*(y1 + y2)       # Locate centre of bonding boxes
            detections.append((class_id, x1, y1, x2, y2, xc, yc, conf))

    detections = suppress_overlapping_boxes(detections, iou)

    # Draw Vial (index 0) first so phase boxes render on top
    detections.sort(key=lambda d: (0 if d[0] == 0 else 1))

    # Cluster values by bbox x-centre - assigns phases to vials
    x_arr = [(d[0], d[5], d[6]) for d in detections]
    vials = cluster_x_values(x_arr)

    return image, detections, vials


def FilterDetections(detections, filename, n_vials)
    # Build a dictionary - for each vial, create vial with index, phase detection, and confidence score
    detections_dict = dict(file=filename, )

# def SaveData(frame, filename, vials, phase_names):
#     for v_idx, v in enumerate(vials):
#         if len(v) == 2:
            
#         if len(v) == 1:
#             phase_index = -1
#             phase_label = 'Null'
#         with open('vial_data.csv', 'w', encoding="utf-8") as f:
#             f.write()
            
    


def suppress_overlapping_boxes(detections, iou_thres=0.5):
    # Our YOLO26 model is end-to-end/NMS-free, so ultralytics' iou/agnostic_nms
    # predict() args are silently ignored (see ultralytics.utils.nms.non_max_suppression).
    # Manually suppress lower-confidence boxes that overlap another box, regardless of class.
    # detections: list of (class_id, x1, y1, x2, y2, conf)
    by_conf = sorted(detections, key=lambda d: d[5], reverse=True)
    kept = []
    for det in by_conf:
        if any(box_iou(det[1:5], k[1:5]) > iou_thres for k in kept):
            continue
        kept.append(det)
    return kept


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


def box_iou(box_a, box_b):
    # IoU between two (x1, y1, x2, y2) boxes
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def VisualisePhases(image, detections, phase_names):
    # Preps the image and labels for visualisation, which is then mostly handled by draw_legend()
    # Copy image so opencv can manipulate directly
    img = image.copy()
    for d in detections:
        class_id, x1, y1, x2, y2 = d[0], d[1], d[2], d[3], d[4]
        color = class_colours[class_id] if class_id < len(class_colours) else (255, 255, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 15)

    if phase_names:
        draw_legend(img, phase_names)

    return img


def draw_legend(img, phase_names):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    thickness = 3
    pad = 12

    entries = list(enumerate(phase_names))

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
        

def fit_to_display(img, max_w=1180, max_h=500):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img