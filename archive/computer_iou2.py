import numpy as np
def compute_iou(box1, box2):
    print(box1)
    print(box2)
    # Calculate the coordinates of the intersection rectangle
    # box1 = list(box1.values())
    # box2 = list(box2.values())

    # Calculate the areas of the two bounding boxes - mine
    box1_area = abs(box1[2] - box1[0]) * abs(box1[3] - box1[1])
    box2_area = abs(box2[2] - box2[0]) * abs(box2[3] - box2[1])
    print(box1_area, box2_area)

    x_left = max(box1[0], box2[0])
    x_right = min(box1[2], box2[2])
    print(x_left, x_right)
    y_top = max(box1[1], box2[1])
    y_bottom = min(box1[3], box2[3])
    print(y_top, y_bottom)

    # Check for intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the areas of the two bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    print(box1_area, box2_area)

    # Calculate the area of the intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou

truths2 = {
    # xmin, ymin, xmax, ymax
    "A": [3, 7, 8, 11],
    "B": [13, 2, 17, 6],
    "D": [5, 2, 7, 4]
}

preds2 = {
    # xmin, ymin, xmax, ymax, confidence
    "A1": [2, 8, 6, 12, 0.82],
    "A2": [5, 10, 9, 14, 0.38],
    "B1": [14, 3, 18, 7, 0.72],
    "B2": [12, 1, 16, 5, 0.83],
    "C1": [1, 3, 4, 6, 0.17],
    "D1": [16, 10, 20, 14, 0.32]
}

iou = compute_iou(box1=[90, 330, 240, 210], box2=[150, 420, 270, 300])
print(iou)


import numpy as np

SMOOTH = 1e-6

def iou_numpy(outputs: np.array, labels: np.array):
  outputs = outputs.squeeze(1)

  intersection = (outputs & labels).sum((1, 2))
  union = (outputs | labels).sum((1, 2))

  iou = (intersection + SMOOTH) / (union + SMOOTH)

  thresholded = np.ceil(np.clip(20 * (iou - 0.7), 0, 10)) / 10

  return thresholded  # Or thresholded.mean()

  iou_numpy