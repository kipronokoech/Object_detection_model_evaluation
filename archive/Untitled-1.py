import numpy as np

def compute_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    box1 = list(box1.values())
    box2 = list(box2.values())

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Check for intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the areas of the two bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of the intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate the union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area

    return iou


def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.3):
    """Evaluate predicted boxes against ground truth boxes"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # convert list of predicted boxes to dictionary
    pred_boxes_dict = {}
    for i in range(len(pred_boxes)):
        pred_boxes_dict[f"box_{i}"] = pred_boxes[i]

    # loop over predicted boxes
    detections = []
    for i, pred_box in pred_boxes_dict.items():

        # find best matching ground truth box (highest IoU)
        best_iou = 0
        best_gt_box = {"detected": False}
        for j in range(len(gt_boxes)):
            gt_box = gt_boxes[j]

            gt_box 
            iou_score = compute_iou(pred_box, gt_box)
            print(gt_box, pred_box, "IoU=", iou_score)
            if iou_score > best_iou:
                best_iou = iou_score
                best_gt_box = gt_box

        # classify predicted box as TP, FP or FN based on IoU threshold
        if best_iou >= iou_threshold:
            true_positives += 1
            # mark ground truth box as detected
            best_gt_box["detected"] = True
        else:
            false_positives += 1
        # store the detection for NMS
        detections.append(list(pred_box.values()) + [best_iou])

    # count remaining undetected ground truth boxes as false negatives
    for gt_box in gt_boxes:
        if not gt_box.get("detected", False):
            false_negatives += 1

    return true_positives, false_positives, false_negatives, detections




truths = {
    # xmin, ymin, xmax, ymax
    "A": [3, 11, 8, 7],
    "B": [13, 6, 17, 2],
    "D": [5, 4, 7, 2]
}

preds = {
    # xmin, ymin, xmax, ymax, confidence
    "A1": [2, 12, 6, 8, 0.82],
    "A2": [5, 14, 9, 10, 0.38],
    "B1": [14, 7, 18, 3, 0.72],
    "B2": [12, 5, 16, 1, 0.83],
    "C1": [1, 6, 4, 3, 0.17],
    "D1": [16, 14, 20, 10, 0.32]
}


truths2 = {
    # xmin, ymin, xmax, ymax
    "A": np.array([3, 7, 8, 11])*30,
    "B": np.array([13, 2, 17, 6])*30,
    "D": np.array([5, 2, 7, 4])*30
}

preds2 = {
    # xmin, ymin, xmax, ymax, confidence
    "A1": np.array([2, 8, 6, 12, 0.82])*30,
    "A2": np.array([5, 10, 9, 14, 0.38])*30,
    "B1": np.array([14, 3, 18, 7, 0.72])*30,
    "B2": np.array([12, 1, 16, 5, 0.83])*30,
    "C1": np.array([1, 3, 4, 6, 0.17])*30,
    "D1": np.array([16, 10, 20, 14, 0.32])*30
}

print("truths", truths2)
print("preds", preds2)
print(3*"\n")
preds2 = [i[:-1] for i in preds2.values()]
# print(preds2)

truths2 = [i for i in truths2.values()]
# print(truths2)

["xmin", "ymin", "xmax", "ymax"],


preds2 = [{"xmin": i[0], "ymin": i[1], "xmax": i[2], "ymax": i[3]} for i in preds2]
truths2 = [{"xmin": i[0], "ymin": i[1], "xmax": i[2], "ymax": i[3]} for i in truths2]


# sample usage
true_positives, false_positives, false_negatives, detections = evaluate_predictions(pred_boxes=preds2, gt_boxes=truths2, iou_threshold=0.2)

print(true_positives, false_positives, false_negatives)

# print(detections)
