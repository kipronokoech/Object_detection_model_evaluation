import numpy as np

def compute_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    print(x_right, x_left, y_bottom, y_top)

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



if __name__ == '__main__':
    print("Testing...")
    factor = 50
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
        "A": [3, 7, 8, 11],
        "B": [13, 6, 17, 2],
        "D": [5, 4, 7, 2]
    }

    preds2 = {
        # xmin, ymin, xmax, ymax, confidence
        "A1": [2, 8, 6, 12, 0.82],
        "A2": [5, 14, 9, 10, 0.38],
        "B1": [14, 7, 18, 3, 0.72],
        "B2": [12, 5, 16, 1, 0.83],
        "C1": [1, 6, 4, 3, 0.17],
        "D1": [16, 14, 20, 10, 0.32]
    }


    iou = compute_iou(box1=np.array(truths["A"])*30, box2=np.array(preds["A2"])*30)
    print(iou)






exit()




def evaluate_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    """Evaluate predicted boxes against ground truth boxes"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # loop over predicted boxes
    detections = []
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]

        # find best matching ground truth box (highest IoU)
        best_iou = 0
        best_gt_box = None
        for j in range(len(gt_boxes)):
            gt_box = gt_boxes[j]
            iou_score = compute_iou(pred_box, gt_box)
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
        detections.append(pred_box + [best_iou])

    # count remaining undetected ground truth boxes as false negatives
    for gt_box in gt_boxes:
        if not gt_box.get("detected", False):
            false_negatives += 1




preds2 = {
        # xmin, ymin, xmax, ymax, confidence
        "A1": [2, 8, 6, 12, 0.82],
        "A2": [5, 14, 9, 10, 0.38],
        "B1": [14, 7, 18, 3, 0.72],
        "B2": [12, 5, 16, 1, 0.83],
        "C1": [1, 6, 4, 3, 0.17],
        "D1": [16, 14, 20, 10, 0.32]
    }

preds2 = [i[:-1] for i in preds2.values()]
print(preds2)

gt_boxes = [
        {"bbox": [3, 11, 8, 7], "detected": False},
        {"bbox": [13, 6, 17, 2], "detected": False},
        {"bbox": [5, 4, 7, 2], "detected": False},
    ]
import numpy as np
true_positives, false_positives, false_negatives = evaluate_predictions(pred_boxes=preds2, gt_boxes=gt_boxes, iou_threshold=0.2)
print(true_positives, false_positives, false_negatives)