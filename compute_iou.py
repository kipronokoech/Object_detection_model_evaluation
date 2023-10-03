def calculate_iou(box1, box2):
    # Extract coordinates of the two boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    ## Calculate the coordinates of the intersection rectangle
    # (x_left, y_top) yield the top-left coordinates for the inersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    # print(f"Top left on intersection: ({x_left}, {y_top})")

    # (x_right, y_botton) yields the bottom-left coordinates for the intersection
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    # print(f"Bottom-right on intersection: ({x_right}, {y_bottom})")
    
    # Check for intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate the areas of the two bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate the area of the intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate the union area
    union_area = (box1_area + box2_area) - intersection_area
    
    # Compute IoU rounded to 3 decimal places
    iou = round(intersection_area / union_area, 8)
    
    return iou

import numpy as np


def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2

    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

if __name__ == "__main__":
    # Given bounding box coordinates (xmin, ymin, xmax, ymax)
    box1 = [390, 300, 510, 420]
    box2 = [420, 270, 540, 390] 

    # Calculate IoU for the two boxes
    iou = calculate_iou(box1, box2)
    print("Intersection over Union:", iou)
    iou2 = box_iou_calc(np.array([box1]), np.array([box2]))
    print(iou2)
