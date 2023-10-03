import numpy as np

def calculate_iou(box1, box2):
    # Extract coordinates of the two boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    ## Calculate the coordinates of the intersection rectangle
    # (x_left, y_top) yield the top-left coordinates for the inersection
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    print(f"Top left on intersection: ({x_left}, {y_top})",)

    # (x_right, y_botton) yields the bottom-left coordinates for the intersection
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    print(f"Bottom-right on intersection: ({x_right}, {y_bottom})",)
    
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
    
    # Calculate the IoU
    iou = round(intersection_area / union_area, 3)
    
    return iou

# Given bounding box coordinates (xmin, ymin, xmax, ymax)
box1 = [390, 300, 510, 420]
box2 = [420, 270, 540, 390] 

# Calculate IoU
iou = calculate_iou(box1, box2)
print("Intersection over Union:", iou)

exit()
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    print()
    # return the intersection over union value
    return round(iou, 3)


iou = bb_intersection_over_union(box1, box2)
print(iou)





# ChatGPT
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
    box1 (tuple): Coordinates of the first box in the format (x1, y1, x2, y2).
    box2 (tuple): Coordinates of the second box in the format (x1, y1, x2, y2).

    Returns:
    float: IoU score between the two boxes.
    """
    # Calculate the intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of intersection
    intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)

    # Calculate the area of each box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the Union area
    union_area = float(area_box1 + area_box2 - intersection_area)
    print(union_area)
    # Calculate IoU
    iou = intersection_area / union_area

    return iou

# Example usage
box1 = [90, 150, 240, 270] #[90, 210, 240, 330]
box2 = [150, 60, 270, 180] #[150, 300, 270, 420]
iou_score = calculate_iou(box1, box2)
print("IoU:", iou_score)


