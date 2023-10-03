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


aa = compute_iou([36, 6, 42, 14], [28, 0, 40, 8])
print(aa)


bb = compute_iou([3, 7, 8, 11], [2, 8, 6, 12])
print(bb)

exit()
def compute_iou2(box1, box2):


    print(box1)
    print(box2)
    return None


aa = compute_iou2([36, 6, 42, 14], [28, 0, 40, 8])
print(aa)