import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def compute_iou(boxes1, boxes2):
    """
    This function computes intersection-over-union of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[M, 4])
        boxes2 (Array[N, 4])
    Returns:
        iou (Array[M, N]): the MxN matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # Compute area for for boxes1 and boxes2
    area1 = np.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1)
    area2 = np.prod(boxes2[:, 2:] - boxes2[:, :2], axis=1)

    # Top left and bottom right of the intersection. 
    top_left = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    
    # Compute intersection
    intersection = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return intersection / (area1[:, None] + area2 - intersection)

if __name__ == "__main__":
    # Calling compute_iou with overlapping boxes - expecting IoU>0
    # Make sure that the input is a Numpy array of [M, 4] shape for 
    # detections and [N, 4] for labels. Since we have one par it will be [1,4] for both
    detection = np.array([[859, 31, 1002, 176]])
    label = np.array([[860, 68, 976, 184]])
    iou_value = compute_iou(detection, label)
    print("IoU for intersecting boxes:", iou_value) #Output: 0.5783

    # Calling compute_iou with non-intersecting boxes
    detection = np.array([[810, 744, 942, 865]])
    label = np.array([[109,563,217,671]])
    iou_value = compute_iou(detection, label)
    print("IoU for intersecting boxes:", iou_value) #Output: 0.0

    # We can also pass several pairs of boxes
    detections = np.array([[374,627,538,792],
                        [330,308,501,471],
                        [474,14,638,181],
                        [810,744,942,865],
                        [905,280,1022,425]])
    
    labels = np.array([[331,303,497,469],
                [385,624,543,782],
                [809,743,941,875],
                [883,410,1024,556],
                [918,287,1024,425],
                [860,68,976,184]])

    # Calling compute_iou with non-intersecting boxes
    # The output will be M by N Numpy Array of IoUs. Intersecting pairs
    # will show IoU>0.
    iou_values = compute_iou(detections, labels)
    print("IoU for several pairs of ground truths and detections:", iou_values)

    df = pd.DataFrame(iou_values)
    df.index = range(1, len(detections)+1)
    df.columns = range(1, len(labels)+1)
    plt.figure(figsize=(10, 7))  # Set the figure size
    sns.heatmap(df, annot=True, linewidths=0.5)
    plt.xlabel("Ground Truths")
    plt.ylabel("Predictions")
    plt.title('')
    plt.show()