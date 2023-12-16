import numpy as np
import matplotlib.pyplot as plt
from compute_iou2 import compute_iou
import seaborn as sns
# Display floating-point numbers without scientific notation and display decimals in 2 places
np.set_printoptions(suppress=True, precision=2)

import os
loc = os.path.dirname(__file__)
os.chdir(loc)

class ConfusionMatrix:
    def __init__(self, num_classes: int, confidence_threshold=0.3, iou_threshold=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

    def process_detections(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), x1, y1, x2, y2, class
        Returns:
            None, updates confusion matrix accordingly
        """
        # Extract class ID from the last column of the labels array
        gt_classes = labels[:, -1].astype(int) 

        try:
            # Pick detections 
            detections = detections[detections[:, 4] > self.confidence_threshold]
        except IndexError or TypeError:
            print("Got_here")
            # No detections, processing the ground truths
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            # terminate processing by returning confusion matrix in the current state.
            return self.matrix

         # Extract class IDs for the detections - fifth column of the 
        detection_classes = detections[:, 5].astype(int)

        # Calculate IoU for all pairs of ground truths and detections
        all_ious = compute_iou(labels[:, :4], detections[:, :4])

        # Loop through all ground truths and merge them up with the detections
        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            # filters all pairs with IoU> threshold
            matches = np.where(all_ious[i, :] > self.iou_threshold)[0]

            if len(matches) == 1:
                # Count 1 match. If there are multiple matches count others as false positives
                detection_class = detection_classes[matches[0]]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        # Count detections that we missed by the model
        for i, detection in enumerate(detections):
            if len(np.where(all_ious[:, i] > self.iou_threshold)[0]) == 0:
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1
            
        return self.matrix

    def compute_tp_fp_fn(self):
        """
        Compute True Positives (TP), False Positives (FP), and False Negatives (FN) for each class.
        Returns:
            tp_fp_fn (dict): A dictionary where keys are class IDs and values are tuples (TP, FP, FN).
        """
        tp_fp_fn = {}
        for class_id in range(self.num_classes):
            # TP: Diagonal element of the confusion matrix for the class
            tp = self.matrix[class_id, class_id]

            # FNs and FPs
            fn = np.sum(self.matrix[:, class_id]) - tp
            fp = np.sum(self.matrix[class_id, :]) - tp

            tp_fp_fn[class_id] = {"TPs": tp, 
                                    "FPs": fp, 
                                        "FNs": fn}
        return tp_fp_fn

    def plot_matrix(self, cmap='viridis', annot=True, fontsize=12):
        """
        Plots the confusion matrix as a heatmap.
        
        Args:
            cmap (str, optional): Colormap for the heatmap. Defaults to 'viridis'.
            annot (bool, optional): Whether to display values in each cell. Defaults to True.
            fontsize (int, optional): Font size for the heatmap. Defaults to 12.
        """
        # Initialize a figure and axis for plotting
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(self.matrix, cmap=cmap, annot=annot, fmt='g', square=True, cbar=True, annot_kws={"size": fontsize})

        # Set font size for axis labels and title
        ax.set_xlabel('Predicted', fontsize=fontsize)
        ax.set_ylabel('True', fontsize=fontsize)
        ax.set_title('Confusion Matrix', fontsize=fontsize + 2)
        plt.savefig("output/matrix.png", bbox_inches="tight", pad_inches=0.1)
        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Instantiate ConfusionMatrix with your desired thresholds
    confusion_matrix = ConfusionMatrix(num_classes=1, confidence_threshold=0.8, iou_threshold=0.5)
    detection = np.array([[859, 31, 1002, 176, 0.973, 0]])
    label = np.array([[860, 68, 976, 184, 0]])
    matrix = confusion_matrix.process_detections(detections= detection, labels=label)
    confusion_matrix.plot_matrix(fontsize=15)
    result = confusion_matrix.compute_tp_fp_fn()
    print(result)

    # Testing the Evaluation Tool on Non-Overlapping boxes
    confusion_matrix = ConfusionMatrix(num_classes=1, confidence_threshold=0.8, iou_threshold=0.5)
    detection = np.array([[810, 744, 942, 865, 0.9966, 0]])
    label = np.array([[109,563,217,671, 0]])
    matrix = confusion_matrix.process_detections(detections= detection, labels=label)
    confusion_matrix.plot_matrix(fontsize=15)
    result = confusion_matrix.compute_tp_fp_fn()
    print(result)

    # Run the Evaluation Tools on multiple detection-labels pairs
    confusion_matrix = ConfusionMatrix(num_classes=1, confidence_threshold=0.8, iou_threshold=0.5)

    detections = np.array([[374,627,538,792,0.9996,0],
    [330,308,501,471,0.9994,0],
    [474,14,638,181,0.9992,0],
    [810,744,942,865,0.9966,0],
    [58,844,204,993,0.9965,0],
    [905,280,1022,425,0.9881,0],
    [887,412,1018,543,0.9811,0],
    [0,871,68,1008,0.9759,0],
    [859,31,1002,176,0.973,0],
    [698,949,808,1023,0.9303,0],
    [0,400,47,505,0.9203,0],
    [234,0,314,58,0.8163,0]])


    labels = np.array([[331,303,497,469, 0],
    [385,624,543,782, 0],
    [809,743,941,875, 0],
    [883,410,1024,556, 0],
    [918,287,1024,425, 0],
    [860,68,976,184, 0],
    [109,563,217,671, 0],
    [0,401,60,515, 0],
    [51,833,207,989, 0],
    [0,867,80,1024, 0],
    [273,877,403,1007, 0],
    [701,939,821,1024, 0],
    [905,608,1021,724, 0],
    [471,17,629,175, 0]])
    matrix = confusion_matrix.process_detections(detections= detections, labels=labels)
    confusion_matrix.plot_matrix(fontsize=15)
    result = confusion_matrix.compute_tp_fp_fn()
    print(result)

