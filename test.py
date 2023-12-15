import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Display floating-point numbers without scientific notation and display decimals in 2 places
np.set_printoptions(suppress=True, precision=2)

class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    @staticmethod
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
        print(boxes1.shape, boxes2.shape)
        # Compute area for for boxes1 and boxes2
        area1 = np.prod(boxes1[:, 2:] - boxes1[:, :2], axis=1)
        area2 = np.prod(boxes2[:, 2:] - boxes2[:, :2], axis=1)

        # Top left and bottom right of the intersection. 
        top_left = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
        
        # Compute intersection
        intersection = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

        return intersection / (area1[:, None] + area2 - intersection)

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
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
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
        all_ious = self.compute_iou(labels[:, :4], detections[:, :4])
        print(all_ious)
        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            print(gt_class, all_ious[i, :])
            matches = np.where(all_ious[i, :] > self.IOU_THRESHOLD)[0]
            print("matches", matches)
            if len(matches) == 1:
                detection_class = detection_classes[matches[0]]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if len(np.where(all_ious[:, i] > self.IOU_THRESHOLD)[0]) == 0:
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
        # Show the plot
        plt.show()


filename = "_MG_3168_10"
pred = f"./examples/preds2/{filename}.txt"
truth = f"./examples/truths/{filename}.txt"

pred = np.genfromtxt(pred, delimiter=',', skip_header=1)
pred2 = pred.copy()
pred2[:, :-2] = pred2[:, :-2].astype(int)

truth = np.genfromtxt(truth, delimiter=',', skip_header=1)
truth2 = truth.copy()
truth2[:, :-2] = truth2[:, :-2].astype(int)

# Instantiate ConfusionMatrix with your desired thresholds
confusion_matrix = ConfusionMatrix(num_classes=1, CONF_THRESHOLD=0.2, IOU_THRESHOLD=0.5)

# Process detections
matrix = confusion_matrix.process_detections(detections= pred2, labels=truth2)
print(matrix)

confusion_matrix.plot_matrix(fontsize=15)

result = confusion_matrix.compute_tp_fp_fn()

print(result)



# Instantiate ConfusionMatrix with your desired thresholds
confusion_matrix = ConfusionMatrix(num_classes=1, CONF_THRESHOLD=0.2, IOU_THRESHOLD=0.5)

 # Calling compute_iou with overlapping boxes - expecting IoU>0
# Make sure that the input is a Numpy array of [M, 4] shape for 
# detections and [N, 4] for labels. Since we have one par it will be [1,4] for both
detection = np.array([[859, 31, 1002, 176, 0.973, 0]])
label = np.array([[860, 68, 976, 184, 0]])
matrix = confusion_matrix.process_detections(detections= detection, labels=label)
print(matrix)

confusion_matrix.plot_matrix(fontsize=15)
result = confusion_matrix.compute_tp_fp_fn()
print(result)

confusion_matrix = ConfusionMatrix(num_classes=1, CONF_THRESHOLD=0.2, IOU_THRESHOLD=0.5)
# Calling compute_iou with non-intersecting boxes
detection = np.array([[810, 744, 942, 865, 0.9966, 0]])
label = np.array([[109,563,217,671, 0]])
matrix = confusion_matrix.process_detections(detections= detection, labels=label)
print(matrix)
confusion_matrix.plot_matrix(fontsize=15)
result = confusion_matrix.compute_tp_fp_fn()
print(result)



exit()

filename = "_MG_3168_10"
pred = f"./examples/preds2/{filename}.txt"
truth = f"./examples/truths/{filename}.txt"

pred = np.genfromtxt(pred, delimiter=',', skip_header=1)
pred2 = pred.copy()
pred2[:, :-2] = pred2[:, :-2].astype(int)

truth = np.genfromtxt(truth, delimiter=',', skip_header=1)
truth2 = truth.copy()
truth2[:, :-2] = truth2[:, :-2].astype(int)

# Process detections
matrix = confusion_matrix.process_detections(detections= pred2, labels=truth2)
print(matrix)

confusion_matrix.plot_matrix(fontsize=15)

result = confusion_matrix.compute_tp_fp_fn()

print(result)