import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(int16)
        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(int16)
        print(labels[:, 1:])
        print(detections[:, :4])
        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        print(all_ious)
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]
        print(all_matches)
        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]
        print(all_matches)

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))

    def plot_matrix(self, cmap='viridis', annot=True, fontsize=12):
        """
        Plot the confusion matrix as a heatmap with custom font size.
        
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



# Dummy detections array (shape: [N, 6])

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


# detections = np.array([
#     [60, 120, 180, 240, 0.82, 0],
#     [100, 135, 260, 250, 0.88, 0],
#     [405, 290, 520, 410, 0.94, 0],
#     [360, 330, 480, 450, 0.67, 0],
#     [480, 60, 600, 180, 0.17, 0],
#     [30, 300, 120, 390, 0.23, 0]
# ])



labels = np.array([[0,331,303,497,469],
[0,385,624,543,782],
[0,809,743,941,875],
[0,883,410,1024,556],
[0,918,287,1024,425],
[0,860,68,976,184],
[0,109,563,217,671],
[0,0,401,60,515],
[0,51,833,207,989],
[0,0,867,80,1024],
[0,273,877,403,1007],
[0,701,939,821,1024],
[0,905,608,1021,724],
[0,471,17,629,175]])

# Dummy labels array (shape: [M, 5])
# labels = np.array([
#     [0, 90, 150, 240, 270],
#     [0, 390, 300, 510, 420],
#     [0, 150, 360, 210, 420]
# ])



# Instantiate ConfusionMatrix with your desired thresholds
confusion_matrix = ConfusionMatrix(num_classes=1, CONF_THRESHOLD=0.0, IOU_THRESHOLD=0.4)

# Process the dummy batch
confusion_matrix.process_batch(detections= detections, labels=labels)

print(confusion_matrix.return_matrix())
# Print or return the confusion matrix
confusion_matrix.print_matrix()

# confusion_matrix.plot_matrix(fontsize=15)

