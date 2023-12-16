from evaluation import ConfusionMatrix
import numpy as np
import cv2, os
import random
import matplotlib.pyplot as plt

loc = os.path.dirname(__file__)
os.chdir(loc)


# Testing the Evaluation Tool on multiple detection-label pairs
filename = "_MG_3168_10"
pred = f"./examples/preds2/{filename}.txt"
truth = f"./examples/truths/{filename}.txt"

#Loading predictions file into NumPy array
pred = np.genfromtxt(pred, delimiter=",", skip_header=1)
pred2 = pred.copy()
# pred2[:, :-2] = pred2[:, :-2].astype(int)

#Loading grouth truth file into NumPy array
truth = np.genfromtxt(truth, delimiter=",", skip_header=1)
truth2 = truth.copy()
# truth2[:, :] = truth2[:, :].astype(int)

# Instantiate ConfusionMatrix with your desired thresholds
confusion_matrix = ConfusionMatrix(num_classes=1, confidence_threshold=0.8, iou_threshold=0.5)
matrix = confusion_matrix.process_detections(detections= pred2, labels=truth2)
confusion_matrix.plot_matrix(fontsize=12)
result = confusion_matrix.compute_tp_fp_fn()

print(result)