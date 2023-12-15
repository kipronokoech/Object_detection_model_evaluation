import numpy as np
from compute_iou import calculate_iou

def calculate_confusion_matrix(truths, preds):
    # Initialize confusion matrix values
    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    # Iterate through ground truth
    for truth_key, truth_box in truths.items():
        # Check if there is a corresponding prediction
        pred_key = truth_key + '1'  # Assuming predictions have the same naming convention
        if pred_key in preds:
            pred_box = preds[pred_key]

            # Calculate IoU for this pair of boxes
            iou = calculate_iou(pred_box, truth_box)

            # Determine if it's a True Positive or False Positive
            if iou >= 0.5:  # You can adjust the threshold as needed
                TP += 1
            else:
                FP += 1

        else:
            # If there is no corresponding prediction, it's a False Negative
            FN += 1

    return TP, FP, FN

truths2 = {
    # Box keys and coordinates in the format [xmin, ymin, xmax, ymax]
    "A": [90, 150, 240, 270],
    "B": [390, 300, 510, 420],
    "D": [150, 360, 210, 420]
}

preds2 = {
    # Box keys and coordinates in the format [xmin, ymin, xmax, ymax, confidence]
    "A1": [60, 120, 180, 240, 0.82],
    "A2": [100, 135, 260, 250, 0.38],
    "B1": [405, 290, 520, 410, 0.72],
    "B2": [360, 330, 480, 450, 0.83],
    "C1": [480, 60, 600, 180, 0.17],
    "D1": [30, 300, 120, 390, 0.32]
}
# Calculate confusion matrix values
TP, FP, FN = calculate_confusion_matrix(truths2, preds2)

# Print the confusion matrix values
print(f'TP (True Positives): {TP}')
print(f'FP (False Positives): {FP}')
print(f'FN (False Negatives): {FN}')




