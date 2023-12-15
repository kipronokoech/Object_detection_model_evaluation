from evaluation import ConfusionMatrix
import numpy as np
import cv2, os
import random
import matplotlib.pyplot as plt

loc = os.path.dirname(__file__)
os.chdir(loc)

import argparse
parser = argparse.ArgumentParser(description="Some description.")

parser.add_argument("-classes", "--num_classes", type=int, default=1, \
                    help="Number of classes in your object detection problem")
parser.add_argument("-conf", "--confidence_threshold", type=float, default=0.5, \
                    help="Confidence score threshold. Any detection who's \
                        confidence is under the value given is skipped during evaluation")
parser.add_argument("-iou", "--iou_threshold", type=float, default=0.5, \
                    help="IoU threshold. Detections with IoU>threshold is regarded as valid\
                        else it is false detection.")
args = parser.parse_args()


# Testing the Evaluation Tool on multiple detection-label pairs
filename = random.choice([i.replace(".txt", "") for i in os.listdir("./examples/preds2")])
pred = f"./examples/preds2/{filename}.txt"
truth = f"./examples/truths/{filename}.txt"

#Loading predictions file into NumPy array
pred = np.genfromtxt(pred, delimiter=',', skip_header=1)
pred2 = pred.copy()
pred2[:, :-2] = pred2[:, :-2].astype(int)

#Loading grouth truth file into NumPy array
truth = np.genfromtxt(truth, delimiter=',', skip_header=1)
truth2 = truth.copy()
truth2[:, :] = truth2[:, :].astype(int)

# Instantiate ConfusionMatrix with your desired thresholds
confusion_matrix = ConfusionMatrix(num_classes=args.num_classes, confidence_threshold=args.confidence_threshold, iou_threshold=args.iou_threshold)
matrix = confusion_matrix.process_detections(detections= pred2, labels=truth2)
confusion_matrix.plot_matrix(fontsize=12)
result = confusion_matrix.compute_tp_fp_fn()

print(result)

# Load two images
image1 = f'examples/plot_truts_and_preds/{filename}.jpg'
image2 = "Output/matrix.png"

from PIL import Image

# Load two images
image1 = Image.open(image1)
image2 = Image.open(image2)

# Determine the target height for resizing
target_height = max(image1.size[1], image2.size[1])

# Resize images to have the same height
image1_resized = image1.resize((int(image1.size[0] * target_height / image1.size[1]), target_height))
image2_resized = image2.resize((int(image2.size[0] * target_height / image2.size[1]), target_height))

# Create a new image with the combined width
total_width = image1_resized.size[0] + image2_resized.size[0]
new_im = Image.new('RGB', (total_width, target_height))

# Paste the resized images horizontally
new_im.paste(image1_resized, (0, 0))
new_im.paste(image2_resized, (image1_resized.size[0], 0))

# Save the result
new_im.save(f'Output/{filename}_merged_image.jpg')

