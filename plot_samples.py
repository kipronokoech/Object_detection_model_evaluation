
import cv2
import numpy as np
from matplotlib import pyplot as plt


height, width = np.array([480, 660])
print(height, width)
channels = 3
color = (50, 50, 50)

# Crate a blank dummy image
image = np.full((height, width, channels), color, dtype="uint8")

truths2 = {
    # xmin, ymin, xmax, ymax
    "A": [90, 150, 240, 270],
    "B": [390, 300, 510, 420],
    "D": [150, 360, 210, 420]
}

preds2 = {
    # xmin, ymin, xmax, ymax, confidence
    "A1": [60, 120, 180, 240, 0.82],
    "A2": [100, 135, 260, 250, 0.38],
    "B1": [405, 290, 520, 410, 0.72],
    "B2": [360, 330, 480, 450, 0.83],
    "C1": [480, 60, 600, 180, 0.17],
    "D1": [30, 300, 120, 390, 0.32]
}

# plot predictions 
for index, (key, value) in enumerate(preds2.items()):
    color_pred = (255, 0, 0)

    confidence = preds2[key][-1]
    value = np.array(list((map(int, value[:-1]))))
    print(key, value, confidence)
    xmin, ymin, xmax, ymax = value
    print(key, xmin, xmax, ymin, ymax)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_pred, thickness=3)
    image = cv2.putText(image, key, (xmin-7, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1, color_pred, 2)

for key, value in truths2.items():
    color_pred = (0, 255, 0)
    
    value = np.array(list((map(int, value))))
    print(key, value)
    xmin, ymin, xmax, ymax = value
    print(key, xmin, xmax, ymin, ymax)
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_pred, thickness=3)
    image = cv2.putText(image, key, (xmin-7, ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 1, color_pred, 2)

plt.figure(figsize=(11, 8))
plt.imshow(image)
plt.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("./samples.png", image)