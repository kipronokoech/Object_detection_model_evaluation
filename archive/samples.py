
import cv2
import numpy as np
from matplotlib import pyplot as plt

factor = 30
height, width, = np.array([16, 22]) * factor 
print(height, width)
channels = 3
color = (0, 0, 0)

# Crate a blank dummy image
image = np.full((height, width, channels), color, dtype="uint8")

truths = {
    # xmin, ymin, xmax, ymax
    "A": [3, 11, 8, 7],
    "B": [13, 6, 17, 2],
    "D": [5, 4, 7, 2]
}

preds = {
    # xmin, ymin, xmax, ymax, confidence
    "A1": [2, 12, 6, 8, 0.82],
    "A2": [5, 14, 9, 10, 0.38],
    "B1": [14, 7, 18, 3, 0.72],
    "B2": [12, 5, 16, 1, 0.83],
    "D1": [1, 6, 4, 3, 0.17],
    "C1": [16, 14, 20, 10, 0.32]
}
# plot predictions 
for index, (key, value) in enumerate(preds.items()):
    color_pred = (255, 0, 0)

    confidence = preds[key][-1]
    value = np.array(list((map(int, value[:-1]))))*factor
    print(key, value, confidence)
    xmin, xmax, ymin, ymax = value
    print(key, xmin, xmax, ymin, ymax)
    image = cv2.rectangle(image, (xmin, height-xmax), (ymin, height-ymax), color_pred, thickness=3)
    image = cv2.putText(image, key, (xmin-5, height-xmax-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_pred, 2)

for key, value in truths.items():
    color_pred = (0, 255, 0)
    
    value = np.array(list((map(int, value))))*factor
    print(key, value)
    xmin, xmax, ymin, ymax = value
    print(key, xmin, xmax, ymin, ymax)
    image = cv2.rectangle(image, (xmin, height-xmax), (ymin, height-ymax), color_pred, thickness=3)
    image = cv2.putText(image, key, (xmin-5, height-xmax-5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_pred, 2)

plt.figure(figsize=(11, 8))
plt.imshow(image)
plt.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite("./samples.png", image)