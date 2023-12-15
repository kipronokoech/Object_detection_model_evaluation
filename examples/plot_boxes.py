import cv2
import numpy as np
from matplotlib import pyplot as plt
import os, json

loc = os.path.dirname(__file__)
os.chdir(loc)

for image1 in os.listdir("./images"):
    if "via_project_3Jul2020_10h24m(231)-complete" in image1:
        # Skip annotations
        continue
    image = cv2.imread(f"./images/{image1}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    color_pred = (0, 0, 255)
    image = cv2.putText(image, image1, (30, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 150, 150), 3)
    file_truth = f"./truths/{os.path.splitext(image1)[0] + '.txt'}"
    print(file_truth)
    with open(file_truth) as f:
        first_line = next(f)  # Skip the first line
        for line in f:
            line = list(map(int, line.strip().split(",")))
            print(line)
            x1, y1, x2, y2, class_id = line

            image = cv2.rectangle(image, (x1, y1), (x2, y2), color_pred, thickness=5)
    
    file_pred = f"./preds/{os.path.splitext(image1)[0] + '.json'}"
    pred = json.load(open(file_pred))
    
    f1 = open(f"./preds2/{os.path.splitext(image1)[0] + '.txt'}", "w+")
    f1.write("x1,y1,x2,y2,confidence,class_id\n")
    for index, box in enumerate(pred[0]["rois"]):
        x1, y1, x2, y2 = box[1], box[0], box[3], box[2] # Mask R-CNN output (y_min, x_min, y_max, x_max)
        confidence, class_id = pred[0]["scores"][index], pred[0]["class_ids"][index]-1
        f1.write(f"{x1},{y1},{x2},{y2},{round(confidence, 4)},{class_id}\n")
        color_pred = (255, 0, 0)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color_pred, thickness=5)
    f1.close()



    plt.figure(figsize=(12,12))
    plt.imshow(image)
    plt.savefig(f"./plot_truts_and_preds/{image1}", bbox_inches="tight", pad_inches=0.1)
    # plt.show()        


# image = cv2.imread("./images/_MG_3168_10.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# h, w, channels = image.shape
# print(image.shape)
# color_pred = (255, 0, 0)
# xmin, ymin, xmax, ymax = 701, 939, 821, 1059

# xmin, ymin = max(xmin, 0), max(ymin, 0)
# xmax, ymax = min(xmax, w), min(ymax, h)

# image =  cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color_pred, thickness=3)


