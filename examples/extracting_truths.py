import os, json, cv2

images = os.listdir("./images")
print(images)

annotations = json.load(open("./images/via_project_3Jul2020_10h24m(231)-complete.json"))["_via_img_metadata"]

for id1 in annotations:
    annotation =  annotations[id1]
    filename = annotation["filename"]
    
    if filename in images:
        file = f"./truths/{os.path.splitext(filename)[0] + '.txt'}"
        f = open(file, "w+")
        print(filename)
        image = cv2.imread("./images/"+filename)
        h, w, channel = image.shape
        f.write("x1,y1,x2,y2,class_id\n")
        print(annotation["regions"])
        for region in annotation["regions"]:
            print(annotation)
            cx = region["shape_attributes"]["cx"]
            cy = region["shape_attributes"]["cy"]
            r = round(region["shape_attributes"]["r"])
            print(cx, cy, r)

            x1, y1, x2, y2, class_id = cx-r, cy-r, cx+r, cy+r, 1
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, w), min(y2, h)
            f.write(f"{x1},{y1},{x2},{y2},{class_id}\n")
        f.close()
    
    