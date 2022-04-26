import cv2
import numpy as np

from classes_dict import *


def detect_image(image, model, allowed_names):
    if allowed_names is not None:
        classes = []
        names = allowed_names[0].split(',')
        for name in names:
            classes.append(class_names[name])
            model.classes = classes
    else:
        model.classes = None

    results = model(image)
    preds = results.pandas().xyxy[0]

    boxes = []
    for box in zip(preds['xmin'], preds['ymin'], preds['xmax'], preds['ymax']):
        boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
    names = preds['name']

    for box, name in zip(boxes, names):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Prints the text.
        image = cv2.rectangle(image, (x1, y1 - 20),
                              (x1 + w, y1), (0, 255, 0), -1)
        image = cv2.putText(image, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    if image.shape[0] > 1000 or image.shape[1] > 1000:
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imwrite('temp.png', image)
