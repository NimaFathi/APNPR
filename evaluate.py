import argparse
import math

import cv2
import easyocr
import numpy as np


def parse_option():
    parser = argparse.ArgumentParser('arguments for predictions')
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--pth_image', type=str, default="./images/5.jpg")
    parser.add_argument('--output_plate_name', type=str, default='cropped.jpg')
    parser.add_argument('--pth_weights', type=str, default='./cfg/model.weights')
    parser.add_argument('--pth_cfg', type=str, default='./cfg/yolov3-custom.cfg')
    parser.add_argument('--pth_classes', type=str, default='./cfg/classes.txt')

    opt = parser.parse_args()
    return opt


def cleanup_text(text):
    text.strip()
    if len(text) > 11 or len(text) < 6:
        text = 'پلاک قابل بازیابی نبود'
    return text


class LicensePlateDetector:
    def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None

    def detect(self, img_path: str):

        orig = cv2.imread(img_path)
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_outputs:
            for detection in output:

                scores = detection[5:]

                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.08:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0, 0.1)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x, y), (x + w, y + h), self.color, 15)
                cv2.putText(img, label + ' ' + confidence, (x, y + 20), self.font, 3, (255, 255, 255), 3)
            self.fig_image = img
            self.coordinates = (x, y, w, h)
        else:
            self.fig_image = img
            self.coordinates = (0, 0, 5, 5)
        return

    def crop_plate(self):
        x, y, w, h = self.coordinates
        roi = self.img[y:y + h, x:x + w]
        self.roi_image = roi
        return


if __name__ == '__main__':
    opt = parse_option()
    lpd = LicensePlateDetector(
        pth_weights=opt.pth_weights,
        pth_cfg=opt.pth_cfg,
        pth_classes=opt.pth_classes
    )

    # Detect license plate
    lpd.detect(opt.pth_image)

    # Crop plate and show cropped plate
    lpd.crop_plate()
    cv2.imwrite(opt.output_plate_name, cv2.cvtColor(lpd.roi_image, cv2.COLOR_BGR2RGB))
    image = cv2.imread(opt.output_plate_name)
    outputs = []
    reader = easyocr.Reader(['fa', 'ar'], gpu=opt.gpu)
    result = reader.readtext(image, detail=1)
    for (bbox, text, prob) in result:
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        br = (int(br[0]), int(br[1]))
        size = math.sqrt(math.pow(br[0] - tl[0], 2) + math.pow(br[1] - tl[0], 2))
        text = cleanup_text(text)
        if prob > 0.2:
            outputs.append([text, size])
    if len(outputs) > 0:
        print(sorted(outputs, key=lambda x: int(x[1]), reverse=True)[0][0])
    else:
        print('پلاک قابل بازیابی نبود')
