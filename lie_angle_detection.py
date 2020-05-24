import cv2
import numpy as np
import os
import math
import time


def getAngle(a, b, c):
    ang = math.degrees(math.atan2(
        c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def draw_line_shaft(img):
    h, w = img.shape[0:2]
    print(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 255, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                            minLineLength=80, maxLineGap=100)
    good = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
        good.append(line[0])
    # cv2.imwrite("tmp1.jpg", img)
    return good[0][0], good[0][1], good[0][2], good[0][3]


def draw_line_horizontal(img):
    h, w = img.shape[0:2]
    tmp_img = img.copy()
    print(h, w)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 255, 255)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40,
                            minLineLength=50, maxLineGap=30)
    good = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        center_x, center_y = int(x1/2 + x2/2), int(y1/2 + y2/2)
        # Only take the line horizontally and in the center of the image.
        if abs(y2-y1) < 10 and center_x > int(w/2 - 50) and center_x < int(w/2 + 50) and center_y > int(h/2 - 50) and center_y < int(h/2 + 50):
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
            good.append(line[0])
    # cv2.imwrite("tmp2.jpg", img)
    cv2.line(tmp_img, (0, good[0][1]), (w, good[0][3]), (128, 0, 128), 2)
    angle = getAngle((0, good[0][1]), (w, good[0][3]), (0, good[0][3]))
    angle = angle if angle < 180 else 360 - angle
    print(angle)
    cv2.putText(tmp_img, str(round(angle, 2)), (int(w/2), int(h/2)),
                cv2.FONT_HERSHEY_PLAIN, 5, (128, 0, 128), 2)
    return tmp_img

def load_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1]
                     for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def load_image(img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    return img, height, width, channels

def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.5:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 1)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    return img

def image_detect(img_path):
    model, classes, colors, output_layers = load_yolo(
        './model/yolo-putter-tiny_10000.weights', './model/yolo-putter-tiny.cfg', './model/putter.names')
    image, height, width, channels = load_image(img_path)
    tmp_image = image.copy()
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    x, y, w, h = boxes[0]
    # image = draw_labels(boxes, confs, colors, class_ids, classes, image)
    image = draw_line_horizontal(image)
    crop_img = tmp_image[y:y+h, x:x+w]
    # get line for shaft in bounding box
    _xmin, _ymin, _xmax, _ymax = draw_line_shaft(crop_img)
    # translating axis
    xmin, ymin = line_intersection(
        ((x + _xmin, y + _ymin), (x + _xmax, y + _ymax)), ((0, 0), (width, 0)))
    xmax, ymax = line_intersection(
        ((x + _xmin, y + _ymin), (x + _xmax, y + _ymax)), ((0, height), (width, height)))
    xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

    cv2.line(image, (xmin, ymin),
             (xmax, ymax), (0, 255, 255), 2)
    angle = getAngle((xmin, ymin), (xmax, ymax), (0, height))
    angle = angle if angle < 180 else 360 - angle
    print(angle)
    cv2.putText(image, str(round(angle, 2)), (int(width/3), int(height - 30)),
                cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 2)
    cv2.imwrite('impp1.jpg', image)

if __name__ == "__main__":
    image_detect('img13.jpg')