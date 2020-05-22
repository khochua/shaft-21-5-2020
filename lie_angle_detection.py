import cv2
import numpy as np
import os
import math

def load_yolo(weights_path, cfg_path, names_path):
    net = cv2.dnn.readNet(weights_path, cfg_path)
    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readline()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1]
                     for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

def load_image(img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    return img, height, width, channels

def get_box_dimensions(outputs, height, width):
    boxes = []
    conft = []
    class_ids = []
    for output in outputs:
        scores = detect[5:]
        class_id = np.argmax(scores)
        conf = scores[class_id]
        if conf > 0.5:
            center_x = int(detect[0]*width)
            center_y = int(detect[1]*height)
            


def detect_shaft(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(
        416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs

if __name__ == "__main__":
    image_detect('1.jpg')
