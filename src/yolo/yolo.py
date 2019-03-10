#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################

import os

import cv2
import argparse
import numpy as np
import _pickle as pickle

ap = argparse.ArgumentParser()
ap.add_argument('-if', '--image_folder', required=True,
                help = 'path to input image folder')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-s', '--save_as', required=True,
                help = 'pickle file to save detections in')

args = ap.parse_args()

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

ndct = {'x':[], 'y':[], 'w':[], 'h':[], 'fr':[], 'r':[]}
net = cv2.dnn.readNet(args.weights, args.config)

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def predict(image, image_id):
    global ndct, args, net

    image = cv2.imread(image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    #net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        if True: #class_id[i] == 0:
            box = boxes[i]
            confidence = confidences[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            ndct['x'].append(x)
            ndct['y'].append(y)
            ndct['w'].append(w)
            ndct['h'].append(h)

            ndct['fr'].append(image_id)
            ndct['r'].append(confidence)



files = sorted(os.listdir(args.image_folder))
for file_id, filename in enumerate(files):
    predict(os.path.join(args.image_folder, filename), file_id+1)
    print(filename, end='\r')
pickle.dump(ndct, open(args.save_as, 'wb+'))



