import cv2
import numpy as np
import os
import glob
from moviepy.editor import VideoFileClip


def load_weights(tiny):
    """
    Load pre-trained weights and labels for YOLOv4 model.

    :param img: tiny or normal
    """ 
    global net
    global layers_names
    if(tiny):
        weights_path = './yolo/yolov4-tiny.weights'
        config_path = './yolo/yolov4-tiny.cfg'
    else:
        weights_path = './yolo/yolov4.weights'
        config_path = './yolo/yolov4.cfg'

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    names = net.getLayerNames()
    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]

    labels_path = './yolo/coco.names'
    labels = open(labels_path).read().strip().split("\n")

def detect_yolo(img):
    """
    Infer the image with YOLOv4 model.

    :param img: input image 
    :return: image annotated with bounding boxes and labels
    """ 
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=False, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(layers_names)
    boxes, confidences, class_ids = [], [], []
    (H, W) = img.shape[:2]
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    idxArr = np.asarray(idxs)
    for n in idxArr.flatten():
        (x, y) = (boxes[n][0], boxes[n][1])
        (w, h) = (boxes[n][2], boxes[n][3])        
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,0), 2)
        text = "{}: {:.4f}".format(labels[class_ids[n]], confidences[n])
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    return img


def pipeline_yolo_only(input_file, output_file, tiny):
    """
    Pipeline for YOLOv4 model.

    :param input_file: input file path 
    :param output_file: output file path
    :param tiny: tiny or normal
    """ 
    load_weights(tiny)
    project_video = VideoFileClip(input_file)
    white_clip = project_video.fl_image(detect_yolo)
    white_clip.write_videofile(output_file, audio=False,threads=8, preset='ultrafast')
