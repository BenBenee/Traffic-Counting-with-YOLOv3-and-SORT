# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import datetime
import os
from yolo import YOLO 
from PIL import Image, ImageDraw, ImageFont
from sort import *


def main(yolo):
    tracker = Sort()
    memory = {}
    line1 = [(455, 384), (827, 384)] #Put your lines coordinate here
    line2 = [(879,523), (1641,747)]
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.4,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    # Return true if line segments AB and CD intersect
    def intersect(A,B,C,D):
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

    def ccw(A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # load the COCO class labels our YOLO model was trained on
    #labelsPath = os.path.sep.join([args["yolo"], "classes.names"])
    #LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture('video.avi')
    writer = None
    (W, H) = (None, None)
    
    font = ImageFont.truetype(font='/font/FiraMono-Medium.otf', size=40)
    
    frameIndex = 0
    car = 0
    motor = 0
    bus = 0
    truck = 0
    car2 = 0 
    motor2 = 0
    bus2 = 0
    truck2 = 0
    # loop over frames from the video file stream
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
  
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break

        image = Image.fromarray(frame[...,::-1])
        boxes, out_class, confidences, midPoint = yolo.detect_image(image)
        image = np.asarray(image)

        # and class IDs, respectively
        classIDs = []
        #classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)

        dets = []
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        dets = np.asarray(dets)
        tracks = tracker.update(dets)

        boxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}
        
        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = boxes[-1]

        if len(boxes) > 0:
            i = int(0)
            for box in boxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    cv2.line(frame, p0, p1, color, 3)

                    if intersect(p0, p1, line1[0], line1[1]):
                        detected_class = yolo.counter(p0,out_class, midPoint)
                        if detected_class == 'car':
                            car = car + 1
                        elif detected_class == 'motorbike':
                            motor = motor+1
                        elif detected_class == 'bus':
                            bus = bus+1
                        else:
                            truck = truck + 1
                            
                    if intersect(p0, p1, line2[0], line2[1]):
                        detected_class = yolo.counter(p0,out_class, midPoint)
                        if detected_class == 'car':
                            car2 = car2 + 1
                        elif detected_class == 'motorbike':
                            motor2 = motor2+1
                        elif detected_class == 'bus':
                            bus2 = bus2+1
                        else:
                            truck2 = truck2 + 1
                            
                            
                # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1
                
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        draw.text((83,152), 'car %d \nmotor %d \nbus %d \ntruck %d' %(car,motor,bus,truck), fill=(255, 255, 255),font=font)
        draw.text((1328,112), 'car %d \nmotor %d \nbus %d \ntruck %d' %(car2,motor2,bus2,truck2), fill=(255, 255, 255),font=font)
        
        frame = np.asarray(frame)
        # draw line
        cv2.line(frame, line1[0], line1[1], (0, 255, 255), 5)
        cv2.line(frame, line2[0], line2[1], (0, 255, 255), 5)

        # draw counter
        # counter += 1

        # check if the video writer is None
        if writer is None:
            # initialize our video writer
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_fps = vs.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, 'fps: %d' %(video_fps), (9,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 5)
            writer = cv2.VideoWriter('output.avi', fourcc, video_fps, (frame.shape[1], frame.shape[0]), True)

        # write the output frame to disk
        writer.write(frame)

        # increase frame index
        frameIndex += 1

    # release the file pointers
    print("[INFO] cleaning up...")
    writer.release()
    vs.release()

if __name__ == '__main__':
    main(YOLO())
