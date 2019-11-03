# import the necessary packages
import tensorflow as tf
from yolov3.models import YoloV3
from absl import app, logging
import numpy as np
from imutils.video import VideoStream
import argparse
from flask import Flask, render_template
import socketio
import eventlet

# setup program arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--classes", default="./data/coco.names", help="path to classes file")
ap.add_argument("-w", "--weights", default="./data/yolov3.tf", help="path to weights file")
ap.add_argument("-s", "--size", type=int, default=512, help="resize images to")
ap.add_argument("-n", "--num_classes", type=int, default=80, help="number of classes in the model")
args = vars(ap.parse_args())

# initialize global detection parameters, we'll set these when we need to.
yolo = class_names = vid = W = H = 0

# SocketIO setup
sio = socketio.Server(async_mode="eventlet", cors_allowed_origins='*', logger=False)
appweb = Flask(__name__)
appweb.wsgi_app = socketio.WSGIApp(sio, appweb.wsgi_app)
thread = None

def background_thread():
    global yolo, class_names, vid, W, H

    while True:
        # sleep 100ms to prevent socketIO overloading
        sio.sleep(0.1)

        # Read video frame
        img = vid.read()
        img = img[1] if args.get("input", False) else img

        # Check if video frame contains data, this to make sure the camera feed is active
        if img is None:
            logging.warning("Empty Frame")
            continue

	    # resize the image to have a maximum width of $ pixels (the less data we have, the faster we can process it), 
        img_in = tf.expand_dims(img, 0)
        img_in = transform_images(img_in, args["size"])

        # if the image dimensions are empty, set them
        if W is None or H is None:
            (H, W) = img_in.get_shape()[1:3]

		# obtain the detections
        boxes, scores, classes, nums = yolo.predict(img_in)
        
        # prepare detection data for socketio sending
        preparedData = {
            "gridLength": int(W), 
            "gridHeight": int(H), 
            "gridDetections": calculate_output_points(img, (boxes, scores, classes, nums), class_names),
        }

        # send detection data to subscribed users on socket
        sio.emit('message', preparedData)
        # logging.info("SocketIO message emitted")

    # stop camera feed
    vid.stop()

def transform_images(x_train, size):
    return (tf.image.resize(x_train, (size, size))) / 255

def calculate_midpoint(p1, p2):
    return tuple((int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)))

def calculate_output_points(img, outputs, class_names):
    # prepare data for iterations
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    sendingDetectiondata = []

    for i in range(nums):
        # calculate top left and bottom right locations
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))

        # calculate middle of detection field
        center = calculate_midpoint(x1y1, x2y2)

        #construct detection object
        sendingDetectiondata.append({
            "topleft": [int(x1y1[0]), int(x1y1[1])],
            "bottomright": [int(x2y2[0]), int(x2y2[1])],
            "center": [center[0], center[1]],
            "type": class_names[int(classes[i])],
            "prediction": '{:.4f}'.format(objectness[i]),
        })

    return sendingDetectiondata

@appweb.route('/start')
def index():
    global thread
    if thread is None:
        thread = sio.start_background_task(background_thread)
    return "Detection server started"

@sio.event
def connect(sid, environ):
    logging.info('Client connected: {}'.format(sid))

@sio.event
def disconnect(sid):
    logging.info('Client disconnected: {}'.format(sid))

@sio.event
def disconnect_request(sid):
    sio.disconnect(sid)

def main(_argv):
    global yolo, class_names, vid, W, H

    # initialize YoloV3 with the provided weights file
    yolo = YoloV3(classes=args["num_classes"])
    yolo.load_weights(args["weights"])
    logging.info('Weights loaded')

    # initialize the list of class labels for the weight to detect
    class_names = [c.strip() for c in open(args["classes"]).readlines()]
    logging.info('Classes loaded')

    # grab a reference to the webcam
    vid = VideoStream(src=0).start()        
    logging.info('video stream started')

    # initialize image dimensions (we'll set them as soon as we read the first image from the stream)
    H = W = None

    # startup the SocketIO server
    eventlet.wsgi.server(eventlet.listen(('', 8080)), appweb)

if __name__ == '__main__':
    app.run(main)