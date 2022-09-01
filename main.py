import cv2
import datetime
import time
import os
from threading import Thread
import argparse

from src.yolov5_grpc import Yolov5_grpc
from src.efnet_grpc import Efnet_grpc


# Capture rtsp stream
class Camera(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None


    def update(self):
        while True:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()


    def read(self):
        if self.status:
            return self.frame
        return None


# Capture webcam or video
class Video_stream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        if src == 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


    def read(self):
        ret, frame = self.cap.read()
        if ret:
            return frame


# Main pipleline
class Pipeline:
    def __init__(self, src, idx, opt):
        self.src = src
        self.idx = idx
        self.opt = opt

        self.images_path_save = 'images'
        self.create_images_folder()
        self.counter = 0

        self.detector = Yolov5_grpc(conf_thresh=detector_tresh)
        self.classifier = Efnet_grpc(conf_thresh=classificator_tresh, classes=classes)
        self.camera = self.get_camera()


    def create_images_folder(self):
        if not os.path.exists(self.images_path_save):
            os.mkdir(self.images_path_save)


    def save_output(self, bbox_id, classifier_res, pred_frame):
        output_path = f'output_{self.idx}_{self.counter}_{bbox_id}_pred_{classes[classifier_res]}.jpeg'
        cv2.imwrite(os.path.join(self.images_path_save, output_path), pred_frame)


    def classifier_pred_loop(self, boxes, frame, classifier, pred_frame):
        for bbox_id, bbox in enumerate(boxes):
            crop = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            classifier_res = classifier.classifier_pred(crop)

            if classifier_res < 2: # small_gun or big_gun
                print(f'Case detected at: {datetime.datetime.now()}')
                self.save_output(bbox_id, classifier_res, pred_frame)


    def get_camera(self):
        if self.opt.src == 'webcam':
            return Video_stream(0)
        if self.opt.src == 'test':
            return Video_stream(self.opt.test_vid_path)

        return Camera(self.src)


    def run(self):
        while True:
            st_time = time.perf_counter()

            frame = self.camera.read()
            boxes, pred_frame, _ = self.detector.get_boxes_debug(frame)
            self.classifier_pred_loop(boxes, frame, self.classifier, pred_frame)

            self.counter += 1
            print('FPS:', round(1 / (time.perf_counter() - st_time), 2))


# Get args
def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--src', type=str, default='rtsp', help='Choose source from rtsp/webcam/test')
    parser.add_argument('--test_vid_path', type=str, default='',
                        help='Path to test video')

    opt = parser.parse_args()
    return opt


# Start process
def main(opt):
    if opt.src in ['webcam', 'test']:
        Pipeline(None, 0, opt).run()
    else:
        for idx, src in enumerate(camera_links):
            pipeline = Pipeline(src, idx, opt)
            Thread(target=pipeline.run).start()


if __name__ == '__main__':
    classes = ['small_gun', 'big_gun', 'phone', 'umbrella', 'empty']
    camera_links = ['rtsp://login:pass@192.168.1.1/ISAPI/Streaming/Channels/101']
    detector_tresh = 0.8
    classificator_tresh = 0.7

    opt = parse_opt()
    main(opt)
