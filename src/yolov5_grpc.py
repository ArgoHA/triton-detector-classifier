# This script is based on different grpc examples for triton server
import tritonclient.grpc as grpcclient
from typing import List
import numpy as np
import cv2
import sys
import time

IOU_THRESHOLD = 0.45


class Yolov5_grpc():
    def __init__(self,
                 url="localhost:8001",
                 model_name="yolov5",
                 input_width=640,
                 input_height=640,
                 model_version="",
                 verbose=False, conf_thresh=0.8) -> None:
        super(Yolov5_grpc).__init__()
        self.model_name = model_name

        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = 1
        self.conf_thresh = conf_thresh
        self.input_shape = [self.batch_size, 3, self.input_height, self.input_width]
        self.input_name = 'images'
        self.output_name = 'output'
        self.output_size = 25200
        self.triton_client = None
        self.init_triton_client(url)
        self.test_predict()


    def init_triton_client(self, url):
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()
        self.triton_client = triton_client


    def show_stats(self):
        statistics = self.triton_client.get_inference_statistics(model_name=self.model_name)
        print(statistics)
        if len(statistics.model_stats) != 1:
            print("FAILED: Inference Statistics")
            sys.exit(1)


    def test_predict(self):
        input_images = np.zeros((*self.input_shape,), dtype=np.float32)
        _ = self.predict(input_images)


    def predict(self, input_images):
        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput(self.input_name, [*input_images.shape], "FP32"))
        # Initialize the data
        inputs[-1].set_data_from_numpy(input_images)
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        # Test with outputs
        results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs)

        # Get the output arrays from the results
        return results.as_numpy(self.output_name)


    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


    def draw_boxes(self, image, coords, scores):
        box_color = (51, 51, 255)
        font_color = (255, 255, 255)

        line_width = max(round(sum(image.shape) / 2 * 0.0025), 2)
        font_thickness = max(line_width - 1, 1)
        draw_image = image.copy()

        if coords and len(coords):
            for idx, tb in enumerate(coords):
                if tb[0] >= tb[2] or tb[1] >= tb[3]:
                    continue
                obj_coords = list(map(int, tb[:4]))

                # bbox
                p1, p2 = (int(obj_coords[0]), int(obj_coords[1])), (int(obj_coords[2]), int(obj_coords[3]))
                cv2.rectangle(draw_image, p1, p2, box_color, thickness=line_width, lineType=cv2.LINE_AA)

                # Conf level
                label = str(int(round(scores[idx], 2) * 100)) + '%'
                w, h = cv2.getTextSize(label, 0, fontScale=2, thickness=3)[0]  # text width, height
                outside = obj_coords[1] - h - 3 >= 0  # label fits outside box

                w, h = cv2.getTextSize(label, 0, fontScale=line_width / 3, thickness=font_thickness)[0]  # text width, height
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3

                cv2.rectangle(draw_image, p1, p2, box_color, -1, cv2.LINE_AA)  # filled
                cv2.putText(draw_image,
                            label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            line_width / 3,
                            font_color,
                            thickness=font_thickness,
                            lineType=cv2.LINE_AA)

        return draw_image


    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou


    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > conf_thresh
        boxes = prediction[prediction[:, 4] >= conf_thres]

        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h - 1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = np.round(boxes[0, -1]) == np.round(boxes[:, -1])
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes


    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_width / origin_w
        r_h = self.input_height / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_height - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_height - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_width - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_width - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y


    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Do nms
        boxes = self.non_max_suppression(output, origin_h, origin_w, conf_thres=self.conf_thresh, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid


    def preprocess(self, img, stride):
        img = self.letterbox(img, max(self.input_width, self.input_height), stride=stride, auto=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.reshape([1, *img.shape])
        return img


    def postprocess(self, host_outputs, batch_origin_h, batch_origin_w, min_accuracy=0.5):
        output = host_outputs[0]
        # Do postprocess
        answer = []
        valid_scores = []
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * self.output_size: (i + 1) * self.output_size], batch_origin_h, batch_origin_w
            )
            for box, score in zip(result_boxes, result_scores):
                if score > min_accuracy:
                    answer.append(box)
                    valid_scores.append(score)
        return answer, valid_scores


    def grpc_detect(self, image: np.ndarray, stride: int = 32, min_accuracy: float = 0.5) -> List:
        processed_image = self.preprocess(image, stride)
        pred = self.predict(processed_image)
        boxes, scores = self.postprocess(pred, image.shape[0], image.shape[1])
        return boxes, scores


    def get_boxes_debug(self, image):
        boxes, scores = self.grpc_detect(image)
        debug_image = self.draw_boxes(image, boxes, scores)
        return boxes, debug_image, scores
