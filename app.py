# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import socket
import struct
import time
import csv
from queue import Queue, Full
from threading import Thread
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device
from utils.cvfpscalc import CvFpsCalc

from pose_classifier.classifier import PoseClassifier
from pose_classifier.utils import calc_landmark_list, pre_process_landmark  # draw_bounding_rect

from prox_classifier import ProximityScore
from grasp import GraspScore, score_bar, Smoother, nice_text
from norfair import Tracker, FilterSetup  # draw_tracked_boxes
from utils.norfair import euclidean_distance, yolo_detections_to_norfair_detections
from typing import Dict, List, Optional

import mediapipe as mp
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

OUTPUT_IP = '127.0.0.1'
OUTPUT_PORT = 11000

CLASSES_NAMES = {
    'bottle': 1.0,
    'cup': 0.4,
    # 'handbag': 1.5,
    'cell phone': 0.25,
    'book': 0.5,
    # 'fork': 0.1,
    # 'sports ball': 0.3,
    # 'wine glass': 0.2,
    # 'spoon': 0.1,
    # 'bowl': 0.7,
    # 'apple': 0.3,
    # 'banana': 0.3,
    'orange': 0.4,
    # 'carrot': 0.1,
    # 'remote': 0.2,
    # 'scissors': 0.1,
    # 'hair drier': 0.5,
    # 'laptop': 1.5,
    # 'donut': 0.2,
    # 'sandwich': 0.4,
    # 'backpack': 1.5,
}

MAX_WEIGHT = max(CLASSES_NAMES.values())

NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']  # class names

NAMES_DICT = {}
for index, obj_name in enumerate(NAMES):
    NAMES_DICT.update({obj_name: index})
CLASSES = [NAMES_DICT[x] for x in CLASSES_NAMES]

max_distance_between_points: int = 100


@torch.no_grad()
def run(
        output_queue,
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save=False,  # save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        use_static_image_mode=False,
        hand_detection_confidence=0.7,
        hand_tracking_confidence=0.5,
        close_multiplier=1.5,
        show_debug=False,
        save_no_draw=False,
        save_label_file=False,
        server=True,
):
    source = str(source)
    save_img = save and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path: List[Optional[str]] = [None] * bs
    vid_writer: List[Optional[cv2.VideoWriter]] = [None] * bs

    pose_class = PoseClassifier()
    grasp_score = GraspScore(close_weight=0.1,
                             proximity_weight=0.1,
                             conf_thres=0.6,
                             release_thres=0.51)
    proximity_score = ProximityScore(holding_samples=30)
    tracker = Tracker(distance_function=euclidean_distance,
                      distance_threshold=max_distance_between_points,
                      hit_inertia_min=10,
                      hit_inertia_max=80,
                      initialization_delay=0,
                      detection_threshold=0,
                      point_transience=2,
                      past_detections_length=2,
                      filter_setup=FilterSetup(R=1.0, Q=0.1, P=1.0)
                      )
    smoother = Smoother()

    video_writer = None
    label_file = None
    csv_writer = None

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    obj_label_dict: Dict[int, str] = {}
    obj_label = None

    close_prob = 0
    close_max_cnt = 10
    close_cnt = close_max_cnt

    try:
        with mp_hands.Hands(
                static_image_mode=use_static_image_mode,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=hand_detection_confidence,
                min_tracking_confidence=hand_tracking_confidence,
        ) as hands:
            fps_counter = CvFpsCalc()
            for path, im, im0s, vid_cap, s in dataset:

                # HANDS TRACKER
                shape_len = len(im.shape)
                mp_img = np.zeros((im.shape[shape_len-2], im.shape[shape_len-1], im.shape[shape_len-3]),
                                  dtype=np.dtype('B'))
                for i in range(im.shape[shape_len-3]):
                    if shape_len == 4:
                        mp_img[:, :, i] = im[0, i, :, :]
                    elif shape_len == 3:
                        mp_img[:, :, i] = im[i, :, :]
                    else:
                        raise ValueError(f'Image shape has length different from 3 or 4:\n'
                                         f'\t shape: {im.shape}')
                hands_results = hands.process(mp_img)

                landmark_list = None
                if hands_results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(
                            hands_results.multi_hand_landmarks,
                            hands_results.multi_handedness
                    ):
                        # b_rect = calc_bounding_rect(mp_img, hand_landmarks)
                        # if handedness.classification[0].label == 'Left':
                        landmark_list = calc_landmark_list(mp_img, hand_landmarks)
                        pp_landmark_list = pre_process_landmark(landmark_list)
                        pose_label, pose_probs = pose_class.predict(pp_landmark_list)
                        close_prob = close_multiplier * pose_probs[1]
                        if close_prob > 1:
                            close_prob = 1
                        close_cnt = close_max_cnt

                close_cnt -= 1
                if close_cnt < 0:
                    close_prob = 0

                # OBJECT DETECTOR
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

                # OBJECTS TRACKING
                norfair_dets = yolo_detections_to_norfair_detections(pred[0], 'bbox')
                tracked_objects = tracker.update(detections=norfair_dets, period=3)

                norfair_bboxes = [obj.filter.x.squeeze()[:4] for obj in tracked_objects if obj.has_inertia]

                prox_score, obj_idx = proximity_score(norfair_bboxes, landmarks_list=landmark_list)
                grasp_prob, grasp_label = grasp_score(close_prob, prox_score)

                try:
                    if obj_idx is not None:
                        if tracked_objects[obj_idx].id not in obj_label_dict:
                            obj_label_dict.update({
                                tracked_objects[obj_idx].id: names[int(tracked_objects[obj_idx].last_detection.data[0])]
                            })
                        obj_label = obj_label_dict[tracked_objects[obj_idx].id]
                        # obj_label = names[int(tracked_objects[obj_idx].last_detection.data[0])]
                except IndexError:
                    pass
                obj_weight = CLASSES_NAMES[obj_label] if obj_label in CLASSES_NAMES else 0
                output_weight = obj_weight if grasp_score.is_grasping else 0
                visualized_weight = smoother(output_weight)

                try:
                    output_queue.put(output_weight, block=False)
                except Full:
                    pass

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        # s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + \
                                  ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    # s += '%gx%g ' % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    imc = im0.copy() if save_crop or save_no_draw else im0  # for save_crop
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))
                                if save_crop:
                                    save_one_box(xyxy,
                                                 imc,
                                                 file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                                 BGR=True)

                    fps = fps_counter.get()

                    # Stream results
                    im0 = annotator.result()

                    if view_img or save:
                        if hands_results.multi_hand_landmarks:
                            for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
                                mp_drawing.draw_landmarks(
                                    im0,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_styles.get_default_hand_landmarks_style(),
                                    mp_styles.get_default_hand_connections_style(),
                                )

                        weight_score_bar_kwargs = {
                            'fill': visualized_weight / MAX_WEIGHT,
                            'lower_corner': (5, 30),
                            'custom_label': f'{obj_label}' if grasp_score.is_grasping else f'No object',
                            'centered_custom_label': f'{visualized_weight:.2f} kg',
                            'low_color': 120,
                            'high_color': 90,
                        }

                        grasp_score_bar_kwargs = {
                            'fill': grasp_prob,
                            'lower_corner': (5, 60),
                            'custom_label': f'Grasp',
                        }

                        if save_no_draw:
                            imc = nice_text(imc, f'fps: {fps:.1f}', (imc.shape[1] - 100, imc.shape[0] - 5))
                            imc = score_bar(
                                imc,
                                **weight_score_bar_kwargs,
                            )
                            imc = score_bar(
                                imc,
                                **grasp_score_bar_kwargs,
                            )

                        im0 = nice_text(im0, f'fps: {fps:.1f}', (im0.shape[1] - 100, im0.shape[0] - 5))
                        im0 = score_bar(
                            im0,
                            **weight_score_bar_kwargs
                        )
                        im0 = score_bar(
                            im0,
                            **grasp_score_bar_kwargs
                        )

                        if show_debug:
                            im0 = score_bar(
                                im0,
                                prox_score,
                                lower_corner=(5, 90),
                                custom_label='Prox',
                                use_value=True,
                                single_color=(100, 255, 100),
                            )
                            im0 = score_bar(
                                im0,
                                close_prob,
                                lower_corner=(5, 120),
                                custom_label='Close',
                                use_value=True,
                                single_color=(255, 100, 100),
                            )

                        # draw_tracked_boxes(im0, tracked_objects, (255, 0, 0), 1, 1.5, 2)
                        if view_img:
                            cv2.imshow(str(p), im0)
                            cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
                        else:  # 'video' or 'stream'
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer[i].write(im0)

                    if save_no_draw:
                        if video_writer is None:
                            video_path = str(Path(save_path)) + '_nodraw.mp4'
                            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        video_writer.write(imc)

                    if save_label_file:
                        if label_file is None:
                            label_file = open(f'{source[:-4]}_pred.csv', 'w')
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(
                                label_file,
                                ('grasp', 'label')
                            )
                            csv_writer.writeheader()
                        csv_writer.writerow({
                            'grasp': f'{grasp_prob:.3f}',
                            'label': f'{obj_label if obj_label is not None else "None"}',
                        })

                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    finally:
        try:
            label_file.close()
        except AttributeError:
            pass

    # Print results
    # t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save', action='store_true', help='save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, default=CLASSES,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=True, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--hand-detection-confidence", help='min_detection_confidence', type=float, default=0.6)
    parser.add_argument("--hand-tracking-confidence", help='min_tracking_confidence', type=float, default=0.35)
    parser.add_argument("--save-no-draw", help='save video without yolo bboxes and mp hands',
                        action='store_true', default=False)
    parser.add_argument("--show-debug", help='show close and prox score bars', action='store_true', default=False)
    parser.add_argument("--server", help='send out result on 127.0.0.1, port 11000', action='store_true', default=False)
    parser.add_argument("--save-label-file", help='save grasp results in a txt file',
                        action='store_true', default=False)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def send_result(input_queue: Queue, rate: float):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    period = 1 / rate
    cnt = 0
    while True:
        start_time = time.time()
        val = None
        while not input_queue.empty():
            val = input_queue.get()
        if val is not None:
            cnt += 1
            sock.sendto(struct.pack('f', val), (OUTPUT_IP, OUTPUT_PORT))
            if cnt % int(rate) == 0:
                print(f'[{time.strftime("%H:%M:%S")}] Weight: {val:.3f} kg')
        elapsed_time = time.time() - start_time
        try:
            time.sleep(period - elapsed_time)
        except ValueError:
            pass


def main(output_queue: Queue, opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(output_queue, **vars(opt))


if __name__ == "__main__":
    args = parse_opt()
    if args.save_label_file:
        args.save = True
    data_queue = Queue()
    if args.server:
        t = Thread(
            target=send_result,
            name='send_data_thread',
            args=(data_queue, 10.0),
            daemon=True,
        )
        t.start()
    main(data_queue, args)
