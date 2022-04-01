import torch
import numpy as np
from typing import List
from norfair import Detection


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor,
    track_points: str = 'centroid'  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections
    """
    norfair_detections: List[Detection] = []

    if track_points == 'centroid':
        detections_as_xywh = yolo_detections
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    (detection_as_xywh[0].item() + detection_as_xywh[2].item()) / 2,
                    (detection_as_xywh[1].item() + detection_as_xywh[3].item()) / 2,
                ]
            )
            scores = np.array([detection_as_xywh[4].item()])
            label = np.array([detection_as_xywh[5].item()])
            norfair_detections.append(
                Detection(points=centroid, scores=scores, data=label)
            )
    elif track_points == 'bbox':
        detections_as_xyxy = yolo_detections
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()]
                ]
            )
            scores = np.array([detection_as_xyxy[4].item(), detection_as_xyxy[4].item()])
            label = np.array([detection_as_xyxy[5].item()])
            norfair_detections.append(
                Detection(points=bbox, scores=scores, data=label)
            )

    return norfair_detections


def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)
