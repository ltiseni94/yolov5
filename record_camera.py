#!/usr/bin/python3
import cv2
import os
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument('source', type=int, help="VideoCapture source")
    parser.add_argument('--output', '-o', default=None, help="Output video path")
    parser.add_argument('--no-record', '-n', action='store_true', default=False)
    ns = parser.parse_args()

    source = ns.source
    output = ns.output
    record = not ns.no_record

    if output is None:
        videos = [video for video in os.listdir('grasp_dataset')
                  if video.endswith('.mp4') and video.startswith('exp_')]
        ids = [int(video.lstrip('exp_')[:-4]) for video in videos]
        if len(ids) > 0:
            new_id = max(ids) + 1
        else:
            new_id = 0
        output = f'grasp_dataset/exp_{new_id}.mp4'
    else:
        if not output.endswith('.mp4'):
            output += '.mp4'

    cap = cv2.VideoCapture(source)
    writer = None
    if record:
        width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (int(width), int(height)))
    if not cap.isOpened():
        raise IOError(f'Could not open videocapture at source {source}')
    try:
        while True:
            res, frame = cap.read()
            if not res:
                raise IOError(f'Could not read frame')
            cv2.imshow('camera', frame)
            if writer is not None:
                writer.write(frame)
            k = cv2.waitKey(1) & 0xff
            if k in (ord('q'), ord('\x1b')):
                break
    except KeyboardInterrupt:
        print('\nTerminated by user')
    finally:
        cv2.destroyAllWindows()
        cap.release()
        if writer is not None:
            writer.release()


if __name__ == '__main__':
    main()
