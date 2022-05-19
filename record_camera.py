#!/usr/bin/python3
import cv2
from argparse import ArgumentParser
from time import strftime

FILE_PATH = f'./runs/video/rec_{strftime("%Y_%m_%d_%H_%M_%S")}.mp4'


def main():
    parser = ArgumentParser()
    parser.add_argument('source', type=int, help="VideoCapture source")
    ns = parser.parse_args()

    source = ns.source
    cap = cv2.VideoCapture(source)
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(FILE_PATH, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (int(width), int(height)))
    if not cap.isOpened():
        raise IOError(f'Could not open videocapture at source {source}')
    try:
        while True:
            res, frame = cap.read()
            if not res:
                raise IOError(f'Could not read frame')
            cv2.imshow('camera', frame)
            writer.write(frame)
            k = cv2.waitKey(1) & 0xff
            if k in (ord('q'), 27):
                break
    except KeyboardInterrupt:
        print('\nTerminated by user')
    finally:
        cv2.destroyAllWindows()
        cap.release()
        writer.release()


if __name__ == '__main__':
    main()

