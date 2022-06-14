import cv2
import argparse
import csv
from grasp import nice_text


obj_labels = (
    'bottle',
    'cup',
    'handbag',
    'cell phone',
    'book',
    'fork',
    'spoon',
    'bowl',
    'apple',
    'banana',
    'orange',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='path to video input')
    parser.add_argument('--output', help='output path', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.video[:-4] + '_true.csv'

    cap = cv2.VideoCapture(args.video)
    orig_frames = []

    while True:
        res, orig_frame = cap.read()
        if not res:
            print('No more frames available')
            break
        orig_frames.append(orig_frame)

    if len(orig_frames) == 0:
        raise ValueError('Could not read any frame')
    frame_idx = 0
    grasps = [0] * len(orig_frames)
    labels = [0] * len(orig_frames)
    hold = 0
    last_grasp = 0
    last_label = 0

    def change_frame(x):
        nonlocal frame_idx, grasps, labels
        frame_idx = cv2.getTrackbarPos('frame', f'{args.video}')
        if hold:
            grasps[frame_idx] = last_grasp
            labels[frame_idx] = last_label
        cv2.setTrackbarPos('grasp', f'{args.video}', grasps[frame_idx])
        cv2.setTrackbarPos('obj_label', f'{args.video}', labels[frame_idx])
        draw_image()

    def change_grasp(x):
        nonlocal grasps
        grasps[frame_idx] = cv2.getTrackbarPos('grasp', f'{args.video}')
        draw_image()

    def change_label(x):
        nonlocal labels
        labels[frame_idx] = cv2.getTrackbarPos('obj_label', f'{args.video}')
        draw_image()

    def change_hold(x):
        nonlocal hold, last_grasp, last_label
        last_grasp = cv2.getTrackbarPos('grasp', f'{args.video}')
        last_label = cv2.getTrackbarPos('obj_label', f'{args.video}')
        if cv2.getTrackbarPos('hold', f'{args.video}') == 1:
            hold = True
        else:
            hold = False

    def draw_image():
        frame = orig_frames[frame_idx].copy()
        if grasps[frame_idx] == 1:
            frame = nice_text(frame, 'GRASP', (10, 50))
        frame = nice_text(frame, f'{obj_labels[labels[frame_idx]]}', (10, 20))
        cv2.imshow(f'{args.video}', frame)

    cv2.namedWindow(f'{args.video}')
    cv2.createTrackbar('grasp', f'{args.video}', 0, 1, change_grasp)
    cv2.createTrackbar('obj_label', f'{args.video}', 0, 10, change_label)
    cv2.createTrackbar('frame', f'{args.video}', 0, len(orig_frames) - 1, change_frame)
    cv2.createTrackbar('hold', f'{args.video}', 0, 1, change_hold)
    draw_image()
    while (cv2.waitKey(1) & 0xFF) != ord('\r'):
        pass
    cv2.destroyAllWindows()

    with open(args.output, 'w') as f:
        csv_writer = csv.DictWriter(f, ('grasp', 'label'))
        csv_writer.writeheader()
        for grasp, label in zip(grasps, labels):
            csv_writer.writerow(dict(
                grasp=grasp,
                label=obj_labels[label],
            ))


if __name__ == '__main__':
    main()
