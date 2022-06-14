import cv2
import argparse
import csv
from grasp import nice_text


obj_labels = (
    'none',
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
    parser.add_argument('--output', '-o', help='output path', default=None)
    parser.add_argument('--input-label', '-i', default=None)
    parser.add_argument('--read-only', '-r', action='store_true', default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = args.video[:-4] + '_true.csv'
    if args.input_label is None:
        args.input_label = args.output

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
    hold = False
    labels = []
    input_file = None
    try:
        input_file = open(args.input_label, 'r')
    except FileNotFoundError:
        labels = [0] * len(orig_frames)
        last_label = 0

    if input_file is not None:
        try:
            reader = csv.DictReader(input_file)
            for row in reader:
                labels.append(obj_labels.index(row['label']))
            last_label = labels[0]
        finally:
            input_file.close()

    def change_frame(x):
        nonlocal frame_idx, labels
        frame_idx = cv2.getTrackbarPos('frame', f'{args.video}')
        if not args.read_only:
            if hold:
                labels[frame_idx] = last_label
            cv2.setTrackbarPos('obj_label', f'{args.video}', labels[frame_idx])
        draw_image()

    def change_label(x):
        nonlocal labels, last_label
        labels[frame_idx] = cv2.getTrackbarPos('obj_label', f'{args.video}')
        if hold:
            last_label = labels[frame_idx]
        draw_image()

    def change_hold(x):
        nonlocal hold, last_label
        last_label = cv2.getTrackbarPos('obj_label', f'{args.video}')
        if cv2.getTrackbarPos('hold', f'{args.video}') == 1:
            hold = True
        else:
            hold = False

    def draw_image():
        frame = orig_frames[frame_idx].copy()
        if labels[frame_idx] > 0:
            frame = nice_text(frame, f'{obj_labels[labels[frame_idx]]}', (10, 20))
        cv2.imshow(f'{args.video}', frame)

    cv2.namedWindow(f'{args.video}')
    cv2.createTrackbar('frame', f'{args.video}', 0, len(orig_frames) - 1, change_frame)
    if not args.read_only:
        cv2.createTrackbar('obj_label', f'{args.video}', 0, 11, change_label)
        cv2.createTrackbar('hold', f'{args.video}', 0, 1, change_hold)
    draw_image()
    if args.read_only:
        print('Press Enter or ctrl + C to terminate')
    else:
        print('Press Enter if you want to save, ctrl + C to abort')
    while (cv2.waitKey(1) & 0xFF) != ord('\r'):
        pass
    cv2.destroyAllWindows()

    if not args.read_only:
        with open(args.output, 'w') as f:
            csv_writer = csv.DictWriter(f, ('grasp', 'label'))
            csv_writer.writeheader()
            for label in labels:
                csv_writer.writerow(dict(
                    grasp=1 if label > 0 else 0,
                    label=obj_labels[label],
                ))
        print('Saved new label file')


if __name__ == '__main__':
    main()
