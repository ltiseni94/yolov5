import os
import subprocess

if __name__ == '__main__':
    videos = ['grasp_dataset/' + video for video in os.listdir('grasp_dataset')
              if video.startswith('exp_') and video.endswith('.mp4')]
    for video in videos:
        subprocess.run(f'python app.py '
                       f'--weights yolov5s.pt '
                       f'--source {video} '
                       f'--iou-thres 0.2 '
                       f'--conf-thres 0.35 '
                       f'--agnostic-nms '
                       f'--save-label-file', shell=True)
