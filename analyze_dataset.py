import os
import subprocess

if __name__ == '__main__':
    labels = ['grasp_dataset/' + label for label in os.listdir('grasp_dataset')
              if label.startswith('exp_') and label.endswith('_true.csv')]

    videos = ['grasp_dataset/' + video for video in os.listdir('grasp_dataset')
              if video.startswith('exp_') and video.endswith('.mp4')]

    for label in labels:
        try:
            videos.remove(f'{label.rstrip("_true.csv")}.mp4')
            print(f'Already analyzed: {label.rstrip("_true.csv")}.mp4')
        except ValueError:
            pass

    for video in videos:
        subprocess.run(f'python app.py '
                       f'--source {video} '
                       f'--save-label-file', shell=True)
