import numpy as np
import tensorflow as tf
from typing import NamedTuple, Tuple
from csv import reader
import copy
import itertools


class PoseClassifier:
    def __init__(self,
                 model_path: str = 'pose_classifier/keypoint_classifier.tflite',
                 label_path: str = 'pose_classifier/keypoint_classifier_label.csv',
                 num_threads: int = 1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        with open(label_path, encoding='utf-8-sig') as f:
            parser = reader(f)
            self.labels = [row[0] for row in parser]

    def predict(self, landmark_list: NamedTuple) -> Tuple[str, np.ndarray]:
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))
        return self.labels[result_index], np.squeeze(result)
