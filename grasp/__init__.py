import cv2
import numpy as np
import time
from typing import Tuple, Union, Optional


class GraspScore:
    def __init__(self,
                 conf_thres=0.6,
                 release_thres=0.3,
                 close_weight=0.35,
                 proximity_weight=0.4,
                 reset_counter_max=120):
        self.conf_thres = conf_thres
        self.release_thres = release_thres
        self.previous_val = 0
        self.reset_counter_max = reset_counter_max
        self.reset_counter = 0
        self.cw = close_weight
        self.pw = proximity_weight
        self.mw = 1 - close_weight - proximity_weight
        self.is_grasping = False
        assert 1 > self.mw > 0

    def __call__(self, close_prob, prox_score):
        self.reset_counter += 1
        if self.reset_counter == self.reset_counter_max:
            self.reset_counter = 0
            new_val = self.cw * close_prob + self.pw * prox_score
            new_val *= 1 / (self.cw + self.pw)
        else:
            new_val = self.mw * self.previous_val + self.cw * close_prob + self.pw * prox_score
        self.previous_val = new_val
        if self.is_grasping:
            self.is_grasping = new_val > self.release_thres
        else:
            self.is_grasping = new_val > self.conf_thres
        return new_val, self.is_grasping


class Smoother:
    def __init__(self, slope: float = 1.0):
        self.slope = slope
        self.time = time.time()
        self.value = 0

    def __call__(self, ref: float):
        new_time = time.time()

        # Avoid excessive dt if the weight is updated after a long time
        if new_time - self.time > (1.0 / 10):
            self.time = new_time
            return self.value

        dt = new_time - self.time
        self.time = new_time

        if ref > self.value:
            self.value += self.slope * dt
            if self.value > ref:
                self.value = ref
        elif ref < self.value:
            self.value -= self.slope * dt
            if self.value < ref:
                self.value = ref

        return self.value


def score_bar(
        fill: float,
        img: np.ndarray,
        *,
        lower_corner: Union[Tuple[int, int], int] = 20,
        width: int = 20,
        height: int = 300,
        line_thickness: int = 2,
        horizontal: bool = False,
        colormap: bool = True,
        use_value: bool = False,
        low_color: Optional[int] = 40,
        high_color: Optional[int] = 10,
        single_color: Tuple[int, int, int] = (255, 255, 255),
        with_centered_fill_value: bool = False,
        centered_custom_label: Optional[str] = None,
        with_label: bool = True,
        char_size: float = 0.5,
        custom_label: Optional[str] = None,
):
    if fill < 0:
        fill = 0
    elif fill > 1:
        fill = 1

    if type(lower_corner) is int:
        lower_corner = (lower_corner, img.shape[0] - lower_corner)

    filled_portion = int((height - 2 * line_thickness) * fill)

    if colormap:
        hue = min([high_color, low_color]) + int(abs(high_color-low_color) * (1 - fill))
        color = np.array([[[hue, 255, 255]]], dtype=np.uint8)

        if use_value:
            hue_dummy_array = np.array([[single_color]], dtype=np.uint8)
            hue_dummy_array = cv2.cvtColor(hue_dummy_array, cv2.COLOR_BGR2HSV)
            color[0, 0, 2] = 130 + int(fill * 125)
            color[0, 0, 0] = hue_dummy_array[0, 0, 0]

        color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
        color = (int(color[0, 0, 0]), int(color[0, 0, 1]), int(color[0, 0, 2]))

    else:
        color = single_color

    if horizontal:
        higher_corner = (lower_corner[0] + height, lower_corner[1] - width)
        fill_corner = (lower_corner[0] + line_thickness + filled_portion, lower_corner[1] - width + line_thickness)
    else:
        higher_corner = (lower_corner[0] + width, lower_corner[1] - height)
        fill_corner = (lower_corner[0] + width - line_thickness, lower_corner[1] - line_thickness - filled_portion)

    # BORDER
    cv2.rectangle(
        img,
        lower_corner,
        higher_corner,
        (0, 0, 0),
        line_thickness,
    )
    cv2.rectangle(
        img,
        lower_corner,
        higher_corner,
        (255, 255, 255),
        line_thickness // 2,
    )

    # FILL
    cv2.rectangle(
        img,
        (lower_corner[0] + line_thickness, lower_corner[1] - line_thickness),
        (fill_corner[0], fill_corner[1]),
        color,
        -1,
    )

    # LABEL
    if with_label:
        if custom_label is None:
            label = f'{fill * 100:.1f} %'
        else:
            label = custom_label
        label_size, font_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            char_size,
            2,
        )

        label_width, label_height = label_size[0], label_size[1]

        if horizontal:
            text_origin = (higher_corner[0] + font_size, lower_corner[1] + (label_height - width) // 2)
        else:
            text_origin = (lower_corner[0] + (width - label_width) // 2 + font_size, higher_corner[1] - 5)
        cv2.putText(
            img,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            char_size,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            img,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            char_size,
            (255, 255, 255),
            2,
        )

    if with_centered_fill_value and horizontal:

        label = f'{fill * 100:.1f} %'
        if centered_custom_label is not None:
            label = centered_custom_label

        text_size = width / 50

        label_size, font_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            3,
        )

        label_width, label_height = label_size[0], label_size[1]

        text_origin = (lower_corner[0] + (height - label_width) // 2, lower_corner[1] - (width - label_height) // 2)

        cv2.putText(
            img,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            img,
            label,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (255, 255, 255),
            2,
        )

    return img
