from typing import Optional


def _landmark_score(modified_coords):
    res = 1
    for i in modified_coords:
        res *= (1-i)
    return res


def _modified_coord(lm_coord: int, obj_coord: int, bbox_dim: int) -> float:
    val = abs((lm_coord - obj_coord) / bbox_dim)
    if val > 1:
        return 1
    if val < 0.5:
        return 0
    return 2 * val - 1


class ProximityScore:
    def __init__(self, holding_samples: int = 10):
        self.value: float = 0
        self.obj_idx: Optional[int] = None
        self.cnt: int = 0
        self.cnt_max: int = holding_samples

    def __call__(self, detections, landmarks_list):
        if not len(detections) > 0:
            return 0, None
        if landmarks_list is None:
            self.cnt += 1
            if self.cnt <= self.cnt_max:
                return self.value, self.obj_idx
            return 0, None
        self.cnt = 0
        res_idx = None
        res_score = 0
        for idx, det in enumerate(detections):
            new_res = 0
            for lm in landmarks_list:
                mod_coords = [
                    _modified_coord(lm[0],
                                    (det[0] + det[2]) / 2,
                                    det[2] - det[0]),
                    _modified_coord(lm[1],
                                    (det[1] + det[3]) / 2,
                                    det[3] - det[1])
                ]
                new_res += _landmark_score(mod_coords) / len(landmarks_list)
            if new_res >= res_score:
                res_score = new_res
                res_idx = idx
        self.value = res_score
        self.obj_idx = res_idx
        return res_score, res_idx
