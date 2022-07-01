import os
import csv
import json
from typing import Dict
from label_video import obj_labels


DATASET = 'grasp_dataset/'


class GraspHysteresis:
    def __init__(self, low: float = 0.51, high: float = 0.6):
        self.bool: bool = False
        self.low: float = low
        self.high: float = high

    def __call__(self, new_val: float):
        if self.bool:
            self.bool = new_val > self.low
        else:
            self.bool = new_val > self.high
        return self.bool


class Result:
    def __init__(self, true_pos: int = 0, false_pos: int = 0, false_neg: int = 0):
        self.true_pos: int = true_pos
        self.false_pos: int = false_pos
        self.false_neg: int = false_neg

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Results can only be sum to another Result')
        return Result(
            true_pos=self.true_pos + other.true_pos,
            false_pos=self.false_pos + other.false_pos,
            false_neg=self.false_neg + other.false_neg,
        )

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError('Results can only be sum to another Result')
        self.true_pos += other.true_pos
        self.false_pos += other.false_pos
        self.false_neg += other.false_neg
        return self

    def __repr__(self):
        p_string = f'{None}'
        if (precision := self.precision) is not None:
            p_string = f'{precision:.3f}'
        r_string = f'{None}'
        if (recall := self.recall) is not None:
            r_string = f'{recall:.3f}'
        return f'Result(P={p_string}, R={r_string}, ' \
               f'TP={self.true_pos}, FP={self.false_pos}, FN={self.false_neg})'

    @property
    def recall(self):
        try:
            return self.true_pos / (self.true_pos + self.false_neg)
        except ZeroDivisionError:
            return None

    @property
    def precision(self):
        try:
            return self.true_pos / (self.true_pos + self.false_pos)
        except ZeroDivisionError:
            return None


class ResultSerializer(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Result):
            res = o.__dict__.copy()
            res.update({'precision': o.precision, 'recall': o.recall})
            return res
        return super().default(o)


def sum_dict(*dicts: Dict[str, Result]) -> Dict[str, Result]:
    if len(dicts) == 0:
        raise ValueError('At least one input dictionary is required')

    if not all(map(lambda x: type(x) == dict, dicts)):
        raise TypeError('Some input element is not a dictionary')

    for d in dicts:
        if not all(map(lambda x: x in d, obj_labels[1:])):
            raise ValueError('Missing label in input dictionary')
        if 'total' not in d:
            raise ValueError('Missing label "total" in input dictionary')
        if not all(map(lambda x: type(x) == Result, d.values())):
            raise TypeError('Some input dictionary has '
                            'a value that is not a Result')

    result = {key: Result() for key in dicts[0]}
    for d in dicts:
        for key in result:
            result[key] += d[key]

    return result


def compare_files(true: str, pred: str) -> Dict[str, Result]:
    result: Dict[str, Result] = {key: Result() for key in obj_labels if key != 'none'}
    grasp_hyst = GraspHysteresis()
    with open(true, 'r') as f:
        lines = len(f.readlines())
    with open(pred, 'r') as f:
        if len(f.readlines()) != lines:
            raise ValueError(f'Selected files have a different number of lines')
    with open(true, 'r') as true_file:
        with open(pred, 'r') as pred_file:
            true_reader = csv.DictReader(true_file)
            pred_reader = csv.DictReader(pred_file)
            for true_row, pred_row in zip(true_reader, pred_reader):
                pred_grasp = grasp_hyst(float(pred_row['grasp']))
                if int(true_row['grasp']) == 0:
                    if pred_grasp:
                        result[pred_row['label']].false_pos += 1
                elif int(true_row['grasp']) == 1:
                    if not pred_grasp:
                        result[true_row['label']].false_neg += 1
                    if pred_grasp:
                        if not pred_row['label'] == true_row['label']:
                            result[true_row['label']].false_neg += 1
                            result[pred_row['label']].false_pos += 1
                        else:
                            result[true_row['label']].true_pos += 1
    result.update({'total': Result(
        true_pos=sum([result[key].true_pos for key in obj_labels if key != 'none']),
        false_pos=sum([result[key].false_pos for key in obj_labels if key != 'none']),
        false_neg=sum([result[key].false_neg for key in obj_labels if key != 'none']),
    )})
    return result


def main() -> None:
    dataset = os.listdir(DATASET)
    videos = [video for video in dataset if video.endswith('.mp4')]
    videos = sorted(videos, key=lambda x: int(x[:-4].lstrip('exp_')))

    for video in videos:
        pred_file = video[:-4] + '_pred.csv'
        true_file = video[:-4] + '_true.csv'
        if not (pred_file in dataset and true_file in dataset):
            raise ValueError(f'Missing labels in dataset: {pred_file}, {true_file}')

    results = [compare_files(
        true=DATASET + video[:-4] + '_true.csv',
        pred=DATASET + video[:-4] + '_pred.csv',
    ) for video in videos]

    participants_result = [sum_dict(*results[i * 5: (i + 1) * 5]) for i in range(len(videos) // 5)]
    participants_result_json = [json.dumps(res, indent=2, cls=ResultSerializer) for res in participants_result]

    aggregate_result = sum_dict(*results)
    aggregate_result_json = json.dumps(aggregate_result, indent=2, cls=ResultSerializer)

    with open('grasp_dataset/aggregate.json', 'w') as f:
        f.write(aggregate_result_json)

    for idx, json_dump in enumerate(participants_result_json):
        with open(f'grasp_dataset/participant_{idx}.json', 'w') as f:
            f.write(json_dump)

    for idx, result in enumerate(results):
        with open(f'grasp_dataset/exp_{idx}.json', 'w') as f:
            f.write(json.dumps(
                result, indent=2, cls=ResultSerializer
            ))


if __name__ == '__main__':
    main()
