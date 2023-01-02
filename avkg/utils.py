import math
from typing import List


def intersect(*args: set):
    arg0, *rest_args = args
    if not rest_args:
        return arg0
    return arg0.intersection(intersect(*rest_args))


def softmax(xs: List[float]):
    x_max = max(xs)
    xs = [x - x_max for x in xs]
    exp_xs = [math.exp(x) for x in xs]
    denom = sum(exp_xs)
    return [exp_x / denom for exp_x in exp_xs]

