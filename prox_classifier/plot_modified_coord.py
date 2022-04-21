import matplotlib.pyplot as plt
import numpy as np


def mod_coord(a: float):
    if a < 0.5:
        return 0
    if a > 1:
        return 1
    return 2*a-1


if __name__ == '__main__':
    x = np.arange(start=0.0,
                  stop=3.0,
                  step=0.01,
                  dtype=np.float32)

    y = np.array([mod_coord(val) for val in x], dtype=np.float32)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='coordinate')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Distance function')
    ax.plot(x, 1 - y, label='distance function')
    ax.legend()
    plt.show()


