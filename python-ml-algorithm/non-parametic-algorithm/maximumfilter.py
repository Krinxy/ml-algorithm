from scipy.ndimage import maximum_filter
import numpy as np


def Array_Maximum_filter(data: np.ndarray, size: int = 0, stencil_size: tuple = (0, 0)):
    if size != 0:
        filtered_data = maximum_filter(data, size)
    elif stencil_size != (0, 0) and size == 0:
        filtered_data = maximum_filter(data, stencil_size)
    else:
        raise ValueError(f"Size measurement incorrect")

    return filtered_data


if __name__ == '__main__':
    data = np.random.randint(0, 100, 100)
    # print(data)
    d = Array_Maximum_filter(data, 3)

    print('1D Maximumfilter\n', d)
    print(f'Audiorate: {max(d)}')
    print(f'Samplerate: {max(d) * 2}')
