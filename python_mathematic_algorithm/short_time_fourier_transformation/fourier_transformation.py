import numpy as np
from scipy.fft import fft 


def FastFourierTransformation(data: np.ndarray):
    fft_data = fft(data)

    return fft_data

if __name__ == '__main__':
    randomdata = np.random.randint(0, 100, 100)
    fft_data = FastFourierTransformation(randomdata)
    print('Data\n', randomdata)
    print('FFT Data \n', fft_data)
