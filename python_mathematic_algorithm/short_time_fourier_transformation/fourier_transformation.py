import numpy as np
import scipy.fft


def FastFourierTransformation_u_scipy(data: np.ndarray):
    fft_data = scipy.fft.fft(data)

    return fft_data


def FastFourierTransformation_u_numpy(data: np.ndarray):
    fft_data = np.fft.fft(data)

    return fft_data

if __name__ == '__main__':
    randomdata = np.random.randint(0, 100, 100)
    
    fft_data = FastFourierTransformation_u_scipy(randomdata)
    print('Data\n', randomdata)
    print('FFT Data w/ scipy \n', fft_data)

    fft_data_np = FastFourierTransformation_u_numpy(randomdata)
    print('Data\n', randomdata)
    print('FFt Data w/ np:\n', fft_data_np)

    print(np.allclose(fft_data_np, fft_data))       # np.allclose checking with tolerance
