import numpy as np
import pywt
from scipy import stats


def pywt_swt(data, level=5, thresh_coe=10):
    r'''
    Use `pywt`(CPU) to swt_denoise.

    Parameter:
    @data: A numpy.ndarray with shape(N,), the data to be denoised.
    @level: A positive integer not greater than pywt.swt_max_level(len(data)).
    @thresh_coe: A float to control the strictness of denoising.

    Return:
    A numpy.ndarray with the same shape of @data, denoised @data.
    '''
    thresh_coe *= 0.67
    w = pywt.Wavelet('db8')
    coeffs = pywt.swt(data, w, level=level)
    coeffs_rec = []
    for i in range(len(coeffs)):
        a_i = coeffs[i][0]
        mad = stats.median_absolute_deviation(coeffs[i][1])
        d_i = pywt.threshold(coeffs[i][1], thresh_coe * mad, 'hard')
        coeffs_rec.append((a_i, d_i))

    data_rec = pywt.iswt(coeffs_rec, w)
    return data_rec


def pypwt_swt(data, level=5, thresh_coe=10):
    r'''
    Use `pypwt`(GPU) to swt_denoise.

    Parameters and return are the same as pywt_swt.

    See https://github.com/pierrepaleo/pypwt/issues/5
    '''
    try:
        from pypwt import Wavelets
    except ImportError:
        raise ImportError("Please go https://github.com/pierrepaleo/pypwt and install pypwt")
    thresh_coe *= 0.67
    W = Wavelets(data, "db8", level, do_swt=1)
    W.forward()
    coeffs = W.coeffs
    for i in range(1, W.levels+1):
        mad = stats.median_absolute_deviation(coeffs[i][0])
        hard_threshold(coeffs[i], thresh_coe * mad)
        W.set_coeff(coeffs[i], i)
    W.inverse()
    return W.image

def hard_threshold(data, value, substitute=0):
    r'''
    reference:
    https://github.com/PyWavelets/pywt/blob/master/pywt/_thresholding.py line52
    '''
    # In-place hard threshold using factor * MAD
    cond = np.less(np.absolute(data), value)
    data[cond] = substitute


if __name__ == "__main__":
    from time import time
    import matplotlib.pyplot as plt

    ori_data = np.load("./data_and_res/ori_data.npy")
    fig = plt.figure(figsize=(15, 15))

    ax_ori = fig.add_subplot(311)
    ax_ori.plot(ori_data)
    ax_ori.set_title("original data")

    cpu_start = time()
    cpu_denoised_data = pywt_swt(ori_data)
    print("cpu denoise: {:f}s".format(time()-cpu_start))
    ax_cpu = fig.add_subplot(312)
    ax_cpu.plot(cpu_denoised_data)
    ax_cpu.set_title("cpu_denoise")

    gpu_start = time()
    gpu_denoised_data = pypwt_swt(ori_data)
    print("gpu denoise: {:f}s".format(time()-gpu_start))
    ax_gpu = fig.add_subplot(313)
    ax_gpu.plot(gpu_denoised_data)
    ax_gpu.set_title("gpu_denoise")

    fig.savefig("./data_and_res/result.png")
    plt.close()
