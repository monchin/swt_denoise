import pywt
import numpy as np
from scipy import stats

def pywt_swt(data, level=5, thresh_coe=6):
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