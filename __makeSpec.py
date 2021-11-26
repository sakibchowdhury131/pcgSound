from scipy import signal



def spectrogram(sig_in):
    nperseg = 256 # default 256
    noverlap = nperseg // 4 # default: nperseg // 8
    fs = sample_rate# raw signal sample rate is 2000Hz
    window = 'triang'
    scaling = 'density' # {'density', 'spectrum'}
    detrend = 'linear' # {'linear', 'constant', False}
    eps = 1e-11
    f, t, Sxx = signal.spectrogram(sig_in, nperseg=nperseg, noverlap=noverlap,
                                   fs=fs, window=window,
                                   scaling=scaling, detrend=detrend)
    return f, t, np.log(Sxx + eps)
