import numpy as np
from scipy.io import wavfile
from scipy.linalg import toeplitz, norm
import math
from scipy.fftpack import fft, ifft
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' 
This is a python script which can be regarded as implementation of matlab script "compute_metrics_norm.m".

Usage: 
    pesq, csig, cbak, covl, ssnr, stoi = compute_metrics_norm(cleanFile, enhancedFile, Fs, norm, align, path)
    cleanFile: clean audio or target audio 
               if path==1, this input should be in .wav format
               else, this input should be a numpy array
    enhancedFile: enhanced audio, output from any speech enhancement algorithms
                  if path==1, this input should be in .wav format
                  else, this input should be a numpy array
    Fs: sampling rate, usually equals to 8000 or 16000 Hz
    norm: whether to normalize the "cleanFile" and "enhancedFile" arguments, 1 indicates true
    align: whether to align the "cleanFile" and "enhancedFile", 1 indicates true
    path: whether the "cleanFile" and "enhancedFile" arguments are in .wav format or in numpy array format, 
          1 indicates "in .wav format"
          
Example call:
    pesq_output, csig_output, cbak_output, covl_output, ssnr_output, stoi_output = \
            compute_metrics_norm(target_audio, output_audio, 16000, 1, 1, 1)

Author: Tu, Qian
'''


def compute_metrics_norm(cleanFile, enhancedFile, Fs, norm, align, path):
    alpha = 0.95

    if path == 1:
        sampling_rate1, data1 = wavfile.read(cleanFile)
        sampling_rate2, data2 = wavfile.read(enhancedFile)
        if sampling_rate1 != sampling_rate2:
            raise ValueError('The two files do not match!\n')
    else:
        data1 = cleanFile
        data2 = enhancedFile
        sampling_rate1 = Fs
        sampling_rate2 = Fs

    data1 = data1 / 32768  # row vector here, but column vector in matlab
    data2 = data2 / 32768  # row vector here, but column vector in matlab
    length = min(len(data1), len(data2))
    data1 = data1[0: length] + np.spacing(1)
    data2 = data2[0: length] + np.spacing(1)

    if norm:
        data1 = np.divide(data1, np.max(np.abs(data1)))
        data2 = np.divide(data2, np.max(np.abs(data2)))

    if align:
        data1_aligned, data2_aligned = align_signal(data1, data2)
        data1 = data1_aligned
        data2 = data2_aligned
        l_cut = np.minimum(np.size(data1), np.size(data2))
        z1 = np.size(data1[np.where(data1 == 0)])
        z2 = np.size(data1[np.where(data2 == 0)])
        zero_count = np.maximum(z1, z2)
        data1 = data1[zero_count: l_cut]
        data2 = data2[zero_count: l_cut]

    # compute the WSS measure
    wss_dist_vec = wss(data1, data2, sampling_rate1)
    wss_dist_vec = np.sort(wss_dist_vec)
    wss_dist = np.mean(wss_dist_vec[0: round(np.size(wss_dist_vec) * alpha)])

    # compute the LLR measure
    LLR_dist = llr(data1, data2, sampling_rate1)
    LLRs = np.sort(LLR_dist)
    LLR_len = round(np.size(LLR_dist) * alpha)
    llr_mean = np.mean(LLRs[0: LLR_len])

    # compute the SNRseg
    snr_dist, segsnr_dist = snr(data1, data2, sampling_rate1)
    snr_mean = snr_dist
    segSNR = np.mean(segsnr_dist)

    # # compute the pesq
    pesq_mos = pesq2(data1 * 32768, data2 * 32768, sampling_rate1, 'wideband')

    # now compute the composite measures
    CSIG = 3.093 - 1.029 * llr_mean + 0.603 * pesq_mos - 0.009 * wss_dist
    CSIG = max(1, CSIG)
    CSIG = min(5, CSIG)    # limit values to [1, 5]
    CBAK = 1.634 + 0.478 * pesq_mos - 0.007 * wss_dist + 0.063 * segSNR
    CBAK = max(1, CBAK)
    CBAK = min(5, CBAK)    # limit values to [1, 5]
    COVL = 1.594 + 0.805 * pesq_mos - 0.512 * llr_mean - 0.007 * wss_dist
    COVL = max(1, COVL)
    COVL = min(5, COVL)    # limit values to [1, 5]

    STOI = stoi(data1, data2, sampling_rate1)

    return pesq_mos, CSIG, CBAK, COVL, segSNR, STOI


def wss(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech, which must be the same.
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError('Files must have same length.')

    # Global variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(int)  # window length in samples
    skiprate = (np.floor(np.divide(winlength, 4))).astype(int)   # window skip in samples
    max_freq = (np.divide(sample_rate, 2)).astype(int)   # maximum bandwidth
    num_crit = 25    # number of critical bands

    USE_FFT_SPECTRUM = 1   # defaults to 10th order LP spectrum
    n_fft = (np.power(2, np.ceil(np.log2(2 * winlength)))).astype(int)
    n_fftby2 = (np.multiply(0.5, n_fft)).astype(int)   # FFT size/2
    Kmax = 20.0    # value suggested by Klatt, pg 1280
    Klocmax = 1.0  # value suggested by Klatt, pg 1280

    # Critical Band Filter Definitions (Center Frequency and Bandwidths in Hz)
    cent_freq = np.array([50.0000, 120.000, 190.000, 260.000, 330.000, 400.000, 470.000,
                          540.000, 617.372, 703.378, 798.717, 904.128, 1020.38, 1148.30,
                          1288.72, 1442.54, 1610.70, 1794.16, 1993.93, 2211.08, 2446.71,
                          2701.97, 2978.04, 3276.17, 3597.63])
    bandwidth = np.array([70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000, 70.0000,
                          77.3724, 86.0056, 95.3398, 105.411, 116.256, 127.914, 140.423,
                          153.823, 168.154, 183.457, 199.776, 217.153, 235.631, 255.255,
                          276.072, 298.126, 321.465, 346.136])

    bw_min = bandwidth[0]  # minimum critical bandwidth

    # Set up the critical band filters.
    # Note here that Gaussianly shaped filters are used.
    # Also, the sum of the filter weights are equivalent for each critical band filter.
    # Filter less than -30 dB and set to zero.
    min_factor = math.exp(-30.0 / (2.0 * 2.303))  # -30 dB point of filter
    crit_filter = np.empty((num_crit, n_fftby2))
    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * n_fftby2
        bw = (bandwidth[i] / max_freq) * n_fftby2
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        j = np.arange(n_fftby2)
        crit_filter[i, :] = np.exp(-11 * np.square(np.divide(j - np.floor(f0), bw)) + norm_factor)
        cond = np.greater(crit_filter[i, :], min_factor)
        crit_filter[i, :] = np.where(cond, crit_filter[i, :], 0)
    # For each frame of input speech, calculate the Weighted Spectral Slope Measure
    num_frames = int(clean_length / skiprate - (winlength / skiprate))   # number of frames
    start = 0   # starting sample
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start: start + winlength]
        processed_frame = processed_speech[start: start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)
        # (2) Compute the Power Spectrum of Clean and Processed
        # if USE_FFT_SPECTRUM:
        clean_spec = np.square(np.abs(fft(clean_frame, n_fft)))
        processed_spec = np.square(np.abs(fft(processed_frame, n_fft)))
        # else:
        #     a_vec = np.zeros(n_fft)
        #     a_vec[0: 11] = lpc(clean_frame, 10)
        #     clean_spec = np.power(np.square(abs(fft(a_vec, n_fft))), -1)
        #
        #     b_vec = np.zeros(n_fft)
        #     b_vec[0: 11] = lpc(processed_frame, 10)
        #     processed_spec = np.power(np.square(abs(fft(b_vec, n_fft))), -1)

        # (3) Compute Filterbank Output Energies (in dB scale)
        clean_energy = np.matmul(crit_filter, clean_spec[0:n_fftby2])
        processed_energy = np.matmul(crit_filter, processed_spec[0:n_fftby2])

        clean_energy = 10 * np.log10(np.maximum(clean_energy, 1E-10))
        processed_energy = 10 * np.log10(np.maximum(processed_energy, 1E-10))

        # (4) Compute Spectral Slope (dB[i+1]-dB[i])
        clean_slope = clean_energy[1:num_crit] - clean_energy[0: num_crit - 1]
        processed_slope = processed_energy[1:num_crit] - processed_energy[0: num_crit - 1]

        # (5) Find the nearest peak locations in the spectra to each critical band.
        #     If the slope is negative, we search to the left. If positive, we search to the right.
        clean_loc_peak = np.empty(num_crit - 1)
        processed_loc_peak = np.empty(num_crit - 1)

        for i in range(num_crit - 1):
            # find the peaks in the clean speech signal
            if clean_slope[i] > 0:   # search to the right
                n = i
                while (n < num_crit - 1) and (clean_slope[n] > 0):
                    n = n + 1
                clean_loc_peak[i] = clean_energy[n - 1]
            else:   # search to the left
                n = i
                while (n >= 0) and (clean_slope[n] <= 0):
                    n = n - 1
                clean_loc_peak[i] = clean_energy[n + 1]

            # find the peaks in the processed speech signal
            if processed_slope[i] > 0:   # search to the right
                n = i
                while (n < num_crit - 1) and (processed_slope[n] > 0):
                    n = n + 1
                processed_loc_peak[i] = processed_energy[n - 1]
            else:   # search to the left
                n = i
                while (n >= 0) and (processed_slope[n] <= 0):
                    n = n - 1
                processed_loc_peak[i] = processed_energy[n + 1]

        # (6) Compute the WSS Measure for this frame. This includes determination of the weighting function.
        dBMax_clean = np.max(clean_energy)
        dBMax_processed = np.max(processed_energy)
        '''
        The weights are calculated by averaging individual weighting factors from the clean and processed frame.
        These weights W_clean and W_processed should range from 0 to 1 and place more emphasis on spectral peaks
        and less emphasis on slope differences in spectral valleys.
        This procedure is described on page 1280 of Klatt's 1982 ICASSP paper.
        '''
        Wmax_clean = np.divide(Kmax, Kmax + dBMax_clean - clean_energy[0: num_crit - 1])
        Wlocmax_clean = np.divide(Klocmax, Klocmax + clean_loc_peak - clean_energy[0: num_crit - 1])
        W_clean = np.multiply(Wmax_clean, Wlocmax_clean)

        Wmax_processed = np.divide(Kmax, Kmax + dBMax_processed - processed_energy[0: num_crit - 1])
        Wlocmax_processed = np.divide(Klocmax, Klocmax + processed_loc_peak - processed_energy[0: num_crit - 1])
        W_processed = np.multiply(Wmax_processed, Wlocmax_processed)

        W = np.divide(np.add(W_clean, W_processed), 2.0)
        slope_diff = np.subtract(clean_slope, processed_slope)[0: num_crit - 1]
        distortion[frame_count] = np.dot(W, np.square(slope_diff)) / np.sum(W)
        # this normalization is not part of Klatt's paper, but helps to normalize the measure.
        # Here we scale the measure by the sum of the weights.
        start = start + skiprate
    return distortion


def llr(clean_speech, processed_speech,sample_rate):
    # Check the length of the clean and processed speech.  Must be the same.
    clean_length = np.size(clean_speech)
    processed_length = np.size(processed_speech)
    if clean_length != processed_length:
        raise ValueError('Both Speech Files must be same length.')

    # Global Variables
    winlength = (np.round(30 * sample_rate / 1000)).astype(int)  # window length in samples
    skiprate = (np.floor(winlength / 4)).astype(int)   # window skip in samples
    if sample_rate < 10000:
        P = 10    # LPC Analysis Order
    else:
        P = 16    # this could vary depending on sampling frequency.

    # For each frame of input speech, calculate the Log Likelihood Ratio
    num_frames = int((clean_length - winlength) / skiprate)   # number of frames
    start = 0   # starting sample
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    distortion = np.empty(num_frames)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start: start + winlength]
        processed_frame = processed_speech[start: start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Get the autocorrelation lags and LPC parameters used to compute the LLR measure.
        R_clean, Ref_clean, A_clean = lpcoeff(clean_frame, P)
        R_processed, Ref_processed, A_processed = lpcoeff(processed_frame, P)

        # (3) Compute the LLR measure
        numerator = np.dot(np.matmul(A_processed, toeplitz(R_clean)), A_processed)
        denominator = np.dot(np.matmul(A_clean, toeplitz(R_clean)), A_clean)
        distortion[frame_count] = math.log(numerator / denominator)
        start = start + skiprate
    return distortion


def lpcoeff(speech_frame, model_order):
    # (1) Compute Autocorrelation Lags
    winlength = np.size(speech_frame)
    R = np.empty(model_order + 1)
    E = np.empty(model_order + 1)
    for k in range(model_order + 1):
        R[k] = np.dot(speech_frame[0:winlength - k], speech_frame[k: winlength])

    # (2) Levinson-Durbin
    a = np.ones(model_order)
    a_past = np.empty(model_order)
    rcoeff = np.empty(model_order)
    E[0] = R[0]
    for i in range(model_order):
        a_past[0: i] = a[0: i]
        sum_term = np.dot(a_past[0: i], R[i:0:-1])
        rcoeff[i] = (R[i + 1] - sum_term) / E[i]
        a[i] = rcoeff[i]
        if i == 0:
            a[0: i] = a_past[0: i] - np.multiply(a_past[i - 1:-1:-1], rcoeff[i])
        else:
            a[0: i] = a_past[0: i] - np.multiply(a_past[i - 1::-1], rcoeff[i])
        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]
    acorr = R
    refcoeff = rcoeff
    lpparams = np.concatenate((np.array([1]), -a))
    return acorr, refcoeff, lpparams


def snr(clean_speech, processed_speech, sample_rate):
    # Check the length of the clean and processed speech. Must be the same.
    clean_length = len(clean_speech)
    processed_length = len(processed_speech)
    if clean_length != processed_length:
        raise ValueError('Both Speech Files must be same length.')

    # Scale both clean speech and processed speech to have same dynamic range.
    # Also remove DC component from each signal.
    # clean_speech = clean_speech - np.mean(clean_speech)
    # processed_speech = processed_speech - np.mean(processed_speech)
    # processed_speech = np.multiply(processed_speech, np.max(abs(clean_speech)/ np.max(abs(processed_speech))))

    overall_snr = 10 * np.log10(np.sum(np.square(clean_speech)) / np.sum(np.square(clean_speech - processed_speech)))

    # Global Variables
    winlength = round(30 * sample_rate / 1000)    # window length in samples
    skiprate = math.floor(winlength / 4)     # window skip in samples
    MIN_SNR = -10    # minimum SNR in dB
    MAX_SNR = 35     # maximum SNR in dB

    # For each frame of input speech, calculate the Segmental SNR
    num_frames = int(clean_length / skiprate - (winlength / skiprate))   # number of frames
    start = 0      # starting sample
    window = 0.5 * (1 - np.cos(2 * math.pi * np.arange(1, winlength + 1) / (winlength + 1)))

    segmental_snr = np.empty(num_frames)
    EPS = np.spacing(1)
    for frame_count in range(num_frames):
        # (1) Get the Frames for the test and reference speech. Multiply by Hanning Window.
        clean_frame = clean_speech[start:start + winlength]
        processed_frame = processed_speech[start:start + winlength]
        clean_frame = np.multiply(clean_frame, window)
        processed_frame = np.multiply(processed_frame, window)

        # (2) Compute the Segmental SNR
        signal_energy = np.sum(np.square(clean_frame))
        noise_energy = np.sum(np.square(clean_frame - processed_frame))
        segmental_snr[frame_count] = 10 * math.log10(signal_energy / (noise_energy + EPS) + EPS)
        segmental_snr[frame_count] = max(segmental_snr[frame_count], MIN_SNR)
        segmental_snr[frame_count] = min(segmental_snr[frame_count], MAX_SNR)

        start = start + skiprate

    return overall_snr, segmental_snr


def stoi(x, y, fs_signal):
    if np.size(x) != np.size(y):
        raise ValueError('x and y should have the same length')

    # initialization, pay attention to the range of x and y(divide by 32768?)
    fs = 10000    # sample rate of proposed intelligibility measure
    N_frame = 256    # window support
    K = 512     # FFT size
    J = 15      # Number of 1/3 octave bands
    mn = 150    # Center frequency of first 1/3 octave band in Hz
    H, _ = thirdoct(fs, K, J, mn)     # Get 1/3 octave band matrix
    N = 30    # Number of frames for intermediate intelligibility measure (Length analysis window)
    Beta = -15     # lower SDR-bound
    dyn_range = 40     # speech dynamic range

    # resample signals if other sample rate is used than fs
    if fs_signal != fs:
        x = signal.resample_poly(x, fs, fs_signal)
        y = signal.resample_poly(y, fs, fs_signal)

    # remove silent frames
    x, y = removeSilentFrames(x, y, dyn_range, N_frame, int(N_frame / 2))

    # apply 1/3 octave band TF-decomposition
    x_hat = stdft(x, N_frame, N_frame / 2, K)    # apply short-time DFT to clean speech
    y_hat = stdft(y, N_frame, N_frame / 2, K)    # apply short-time DFT to processed speech

    x_hat = np.transpose(x_hat[:, 0:(int(K / 2) + 1)])    # take clean single-sided spectrum
    y_hat = np.transpose(y_hat[:, 0:(int(K / 2) + 1)])    # take processed single-sided spectrum

    X = np.sqrt(np.matmul(H, np.square(np.abs(x_hat))))  # apply 1/3 octave bands as described in Eq.(1) [1]
    Y = np.sqrt(np.matmul(H, np.square(np.abs(y_hat))))

    # loop al segments of length N and obtain intermediate intelligibility measure for all TF-regions
    d_interm = np.zeros(np.size(np.arange(N - 1, x_hat.shape[1])))
    # init memory for intermediate intelligibility measure
    c = 10 ** (-Beta / 20)
    # constant for clipping procedure

    for m in range(N - 1, x_hat.shape[1]):
        X_seg = X[:, (m - N + 1): (m + 1)]    # region with length N of clean TF-units for all j
        Y_seg = Y[:, (m - N + 1): (m + 1)]    # region with length N of processed TF-units for all j
        # obtain scale factor for normalizing processed TF-region for all j
        alpha = np.sqrt(np.divide(np.sum(np.square(X_seg), axis=1, keepdims=True),
                                  np.sum(np.square(Y_seg), axis=1, keepdims=True)))
        # obtain \alpha*Y_j(n) from Eq.(2) [1]
        aY_seg = np.multiply(Y_seg, alpha)
        # apply clipping from Eq.(3)
        Y_prime = np.minimum(aY_seg, X_seg + X_seg * c)
        # obtain correlation coeffecient from Eq.(4) [1]
        d_interm[m - N + 1] = taa_corr(X_seg, Y_prime) / J

    d = d_interm.mean()    # combine all intermediate intelligibility measures as in Eq.(4) [1]
    return d


def thirdoct(fs, N_fft, numBands, mn):
    """
    [A CF] = THIRDOCT(FS, N_FFT, NUMBANDS, MN) returns 1/3 octave band matrix
    inputs:
        FS:         samplerate
        N_FFT:      FFT size
        NUMBANDS:   number of bands
        MN:         center frequency of first 1/3 octave band
    outputs:
        A:          octave band matrix
        CF:         center frequencies
    """
    f = np.linspace(0, fs, N_fft + 1)
    f = f[0:int(N_fft / 2 + 1)]
    k = np.arange(numBands)
    cf = np.multiply(np.power(2, k / 3), mn)
    fl = np.sqrt(np.multiply(np.multiply(np.power(2, k / 3), mn), np.multiply(np.power(2, (k - 1) / 3), mn)))
    fr = np.sqrt(np.multiply(np.multiply(np.power(2, k / 3), mn), np.multiply(np.power(2, (k + 1) / 3), mn)))
    A = np.zeros((numBands, len(f)))

    for i in range(np.size(cf)):
        b = np.argmin((f - fl[i]) ** 2)
        fl[i] = f[b]
        fl_ii = b

        b = np.argmin((f - fr[i]) ** 2)
        fr[i] = f[b]
        fr_ii = b
        A[i, fl_ii: fr_ii] = 1

    rnk = np.sum(A, axis=1)
    end = np.size(rnk)
    rnk_back = rnk[1: end]
    rnk_before = rnk[0: (end-1)]
    for i in range(np.size(rnk_back)):
        if (rnk_back[i] >= rnk_before[i]) and (rnk_back[i] != 0):
            result = i
    numBands = result + 2
    A = A[0:numBands, :]
    cf = cf[0:numBands]
    return A, cf


def stdft(x, N, K, N_fft):
    """
    X_STDFT = X_STDFT(X, N, K, N_FFT) returns the short-time hanning-windowed dft of X with frame-size N,
    overlap K and DFT size N_FFT. The columns and rows of X_STDFT denote the frame-index and dft-bin index,
    respectively.
    """
    frames_size = int((np.size(x) - N) / K)
    w = signal.windows.hann(N+2)
    w = w[1: N+1]

    x_stdft = signal.stft(x, window=w, nperseg=N, noverlap=K, nfft=N_fft, return_onesided=False, boundary=None)[2]
    x_stdft = np.transpose(x_stdft)[0:frames_size, :]

    return x_stdft


def removeSilentFrames(x, y, dyrange, N, K):
    """
    [X_SIL Y_SIL] = REMOVESILENTFRAMES(X, Y, RANGE, N, K) X and Y are segmented with frame-length N
    and overlap K, where the maximum energy of all frames of X is determined, say X_MAX.
    X_SIL and Y_SIL are the reconstructed signals, excluding the frames, where the energy of a frame
    of X is smaller than X_MAX-RANGE
    """

    frames = np.arange(0, (np.size(x) - N), K)
    w = signal.windows.hann(N+2)
    w = w[1: N+1]

    jj_list = np.empty((np.size(frames), N), dtype=int)
    for j in range(np.size(frames)):
        jj_list[j, :] = np.arange(frames[j] - 1, frames[j] + N - 1)

    msk = 20 * np.log10(np.divide(norm(np.multiply(x[jj_list], w), axis=1), np.sqrt(N)))

    msk = (msk - np.max(msk) + dyrange) > 0
    count = 0

    x_sil = np.zeros(np.size(x))
    y_sil = np.zeros(np.size(y))

    for j in range(np.size(frames)):
        if msk[j]:
            jj_i = np.arange(frames[j], frames[j] + N)
            jj_o = np.arange(frames[count], frames[count] + N)
            x_sil[jj_o] = x_sil[jj_o] + np.multiply(x[jj_i], w)
            y_sil[jj_o] = y_sil[jj_o] + np.multiply(y[jj_i], w)
            count = count + 1

    x_sil = x_sil[0: jj_o[-1] + 1]
    y_sil = y_sil[0: jj_o[-1] + 1]
    return x_sil, y_sil


def taa_corr(x, y):
    """
    RHO = TAA_CORR(X, Y) Returns correlation coeffecient between column
    vectors x and y. Gives same results as 'corr' from statistics toolbox.
    """
    xn = np.subtract(x, np.mean(x, axis=1, keepdims=True))
    xn = np.divide(xn, norm(xn, axis=1, keepdims=True))
    yn = np.subtract(y, np.mean(y, axis=1, keepdims=True))
    yn = np.divide(yn, norm(yn, axis=1, keepdims=True))
    rho = np.trace(np.matmul(xn, np.transpose(yn)))

    return rho


def phase_align(target, reference):
    '''
    Cross-correlate Datapipeline within region of interest at a precision of 1./res
    if Datapipeline is cross-correlated at native resolution (i.e. res=1) this function
    can only achieve integer precision

    Args:
        target (1d array/list): signal that won't be shifted
        reference (1d array/list): signal to be shifted to target
        roi (tuple): region of interest to compute chi-squared
        res (int): factor to increase resolution of Datapipeline via linear interpolation

    Returns:
        shift (float): offset between target and reference signal
    '''
    r1 = target
    r2 = reference

    maxlag = max(len(r1), len(r2)) - 1
    m = max(len(r1), len(r2))
    maxlagDefault = m - 1
    mxl = min(maxlag, maxlagDefault)
    m2 = findTransformLength(m)
    c1 = ifft(np.multiply(fft(r1, m2), fft(r2, m2).conj()))
    # Perform the cross-correlation
    c = np.concatenate((c1[m2 - mxl + np.arange(mxl)], c1[0: mxl + 1]))
    cxx0 = np.dot(r1, r1)
    cyy0 = np.dot(r2, r2)
    c_normalized = np.divide(np.abs(c), np.sqrt(np.multiply(cxx0, cyy0)))

    max_c_pos = np.max(c_normalized[maxlag:])
    index_max_pos = np.argmax(c_normalized[maxlag:])
    max_c_neg = np.max(np.flipud(c_normalized[:maxlag]))
    index_max_neg = np.argmax(np.flipud(c_normalized[:maxlag]))

    if np.size(max_c_neg) == 0:
        index_max = maxlag + index_max_pos
    else:
        if max_c_pos > max_c_neg:
            index_max = maxlag + index_max_pos
            max_c = max_c_pos
        elif max_c_pos < max_c_neg:
            index_max = maxlag + 1 - index_max_neg
            max_c = max_c_neg
        elif max_c_pos == max_c_neg:
            if index_max_pos <= index_max_neg:
                index_max = maxlag + index_max_pos
                max_c = max_c_pos
            else:
                index_max = maxlag + 1 - index_max_neg
                max_c = max_c_neg

    d = maxlag - index_max
    if max_c < 1e-8:
        d = 0

    return d

    # # subtract mean
    # r1 -= r1.mean()
    # r2 -= r2.mean()
    #
    # # compute cross covariance
    # cc = ccovf(r1, r2, demean=False,unbiased=False)
    #
    # # determine if shift if positive/negative
    # if np.argmax(cc) == 0:
    #     cc = ccovf(r2, r1, demean=False, unbiased=False)
    #     mod = -1
    # else:
    #     mod = 1
    #
    # # often found this method to be more accurate then the way below
    # return np.argmax(cc)*mod

def findTransformLength(m):
    m = 2 * m
    while True:
        r = m
        for p in [2, 3, 5, 7]:
            while (r > 1) and (np.mod(r, p) == 0):
                r = r / p
        if r == 1:
            break
        m = m + 1
    return m


def align_signal(data1, data2):
    delay = int(phase_align(data1, data2))
    if delay == 0:   # data1 and data2 are already aligned
        data1_aligned = data1
        data2_aligned = data2
    elif delay > 0:   # data2 is estimated to be delayed with respect to data1
        data1_aligned = np.concatenate((np.zeros(delay), data1))
        data2_aligned = data2
    else:   # data1 is estimated to be delayed with respect to data2
        data2_aligned = np.concatenate((np.zeros(-delay), data2))
        data1_aligned = data1
    return data1_aligned, data2_aligned


def pretty_table(rows, column_count, column_spacing=4):
    aligned_columns = []
    for column in range(column_count):
        column_data = list(map(lambda row: row[column], rows))
        aligned_columns.append((max(map(len, column_data)) + column_spacing, column_data))

    for row in range(len(rows)):
        aligned_row = map(lambda x: (x[0], x[1][row]), aligned_columns)
        yield ''.join(map(lambda x: x[1] + ' ' * (x[0] - len(x[1])), aligned_row))


'''
PESQ objective speech quality measure (narrowband and wideband implementations)

This function implements the PESQ measure based on the ITU standards P.862 [1] and 
P.862.1 [2] for narrowband speech and P.862.2 for wideband speech [3].

Usage:  scores = pesq2( clean_data, enhanced_data, sampling_rate, mode )

Example call:  scores = pesq2(clean_data, enhanced_data, 8000, 'narrowband')

References:
[1] ITU (2000). Perceptual evaluation of speech quality (PESQ), and 
    objective method for end-to-end speech quality assessment of 
    narrowband telephone networks and speech codecs. ITU-T
    Recommendation P.862 

[2] ITU (2003).  Mapping function for transforming P.862 raw result 
    scores to MOS-LQO, ITU-T Recommendation P. 862.1 

[3] ITU (2007). Wideband extension to Recommendation P.862 for the
    assessment of wideband telephone networks and speech codecs. ITU-T
    Recommendation P.862.2

Authors: Yi Hu, Kamil Wojcicki and Philipos C. Loizou 

Copyright (c) 2006, 2012 by Philipos C. Loizou
$Revision: 2.0 $  $Date: 5/14/2012 $
'''


def pesq2(ref_data, deg_data, sampling_rate, mode):
    '''
    if not isinstance(ref_wav, str):
        raise TypeError('First input argument has to be a reference wav filename as string.'
                        'For usage help type: help pesq')
    if not isinstance(deg_wav, str):
        raise TypeError('Second input argument has to be a processed wav filename as string.'
                        'For usage help type: help pesq')

    if not os.path.exists(ref_wav):
        raise OSError('Reference wav file: %s not found.' % ref_wav)
    if not os.path.exists(deg_wav):
        raise OSError('Processed wav file: %s not found.' % deg_wav)

    ref_sampling_rate, ref_data = wavfile.read(ref_wav)
    deg_sampling_rate, deg_data = wavfile.read(deg_wav)

    if ref_sampling_rate != deg_sampling_rate:
        raise ValueError('Sampling rate mismatch.\n'
                         'The sampling rate of the reference wav file (%d Hz) has to match '
                         'sampling rate of the degraded wav file (%d Hz).' % (ref_sampling_rate, deg_sampling_rate))
    else:
        sampling_rate = ref_sampling_rate

    if sampling_rate == 8000:
        mode = 'narrowband'
    elif sampling_rate == 16000:
        mode = 'wideband'
    else:
        raise ValueError('Error: Unsupported sampling rate (%d Hz).\n'
                         'Only sampling rates of 8000 Hz (for narrowband assessment)\n'
                         'and 16000 Hz (for wideband assessment) are supported.' % sampling_rate)
    '''

    global Downsample, DATAPADDING_MSECS, SEARCHBUFFER, Fs, WHOLE_SIGNAL
    global Align_Nfft, Window

    setup_global(sampling_rate)
    twopi = 6.28318530717959
    Window = np.zeros(Align_Nfft)  # 增加的初始化
    for count in range(Align_Nfft):
        Window[count] = 0.5 * (1.0 - np.cos((twopi * count) / Align_Nfft))

    ref_Nsamples = len(ref_data) + 2 * SEARCHBUFFER * Downsample
    ref_data = np.concatenate((np.zeros(SEARCHBUFFER * Downsample), ref_data))
    ref_data = np.concatenate((ref_data, np.zeros(DATAPADDING_MSECS * int(Fs / 1000) + SEARCHBUFFER * Downsample)))

    deg_Nsamples = len(deg_data) + 2 * SEARCHBUFFER * Downsample
    deg_data = np.concatenate((np.zeros(SEARCHBUFFER * Downsample), deg_data))
    deg_data = np.concatenate((deg_data, np.zeros(DATAPADDING_MSECS * int(Fs / 1000) + SEARCHBUFFER * Downsample)))

    maxNsamples = max(ref_Nsamples, deg_Nsamples)
    ref_data = fix_power_level(ref_data, ref_Nsamples, maxNsamples)
    deg_data = fix_power_level(deg_data, deg_Nsamples, maxNsamples)

    if mode == 'narrowband':
        standard_IRS_filter_dB = np.array([[0, -200], [50, -40], [100, -20], [125, -12], [160, -6], [200, 0],
                                           [250, 4], [300, 6], [350, 8], [400, 10], [500, 11], [600, 12], [700, 12],
                                           [800, 12], [1000, 12], [1300, 12], [1600, 12], [2000, 12], [2500, 12],
                                           [3000, 12], [3250, 12], [3500, 4], [4000, -200], [5000, -200],
                                           [6300, -200], [8000, -200]])

        ref_data = apply_filter(ref_data, ref_Nsamples, standard_IRS_filter_dB)
        deg_data = apply_filter(deg_data, deg_Nsamples, standard_IRS_filter_dB)
    else:
        ref_data = apply_filters_WB(ref_data, ref_Nsamples)
        deg_data = apply_filters_WB(deg_data, deg_Nsamples)

    model_ref = ref_data.copy()
    model_deg = deg_data.copy()

    ref_data, deg_data = input_filter(ref_data, ref_Nsamples, deg_data, deg_Nsamples)

    ref_VAD, ref_logVAD = apply_VAD(ref_data, ref_Nsamples)
    deg_VAD, deg_logVAD = apply_VAD(deg_data, deg_Nsamples)

    crude_align(ref_logVAD, ref_Nsamples, deg_logVAD, deg_Nsamples, WHOLE_SIGNAL)

    utterance_locate(ref_data, ref_Nsamples, ref_VAD, ref_logVAD, deg_data, deg_Nsamples, deg_VAD, deg_logVAD)

    # make ref_data and deg_data equal length
    if ref_Nsamples < deg_Nsamples:
        newlen = deg_Nsamples + DATAPADDING_MSECS * int(Fs / 1000)
        addlen = newlen - len(model_ref)
        model_ref = np.append(model_ref, np.zeros(addlen))
    elif ref_Nsamples > deg_Nsamples:
        newlen = ref_Nsamples + DATAPADDING_MSECS * int(Fs / 1000)
        addlen = newlen - len(model_deg)
        model_deg = np.append(model_deg, np.zeros(addlen))

    pesq_mos = pesq_psychoacoustic_model(model_ref, ref_Nsamples, model_deg, deg_Nsamples)

    if mode == 'narrowband':
        # NB: P.862.1->P.800.1 (PESQ_MOS->MOS_LQO)
        mos_lqo = 0.999 + (4.999 - 0.999) / (1 + math.exp(-1.4945 * pesq_mos + 4.6607))
        # scores = [pesq_mos, mos_lqo]
        scores = mos_lqo
    else:
        mos_lqo = 0.999 + (4.999 - 0.999) / (1 + math.exp(-1.3669 * pesq_mos + 3.8224))
        scores = mos_lqo

    # delvars()
    return scores


def delvars():
    global Downsample, DATAPADDING_MSECS, SEARCHBUFFER, Fs, WHOLE_SIGNAL, Align_Nfft, Window
    globals().clear()


def apply_filter(data, data_Nsamples, align_filter_dB):
    global Downsample, DATAPADDING_MSECS, SEARCHBUFFER, Fs

    align_filtered = data.copy()
    n = data_Nsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000)
    # now find the next power of 2 which is greater or equal to n
    pow_of_2 = pow(2, math.ceil(math.log2(n)))

    overallGainFilter = np.interp(1000, align_filter_dB[:, 0], align_filter_dB[:, 1])

    x = np.zeros(pow_of_2)
    x[0: n] = data[SEARCHBUFFER * Downsample: SEARCHBUFFER * Downsample + n]

    x_fft = fft(x, pow_of_2)

    freq_resolution = Fs / pow_of_2

    factorDb = np.zeros(int(pow_of_2 / 2) + 1)
    factorDb[0: int(pow_of_2 / 2) + 1] = np.interp(np.arange(0, int(pow_of_2 / 2) + 1) * freq_resolution,
                                                   align_filter_dB[:, 0], align_filter_dB[:, 1]) - overallGainFilter
    factor = pow(10, factorDb / 20)

    factor = np.concatenate((factor, factor[(int(pow_of_2 / 2) - 1):0:-1]))
    x_fft = np.multiply(x_fft, factor)

    y = ifft(x_fft, int(pow_of_2)).real
    align_filtered[SEARCHBUFFER * Downsample: SEARCHBUFFER * Downsample + n] = y[0: n]

    return align_filtered


def apply_filters(data, Nsamples):
    global InIIR_Hsos, InIIR_Nsos, DATAPADDING_MSECS, Fs
    # now we construct the second order section matrix
    sosMatrix = np.zeros((InIIR_Nsos, 6))
    sosMatrix[:, 3] = 1  # set a(1) to 1
    # each row of sosMatrix holds [b(1*3) a(1*3)] for each section
    sosMatrix[:, 0: 3] = InIIR_Hsos[:, 0: 3]
    sosMatrix[:, 4: 6] = InIIR_Hsos[:, 3: 5]
    # sosMatrix

    # now we construct second order section direct form II filter
    mod_data = signal.sosfilt(sosMatrix, data)
    return mod_data


def apply_filters_WB(data, Nsamples):
    global WB_InIIR_Hsos, WB_InIIR_Nsos, DATAPADDING_MSECS, Fs

    # now we construct the second order section matrix
    sosMatrix = np.zeros((WB_InIIR_Nsos, 6))
    sosMatrix[:, 3] = 1  # set a(1) to 1

    # each row of sosMatrix holds[b(1 * 3) a(1 * 3)] for each section
    sosMatrix[:, 0:3] = WB_InIIR_Hsos[:, 0:3]
    sosMatrix[:, 4:6] = WB_InIIR_Hsos[:, 3:5]
    # sosMatrix

    # now we construct second order section direct form II filter
    # 存在较大疑问 从sos矩阵中返回的对象是什么 returns a direct-form II filter object?
    mod_data = signal.sosfilt(sosMatrix, data)
    return mod_data


def apply_VAD(data, Nsamples):
    global Downsample, MINSPEECHLGTH, JOINSPEECHLGTH

    Nwindows = math.floor(Nsamples / Downsample)
    # number of 4ms window
    VAD = np.zeros((Nwindows))
    for count in range(Nwindows):
        VAD[count] = np.sum(np.square(data[count * Downsample: (count + 1) * Downsample])) / Downsample
    # VAD is the power of each 4ms window

    LevelThresh = np.sum(VAD) / Nwindows
    # LevelThresh is set to mean value of VAD
    LevelMin = np.max(VAD)
    if LevelMin > 0:
        LevelMin = LevelMin * 1.0e-4
    else:
        LevelMin = 1.0

    VAD[np.where(VAD < LevelMin)] = LevelMin

    for iteration in range(12):
        LevelNoise = 0
        length = 0
        StDNoise = 0

        VAD_lessthan_LevelThresh = VAD[np.where(VAD <= LevelThresh)]
        length = len(VAD_lessthan_LevelThresh)
        LevelNoise = np.sum(VAD_lessthan_LevelThresh)
        if length > 0:
            LevelNoise = LevelNoise / length
            StDNoise = math.sqrt(np.sum(np.power((VAD_lessthan_LevelThresh - LevelNoise), 2)) / length)
        LevelThresh = 1.001 * (LevelNoise + 2 * StDNoise)

    LevelNoise = 0
    LevelSig = 0
    length = 0
    VAD_greaterthan_LevelThresh = VAD[np.where(VAD > LevelThresh)]
    length = len(VAD_greaterthan_LevelThresh)
    LevelSig = np.sum(VAD_greaterthan_LevelThresh)

    VAD_lessorequal_LevelThresh = VAD[np.where(VAD <= LevelThresh)]
    LevelNoise = np.sum(VAD_lessorequal_LevelThresh)

    if length > 0:
        LevelSig = LevelSig / length
    else:
        LevelThresh = -1

    if length < Nwindows:
        LevelNoise = LevelNoise / (Nwindows - length)
    else:
        LevelNoise = 1

    VAD[np.where(VAD <= LevelThresh)] = -VAD[np.where(VAD <= LevelThresh)]
    VAD[0] = -LevelMin
    VAD[Nwindows - 1] = -LevelMin

    start = 0
    finish = 0
    for count in range(2, Nwindows + 1):
        if (VAD[count - 1] > 0.0) and (VAD[count - 2] <= 0.0):
            start = count
        if (VAD[count - 1] <= 0.0) and (VAD[count - 2] > 0.0):
            finish = count
            if (finish - start) <= MINSPEECHLGTH:
                VAD[start - 1: finish - 1] = -VAD[start - 1: finish - 1]
    # to make sure finish- start is more than 4

    if LevelSig >= (LevelNoise * 1000):
        for count in range(2, Nwindows + 1):
            if (VAD[count - 1] > 0) and (VAD[count - 2] <= 0):
                start = count
            if (VAD[count - 1] <= 0) and (VAD[count - 2] > 0):
                finish = count
                g = np.sum(VAD[start - 1: finish - 1])
                if g < 3.0 * LevelThresh * (finish - start):
                    VAD[start - 1: finish - 1] = -VAD[start - 1: finish - 1]

    start = 0
    finish = 0
    for count in range(2, Nwindows + 1):
        if (VAD[count - 1] > 0.0) and (VAD[count - 2] <= 0.0):
            start = count
            if (finish > 0) and ((start - finish) <= JOINSPEECHLGTH):
                VAD[finish - 1: start - 1] = LevelMin
        if (VAD[count - 1] <= 0.0) and (VAD[count - 2] > 0.0):
            finish = count

    start = 0
    for count in range(2, Nwindows + 1):
        if (VAD[count - 1] > 0) and (VAD[count - 2] <= 0):
            start = count
    if start == 0:
        VAD = np.fabs(VAD)
        VAD[0] = -LevelMin
        VAD[Nwindows - 1] = -LevelMin

    count = 3  # 索引和下一行的控制项都减了1
    while count < (Nwindows - 2):
        if (VAD[count] > 0) and (VAD[count - 2] <= 0):
            VAD[count - 2] = VAD[count] * 0.1
            VAD[count - 1] = VAD[count] * 0.3
            count = count + 1
        if (VAD[count] <= 0) and (VAD[count - 1] > 0):
            VAD[count] = VAD[count - 1] * 0.3
            VAD[count + 1] = VAD[count - 1] * 0.1
            count = count + 3
        count = count + 1

    VAD[np.where(VAD < 0)] = 0

    if LevelThresh <= 0:
        LevelThresh = LevelMin

    logVAD = np.zeros(len(VAD))
    VAD_greaterthan_LevelThresh = np.where(VAD > LevelThresh)
    logVAD[VAD_greaterthan_LevelThresh] = np.log(VAD[VAD_greaterthan_LevelThresh] / LevelThresh)

    return VAD, logVAD


def crude_align(ref_logVAD, ref_Nsamples, deg_logVAD, deg_Nsamples, Utt_id):
    global Downsample
    global Nutterances, Largest_uttsize, Nsurf_samples, Crude_DelayEst
    global Crude_DelayConf, UttSearch_Start, UttSearch_End, Utt_DelayEst
    global Utt_Delay, Utt_DelayConf, Utt_Start, Utt_End
    global MAXNUTTERANCES, WHOLE_SIGNAL
    global pesq_mos, subj_mos, cond_nr

    if Utt_id == WHOLE_SIGNAL:
        nr = math.floor(ref_Nsamples / Downsample)
        nd = math.floor(deg_Nsamples / Downsample)
        startr = 1
        startd = 1
    elif Utt_id == MAXNUTTERANCES:
        startr = int(UttSearch_Start[MAXNUTTERANCES - 1])
        startd = startr + int(Utt_DelayEst[MAXNUTTERANCES - 1] / Downsample)
        if startd < 0:
            startr = 1 - int(Utt_DelayEst[MAXNUTTERANCES - 1] / Downsample)
            startd = 1

        nr = int(UttSearch_End[MAXNUTTERANCES - 1]) - startr
        nd = nr

        if startd + nd > math.floor(deg_Nsamples / Downsample):
            nd = math.floor(deg_Nsamples / Downsample) - startd

    else:
        startr = int(UttSearch_Start[Utt_id - 1])
        startd = startr + int(Crude_DelayEst / Downsample)
        if startd < 0:
            startr = 1 - int(Crude_DelayEst / Downsample)
            startd = 1

        nr = int(UttSearch_End[Utt_id - 1]) - startr
        nd = nr
        if startd + nd > math.floor(deg_Nsamples / Downsample) + 1:
            nd = math.floor(deg_Nsamples / Downsample) - startd + 1

    startr = max(1, startr)
    startd = max(1, startd)

    max_Y = 0.0
    I_max_Y = nr
    if (nr > 1) and (nd > 1):
        y = FFTNXCorr(ref_logVAD, startr, nr, deg_logVAD, startd, nd)
        max_Y = np.max(y)
        I_max_Y = np.argmax(y) + 1  # 此处加了1
        if max_Y <= 0:
            max_Y = 0
            I_max_Y = nr

    if Utt_id == WHOLE_SIGNAL:
        Crude_DelayEst = (I_max_Y - nr) * Downsample
        Crude_DelayConf = 0.0
    elif Utt_id == MAXNUTTERANCES:
        Utt_Delay[MAXNUTTERANCES - 1] = (I_max_Y - nr) * Downsample + Utt_DelayEst[MAXNUTTERANCES - 1]
    else:
        Utt_DelayEst[Utt_id - 1] = (I_max_Y - nr) * Downsample + Crude_DelayEst


def DC_block(data, Nsamples):
    global Downsample, DATAPADDING_MSECS, SEARCHBUFFER

    ofs = SEARCHBUFFER * Downsample
    mod_data = data

    # compute dc component, it is a little weird
    facc = np.sum(data[ofs: Nsamples - ofs]) / Nsamples
    mod_data[ofs: Nsamples - ofs] = data[ofs: Nsamples - ofs] - facc

    mod_data[ofs: ofs + Downsample] = np.multiply(mod_data[ofs: ofs + Downsample],
                                                  (0.5 + np.arange(Downsample)) / Downsample)
    mod_data[Nsamples - ofs - 1: Nsamples - ofs - Downsample - 1: -1] = np.multiply(
        mod_data[Nsamples - ofs - 1: Nsamples - ofs - Downsample - 1: -1], (0.5 + np.arange(Downsample)) / Downsample)
    return mod_data


def FFTNXCorr(ref_VAD, startr, nr, deg_VAD, startd, nd):
    x1 = ref_VAD[startr - 1: startr + nr - 1]
    x2 = deg_VAD[startd - 1: startd + nd - 1]
    x1 = x1[::-1]
    y = signal.convolve(x2, x1)
    return y


def fix_power_level(data, data_Nsamples, maxNsamples):
    # this function is used for level normalization, i.e., to fix the power
    # level of Datapipeline to a preset number, and return it to mod_data.

    global Downsample, DATAPADDING_MSECS, SEARCHBUFFER, Fs
    global TARGET_AVG_POWER
    TARGET_AVG_POWER = 1e7

    align_filter_dB = np.array(
        [[0, -500], [50, -500], [100, -500], [125, -500], [160, -500], [200, -500], [250, -500], [300, -500],
         [350, 0], [400, 0], [500, 0], [600, 0], [630, 0], [800, 0], [1000, 0], [1250, 0], [1600, 0], [2000, 0],
         [2500, 0], [3000, 0], [3250, 0], [3500, -500], [4000, -500], [5000, -500], [6300, -500], [8000, -500]])

    align_filtered = apply_filter(data, data_Nsamples, align_filter_dB)
    power_above_300Hz = pow_of(align_filtered, SEARCHBUFFER * Downsample + 1,
                               data_Nsamples - SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000),
                               maxNsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000))

    global_scale = math.sqrt(TARGET_AVG_POWER / power_above_300Hz)
    mod_data = np.multiply(data, global_scale)
    return mod_data


def id_searchwindows(ref_VAD, ref_Nsamples, deg_VAD, deg_Nsamples):
    global MINUTTLENGTH, Downsample, SEARCHBUFFER
    global Crude_DelayEst, Nutterances, UttSearch_Start, UttSearch_End

    Utt_num = 1
    speech_flag = 0

    VAD_length = math.floor(ref_Nsamples / Downsample)
    del_deg_start = MINUTTLENGTH - int(Crude_DelayEst / Downsample)
    del_deg_end = math.floor((deg_Nsamples - Crude_DelayEst) / Downsample) - MINUTTLENGTH

    for count in range(1, VAD_length + 1):
        VAD_value = ref_VAD[count - 1]
        if (VAD_value > 0) and (speech_flag == 0):
            speech_flag = 1
            this_start = count
            UttSearch_Start[Utt_num - 1] = count - SEARCHBUFFER
            if UttSearch_Start[Utt_num - 1] < 1:
                UttSearch_Start[Utt_num - 1] = 1
        if ((VAD_value == 0) or (count == (VAD_length - 1))) and (speech_flag == 1):
            speech_flag = 0
            UttSearch_End[Utt_num - 1] = count + SEARCHBUFFER
            if UttSearch_End[Utt_num - 1] > VAD_length:
                UttSearch_End[Utt_num - 1] = VAD_length
            if ((count - this_start) >= MINUTTLENGTH) and (this_start < del_deg_end) and (count > del_deg_start):
                Utt_num = Utt_num + 1
    Utt_num = Utt_num - 1
    Nutterances = Utt_num


def id_utterances(ref_Nsamples, ref_VAD, deg_Nsamples):
    global Largest_uttsize, MINUTTLENGTH, Crude_DelayEst
    global Downsample, SEARCHBUFFER, Nutterances, Utt_Start
    global Utt_End, Utt_Delay

    Utt_num = 1
    speech_flag = 0
    VAD_length = math.floor(ref_Nsamples / Downsample)

    del_deg_start = MINUTTLENGTH - Crude_DelayEst / Downsample
    del_deg_end = math.floor((deg_Nsamples - Crude_DelayEst) / Downsample) - MINUTTLENGTH

    for count in range(1, VAD_length + 1):
        VAD_value = ref_VAD[count - 1]
        if (VAD_value > 0.0) and (speech_flag == 0):
            speech_flag = 1
            this_start = count
            Utt_Start[Utt_num - 1] = count

        if ((VAD_value == 0) or (count == VAD_length)) and (speech_flag == 1):
            speech_flag = 0
            Utt_End[Utt_num - 1] = count
            if ((count - this_start) >= MINUTTLENGTH) and (this_start < del_deg_end) and (count > del_deg_start):
                Utt_num = Utt_num + 1

    Utt_Start[0] = SEARCHBUFFER + 1
    Nutterances = max(1, Nutterances)
    Utt_End[Nutterances - 1] = VAD_length - SEARCHBUFFER + 1

    Utt_num = np.arange(1, Nutterances)
    this_start = Utt_Start[Utt_num] - 1
    last_end = Utt_End[Utt_num - 1] - 1
    count = np.floor((this_start + last_end) / 2)
    Utt_Start[Utt_num] = count + 1
    Utt_End[Utt_num - 1] = count + 1
    # for Utt_num in range(1, Nutterances):
    #     this_start = Utt_Start[Utt_num] - 1
    #     last_end = Utt_End[Utt_num - 1] - 1
    #     count = math.floor((this_start + last_end) / 2)
    #     Utt_Start[Utt_num] = count + 1
    #     Utt_End[Utt_num - 1] = count + 1

    this_start = (Utt_Start[0] - 1) * Downsample + Utt_Delay[0]
    if this_start < (SEARCHBUFFER * Downsample):
        count = SEARCHBUFFER + math.floor((Downsample - 1 - Utt_Delay[0]) / Downsample)
        Utt_Start[0] = count + 1

    last_end = (Utt_End[Nutterances - 1] - 1) * Downsample + 1 + Utt_Delay[Nutterances - 1]
    if last_end > (deg_Nsamples - SEARCHBUFFER * Downsample + 1):
        count = math.floor((deg_Nsamples - Utt_Delay[Nutterances - 1]) / Downsample) - SEARCHBUFFER
        Utt_End[Nutterances - 1] = count + 1

    for Utt_num in range(2, Nutterances + 1):
        this_start = (Utt_Start[Utt_num - 1] - 1) * Downsample + Utt_Delay[Utt_num - 1]
        last_end = (Utt_End[Utt_num - 2] - 1) * Downsample + Utt_Delay[Utt_num - 2]
        if this_start < last_end:
            count = math.floor((this_start + last_end) / 2)
            this_start = math.floor((Downsample - 1 + count - Utt_Delay[Utt_num - 1]) / Downsample)
            last_end = math.floor((count - Utt_Delay[Utt_num - 2]) / Downsample)
            Utt_Start[Utt_num - 1] = this_start + 1
            Utt_End[Utt_num - 2] = last_end + 1

    Largest_uttsize = np.max(np.subtract(Utt_End, Utt_Start))


def input_filter(ref_data, ref_Nsamples, deg_data, deg_Nsamples):
    mod_ref_data = DC_block(ref_data, ref_Nsamples)
    mod_deg_data = DC_block(deg_data, deg_Nsamples)

    mod_ref_data = apply_filters(mod_ref_data, ref_Nsamples)
    mod_deg_data = apply_filters(mod_deg_data, deg_Nsamples)
    return mod_ref_data, mod_deg_data


def pesq_psychoacoustic_model(ref_data, ref_Nsamples, deg_data, deg_Nsamples):
    global CALIBRATE, Nfmax, Nb, Sl, Sp
    global nr_of_hz_bands_per_bark_band, centre_of_band_bark
    global width_of_band_hz, centre_of_band_hz, width_of_band_bark
    global Downsample, SEARCHBUFFER, DATAPADDING_MSECS, Fs, Nutterances
    global Utt_Start, Utt_End, Utt_Delay, NUMBER_OF_PSQM_FRAMES_PER_SYLLABE
    global Fs, Plot_Frame

    Plot_Frame = -1

    NUMBER_OF_PSQM_FRAMES_PER_SYLLABE = 20

    maxNsamples = max(ref_Nsamples, deg_Nsamples)
    Nf = Downsample * 8
    MAX_NUMBER_OF_BAD_INTERVALS = 1000

    start_frame_of_bad_interval = np.zeros(MAX_NUMBER_OF_BAD_INTERVALS)
    stop_frame_of_bad_interval = np.zeros(MAX_NUMBER_OF_BAD_INTERVALS)
    start_sample_of_bad_interval = np.zeros(MAX_NUMBER_OF_BAD_INTERVALS)
    stop_sample_of_bad_interval = np.zeros(MAX_NUMBER_OF_BAD_INTERVALS)
    number_of_samples_in_bad_interval = np.zeros(MAX_NUMBER_OF_BAD_INTERVALS)
    delay_in_samples_in_bad_interval = np.zeros(MAX_NUMBER_OF_BAD_INTERVALS)
    # number_of_bad_intervals = 0
    there_is_a_bad_frame = False

    Whanning = signal.windows.hann(Nf, False)

    D_POW_F = 2
    D_POW_S = 6
    D_POW_T = 2
    A_POW_F = 1
    A_POW_S = 6
    A_POW_T = 2
    D_WEIGHT = 0.1
    A_WEIGHT = 0.0309

    CRITERIUM_FOR_SILENCE_OF_5_SAMPLES = 500
    samples_to_skip_at_start = 0
    sum_of_5_samples = 0
    ref_list = ref_data.tolist()
    first_while_const = SEARCHBUFFER * Downsample
    while (sum_of_5_samples < CRITERIUM_FOR_SILENCE_OF_5_SAMPLES) and (samples_to_skip_at_start < (maxNsamples / 2)):
        left = samples_to_skip_at_start + first_while_const
        right = left + 5
        sum_of_5_samples = sum(map(abs, ref_list[left: right]))
        if sum_of_5_samples < CRITERIUM_FOR_SILENCE_OF_5_SAMPLES:
            samples_to_skip_at_start = samples_to_skip_at_start + 1

    samples_to_skip_at_end = 0
    sum_of_5_samples = 0
    second_while_const = maxNsamples - SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000)
    while (sum_of_5_samples < CRITERIUM_FOR_SILENCE_OF_5_SAMPLES) and (samples_to_skip_at_end < (maxNsamples / 2)):
        right = second_while_const - samples_to_skip_at_end
        left = right - 5
        sum_of_5_samples = sum(map(abs, ref_list[left: right]))
        if sum_of_5_samples < CRITERIUM_FOR_SILENCE_OF_5_SAMPLES:
            samples_to_skip_at_end = samples_to_skip_at_end + 1

    start_frame = math.floor(samples_to_skip_at_start / (Nf / 2))
    stop_frame = math.floor((maxNsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS * (Fs / 1000) -
                             samples_to_skip_at_end) / (Nf / 2)) - 1
    # number of frames in speech Datapipeline plus DATAPADDING_MSECS

    D_disturbance = np.zeros((stop_frame + 1, Nb))
    DA_disturbance = np.zeros((stop_frame + 1, Nb))

    # power_ref = pow_of(ref_data, SEARCHBUFFER * Downsample,
    #                    maxNsamples - SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000),
    #                    maxNsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000))
    # power_deg = pow_of(deg_data, SEARCHBUFFER * Downsample,
    #                    maxNsamples - SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000),
    #                    maxNsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000))

    # hz_spectrum_ref = np.zeros(int(Nf / 2))
    hz_spectrum_deg = np.zeros(int(Nf / 2))
    # frame_is_bad = np.zeros(stop_frame + 1)
    smeared_frame_is_bad = np.zeros(stop_frame + 1)
    silent = np.zeros(stop_frame + 1)

    pitch_pow_dens_ref = np.zeros((stop_frame + 1, Nb))
    pitch_pow_dens_deg = np.zeros((stop_frame + 1, Nb))

    frame_was_skipped = np.zeros(stop_frame + 1)
    frame_disturbance = np.zeros(stop_frame + 1)
    frame_disturbance_asym_add = np.zeros(stop_frame + 1)

    # avg_pitch_pow_dens_ref = np.zeros(Nb)
    # avg_pitch_pow_dens_deg = np.zeros(Nb)
    # loudness_dens_ref = np.zeros(Nb)
    # loudness_dens_deg = np.zeros(Nb)
    # deadzone = np.zeros(Nb)
    # disturbance_dens = np.zeros(Nb)
    # disturbance_dens_asym_add = np.zeros(Nb)

    time_weight = np.ones(stop_frame + 1)
    total_power_ref = np.zeros(stop_frame + 1)

    for frame in range(stop_frame + 1):
        start_sample_ref = 1 + SEARCHBUFFER * Downsample + frame * int(Nf / 2)
        hz_spectrum_ref = short_term_fft(Nf, ref_data, Whanning, start_sample_ref)

        utt = Nutterances
        while (utt >= 1) and ((Utt_Start[utt - 1] - 1) * Downsample + 1 > start_sample_ref):
            utt = utt - 1
        # cond = np.less_equal((Utt_Start[: Nutterances] - 1) * Downsample + 1, start_sample_ref)
        # utt = np.max(np.where(cond == 1))

        if utt >= 1:
            delay = Utt_Delay[utt - 1]
        else:
            delay = Utt_Delay[0]

        start_sample_deg = start_sample_ref + delay
        if (start_sample_deg > 0) and (start_sample_deg + Nf - 1 < maxNsamples + DATAPADDING_MSECS * (Fs / 1000)):
            hz_spectrum_deg = short_term_fft(Nf, deg_data, Whanning, start_sample_deg)
        else:
            hz_spectrum_deg[0: int(Nf / 2)] = 0

        pitch_pow_dens_ref[frame, :] = freq_warping(hz_spectrum_ref, Nb, frame)
        pitch_pow_dens_deg[frame, :] = freq_warping(hz_spectrum_deg, Nb, frame)

        total_audible_pow_ref = total_audible(frame, pitch_pow_dens_ref, 1E2)
        # total_audible_pow_deg = total_audible(frame, pitch_pow_dens_deg, 1E2)
        silent[frame] = (total_audible_pow_ref < 1E7)

        if frame == Plot_Frame:
            fig = plt.figure()
            freq_resolution = Fs / Nf
            axis_freq = np.arange(Nf / 2) * freq_resolution
            plt.subplot(121)
            plt.plot(axis_freq, 10 * math.log10(hz_spectrum_ref + np.spacing(1)))
            plt.axis([0, Fs / 2, -10, 120])
            plt.title('reference signal power spectrum')
            plt.subplot(122)
            plt.plot(axis_freq, 10 * math.log10(hz_spectrum_deg + np.spacing(1)))
            plt.axis([0, Fs / 2, -10, 120])
            plt.title('degraded signal power spectrum')

            fig = plt.figure()
            plt.subplot(121)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + pitch_pow_dens_ref[frame, :]))
            plt.axis([0, Fs / 2, 0, 95])
            plt.title('reference signal bark spectrum')
            plt.subplot(122)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + pitch_pow_dens_deg[frame, :]))
            plt.axis([0, Fs / 2, 0, 95])
            plt.title('degraded signal bark spectrum')

    avg_pitch_pow_dens_ref = time_avg_audible_of(stop_frame + 1, silent, pitch_pow_dens_ref,
                                                 math.floor((maxNsamples - 2 * SEARCHBUFFER * Downsample +
                                                             DATAPADDING_MSECS * (Fs / 1000)) / (Nf / 2)) - 1)

    avg_pitch_pow_dens_deg = time_avg_audible_of(stop_frame + 1, silent, pitch_pow_dens_deg,
                                                 math.floor((maxNsamples - 2 * SEARCHBUFFER * Downsample +
                                                            DATAPADDING_MSECS * (Fs / 1000)) / (Nf / 2)) - 1)

    if CALIBRATE == 0:
        pitch_pow_dens_ref = freq_resp_compensation(stop_frame + 1, pitch_pow_dens_ref, avg_pitch_pow_dens_ref,
                                                    avg_pitch_pow_dens_deg, 1000)
        if Plot_Frame >= 0:
            fig = plt.figure()
            plt.subplot(121)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + pitch_pow_dens_ref[Plot_Frame, :]))
            plt.axis([0, Fs / 2, 0, 95])
            plt.title('reference signal bark spectrum with frequency compensation')
            plt.subplot(122)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + pitch_pow_dens_deg[Plot_Frame, :]))
            plt.axis([0, Fs / 2, 0, 95])
            plt.title('degraded signal bark spectrum')

    MAX_SCALE = 5.0
    MIN_SCALE = 3e-4
    oldScale = 1
    THRESHOLD_BAD_FRAMES = 30

    for frame in range(stop_frame + 1):
        total_audible_pow_ref = total_audible(frame, pitch_pow_dens_ref, 1)
        total_audible_pow_deg = total_audible(frame, pitch_pow_dens_deg, 1)
        total_power_ref[frame] = total_audible_pow_ref

        scale = (total_audible_pow_ref + 5e3) / (total_audible_pow_deg + 5e3)
        if frame > 0:
            scale = 0.2 * oldScale + 0.8 * scale
        oldScale = scale

        if scale > MAX_SCALE:
            scale = MAX_SCALE
        elif scale < MIN_SCALE:
            scale = MIN_SCALE

        pitch_pow_dens_deg[frame, :] = np.multiply(pitch_pow_dens_deg[frame, :], scale)

        if frame == Plot_Frame:
            fig = plt.figure()
            plt.subplot(121)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + pitch_pow_dens_ref[Plot_Frame, :]))
            plt.axis([0, Fs / 2, 0, 95])
            plt.subplot(122)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + pitch_pow_dens_deg[Plot_Frame, :]))
            plt.axis([0, Fs / 2, 0, 95])

        loudness_dens_ref = intensity_warping_of(frame, pitch_pow_dens_ref)
        loudness_dens_deg = intensity_warping_of(frame, pitch_pow_dens_deg)
        disturbance_dens = loudness_dens_deg - loudness_dens_ref

        if frame == Plot_Frame:
            fig = plt.figure()
            plt.subplot(121)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + loudness_dens_ref))
            plt.axis([0, Fs / 2, 0, 15])
            plt.title('reference signal loudness density')
            plt.subplot(122)
            plt.plot(centre_of_band_hz, 10 * math.log10(np.spacing(1) + loudness_dens_deg))
            plt.axis([0, Fs / 2, 0, 15])
            plt.title('degraded signal loudness density')

        deadzone = np.multiply(0.25, np.minimum(loudness_dens_deg, loudness_dens_ref))

        cond1 = np.greater(disturbance_dens, deadzone)
        cond2 = np.less(disturbance_dens, np.negative(deadzone))
        cond3 = np.less_equal(disturbance_dens, np.abs(deadzone))
        d1 = np.where(cond3, 0, disturbance_dens)
        d2 = np.where(cond1, np.subtract(disturbance_dens, deadzone), d1)
        disturbance_dens = np.where(cond2, np.add(disturbance_dens, deadzone), d2)

        if frame == Plot_Frame:
            fig = plt.figure()
            plt.subplot(121)
            plt.plot(centre_of_band_hz, disturbance_dens)
            plt.axis([0, Fs / 2, -1, 50])
            plt.title('disturbance')
        D_disturbance[frame, :] = disturbance_dens

        frame_disturbance[frame] = pseudo_Lp(disturbance_dens, D_POW_F)

        if frame_disturbance[frame] > THRESHOLD_BAD_FRAMES:
            there_is_a_bad_frame = True

        disturbance_dens = multiply_with_asymmetry_factor(disturbance_dens, frame,
                                                          pitch_pow_dens_ref, pitch_pow_dens_deg)

        if frame == Plot_Frame:
            plt.subplot(122)
            plt.plot(centre_of_band_hz, disturbance_dens)
            plt.axis([0, Fs / 2, -1, 50])
            plt.title('disturbance after asymmetry processing')
        DA_disturbance[frame, :] = disturbance_dens

        frame_disturbance_asym_add[frame] = pseudo_Lp(disturbance_dens, A_POW_F)

    frame_was_skipped[0: 1 + stop_frame] = False

    for utt in range(2, Nutterances + 1):
        frame1 = math.floor(((Utt_Start[utt - 1] - 1 - SEARCHBUFFER) * Downsample + 1 + Utt_Delay[utt - 1]) / (Nf / 2))
        j = math.floor(math.floor(((Utt_End[utt - 2] - 1 - SEARCHBUFFER) *
                                   Downsample + 1 + Utt_Delay[utt - 2])) / (Nf / 2))
        delay_jump = Utt_Delay[utt - 1] - Utt_Delay[utt - 2]

        frame1 = min(frame1, j)
        frame1 = max(frame1, 0)

        if delay_jump < -(Nf / 2):
            frame2 = math.floor(((Utt_Start[utt - 1] - 1 - SEARCHBUFFER) * Downsample + 1
                                 + max(0, abs(delay_jump))) / (Nf / 2)) + 1
            end_frame = min(stop_frame, frame2 + 1)
            frame_index = np.arange(frame1, end_frame)
            frame_was_skipped[frame_index] = True
            frame_disturbance[frame_index] = 0
            frame_disturbance_asym_add[frame_index] = 0

    nn = DATAPADDING_MSECS * int(Fs / 1000) + maxNsamples
    tweaked_deg = np.zeros(nn)
    # this method using list slice takes 0.02 seconds, for comparison, matlab takes 0.006 seconds
    # start_utt = (Utt_Start[0: Nutterances]).tolist()
    # final_utt = np.empty((nn - SEARCHBUFFER * Downsample * 2), dtype=int)
    # for i in range(SEARCHBUFFER * Downsample + 1, nn - SEARCHBUFFER * Downsample + 1):
    #     utt = Nutterances
    #     while (utt >= 1) and ((start_utt[utt - 1] - 1) * Downsample > i):
    #         utt = utt - 1
    #     final_utt[i - SEARCHBUFFER * Downsample - 1] = utt
    # this method using numpy takes 0.002 seconds, but can be problematic if satis_num not continuous
    start_utt = Utt_Start[0: Nutterances]
    shape = nn - 2 * SEARCHBUFFER * Downsample
    i = np.arange(SEARCHBUFFER * Downsample + 1, nn - SEARCHBUFFER * Downsample + 1).reshape((1, shape))
    to_compare = np.tile(np.expand_dims(np.multiply(np.subtract(start_utt, 1), Downsample), 1), [1, shape])
    satis_num = np.count_nonzero(np.greater(to_compare, i), axis=0)
    final_utt = np.subtract(Nutterances, satis_num)
    utt_index = np.where(np.greater_equal(final_utt, 1), np.subtract(final_utt, 1), 0)
    delay = Utt_Delay[utt_index]
    j = np.add(i, delay)
    j = np.maximum(SEARCHBUFFER * Downsample + 1, j)
    j = np.minimum(nn - SEARCHBUFFER * Downsample, j)
    tweaked_deg[i - 1] = deg_data[j - 1]

    if there_is_a_bad_frame:
        frame_is_bad = np.greater(frame_disturbance, THRESHOLD_BAD_FRAMES)
        smeared_frame_is_bad[:] = False
        frame_is_bad[0] = False
        SMEAR_RANGE = 2

        for frame in range(SMEAR_RANGE, stop_frame - SMEAR_RANGE):
            max_itself_and_left = frame_is_bad[frame]
            max_itself_and_right = frame_is_bad[frame]

            for i in range(-SMEAR_RANGE, 1):
                if max_itself_and_left < frame_is_bad[frame + i]:
                    max_itself_and_left = frame_is_bad[frame + i]
            for i in range(SMEAR_RANGE + 1):
                if max_itself_and_right < frame_is_bad[frame + i]:
                    max_itself_and_right = frame_is_bad[frame + i]

            mini = max_itself_and_left
            if mini > max_itself_and_right:
                mini = max_itself_and_right

            smeared_frame_is_bad[frame] = mini

        MINIMUM_NUMBER_OF_BAD_FRAMES_IN_BAD_INTERVAL = 5
        number_of_bad_intervals = 0
        frame = 0
        while frame <= stop_frame:
            while (frame <= stop_frame) and (not smeared_frame_is_bad[frame]):
                frame = frame + 1

            if frame <= stop_frame:
                start_frame_of_bad_interval[number_of_bad_intervals] = 1 + frame

                while (frame <= stop_frame) and (smeared_frame_is_bad[frame]):
                    frame = frame + 1

                if frame <= stop_frame:
                    stop_frame_of_bad_interval[number_of_bad_intervals] = 1 + frame
                    if stop_frame_of_bad_interval[number_of_bad_intervals] - \
                            start_frame_of_bad_interval[number_of_bad_intervals] \
                            >= MINIMUM_NUMBER_OF_BAD_FRAMES_IN_BAD_INTERVAL:
                        number_of_bad_intervals = number_of_bad_intervals + 1

        bad_interval_index = np.arange(number_of_bad_intervals)
        start_sample_of_bad_interval[bad_interval_index] = \
            (start_frame_of_bad_interval[bad_interval_index] - 1) * (Nf / 2) + SEARCHBUFFER * Downsample + 1
        stop_sample_of_bad_interval[bad_interval_index] = \
            (stop_frame_of_bad_interval[bad_interval_index] - 1) * (Nf / 2) + SEARCHBUFFER * Downsample + Nf
        stop_frame_of_bad_interval[bad_interval_index] = \
            np.minimum(stop_frame_of_bad_interval[bad_interval_index], stop_frame + 1)
        number_of_samples_in_bad_interval[bad_interval_index] = \
            stop_sample_of_bad_interval[bad_interval_index] - start_sample_of_bad_interval[bad_interval_index] + 1

        SEARCH_RANGE_IN_TRANSFORM_LENGTH = 4
        search_range_in_samples = SEARCH_RANGE_IN_TRANSFORM_LENGTH * Nf
        number_of_samples_in_bad_interval = number_of_samples_in_bad_interval.astype(np.int32)
        start_sample_of_bad_interval = start_sample_of_bad_interval.astype(np.int32)
        stop_sample_of_bad_interval = stop_sample_of_bad_interval.astype(np.int32)

        for bad_interval in range(number_of_bad_intervals):
            ref = np.zeros(2 * search_range_in_samples + number_of_samples_in_bad_interval[bad_interval])
            deg = np.zeros(2 * search_range_in_samples + number_of_samples_in_bad_interval[bad_interval])

            ref[0: search_range_in_samples] = 0
            ref[search_range_in_samples: search_range_in_samples + number_of_samples_in_bad_interval[bad_interval]] = \
                ref_data[start_sample_of_bad_interval[bad_interval]:
                         start_sample_of_bad_interval[bad_interval] + number_of_samples_in_bad_interval[bad_interval]]
            ref[search_range_in_samples + number_of_samples_in_bad_interval[bad_interval]:
                search_range_in_samples + number_of_samples_in_bad_interval[bad_interval] + search_range_in_samples] \
                = 0

            i = np.arange(2 * search_range_in_samples + number_of_samples_in_bad_interval[bad_interval])
            j = start_sample_of_bad_interval[bad_interval] - search_range_in_samples + i
            nn = maxNsamples - SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000)
            j = np.maximum(j, SEARCHBUFFER * Downsample + 1)
            j = np.minimum(j, nn)
            deg[i] = tweaked_deg[j - 1]
            # for i in range(2 * search_range_in_samples + number_of_samples_in_bad_interval[bad_interval]):
            #     j = start_sample_of_bad_interval[bad_interval] - search_range_in_samples + i
            #     nn = maxNsamples - SEARCHBUFFER * Downsample + DATAPADDING_MSECS * int(Fs / 1000)
            #     if j <= SEARCHBUFFER * Downsample:
            #         j = SEARCHBUFFER * Downsample + 1
            #     if j > nn:
            #         j = nn
            #     deg[i] = tweaked_deg[j - 1]

            delay_in_samples, best_correlation = compute_delay(
                1, 2 * search_range_in_samples + number_of_samples_in_bad_interval[bad_interval],
                search_range_in_samples, ref, deg)
            delay_in_samples_in_bad_interval[bad_interval] = delay_in_samples

            if best_correlation < 0.5:
                delay_in_samples_in_bad_interval[bad_interval] = 0

        if number_of_bad_intervals > 0:
            doubly_tweaked_deg = tweaked_deg[0: maxNsamples + DATAPADDING_MSECS * int(Fs / 1000)]
            for bad_interval in range(number_of_bad_intervals):
                delay = delay_in_samples_in_bad_interval[bad_interval]
                i = np.arange(start_sample_of_bad_interval[bad_interval], stop_sample_of_bad_interval[bad_interval])
                j = np.add(i, int(delay))
                j = np.maximum(j, 1)
                j = np.minimum(j, maxNsamples)
                doubly_tweaked_deg[i - 1] = tweaked_deg[j - 1]

            untweaked_deg = deg_data
            deg_data = doubly_tweaked_deg

            for bad_interval in range(number_of_bad_intervals):
                for frame in range(int(start_frame_of_bad_interval[bad_interval]),
                                   int(stop_frame_of_bad_interval[bad_interval])):
                    frame = frame - 1
                    start_sample_ref = SEARCHBUFFER * Downsample + frame * int(Nf / 2) + 1
                    start_sample_deg = start_sample_ref
                    hz_spectrum_deg = short_term_fft(Nf, deg_data, Whanning, start_sample_deg)
                    pitch_pow_dens_deg[frame, :] = freq_warping(hz_spectrum_deg, Nb, frame)

                oldScale = 1
                for frame in range(int(start_frame_of_bad_interval[bad_interval]),
                                   int(stop_frame_of_bad_interval[bad_interval])):
                    frame = frame - 1
                    total_audible_pow_ref = total_audible(frame, pitch_pow_dens_ref, 1)
                    total_audible_pow_deg = total_audible(frame, pitch_pow_dens_deg, 1)
                    scale = (total_audible_pow_ref + 5e3) / (total_audible_pow_deg + 5e3)
                    if frame > 0:
                        scale = 0.2 * oldScale + 0.8 * scale
                    oldScale = scale
                    if scale > MAX_SCALE:
                        scale = MAX_SCALE
                    if scale < MIN_SCALE:
                        scale = MIN_SCALE

                    pitch_pow_dens_deg[frame, :] = pitch_pow_dens_deg[frame, :] * scale
                    loudness_dens_ref = intensity_warping_of(frame, pitch_pow_dens_ref)
                    loudness_dens_deg = intensity_warping_of(frame, pitch_pow_dens_deg)
                    disturbance_dens = loudness_dens_deg - loudness_dens_ref

                    deadzone = np.minimum(loudness_dens_deg, loudness_dens_ref)
                    deadzone = np.multiply(deadzone, 0.25)

                    cond1 = np.greater(disturbance_dens, deadzone)
                    cond2 = np.less(disturbance_dens, np.negative(deadzone))
                    cond3 = np.less_equal(disturbance_dens, np.abs(deadzone))
                    d1 = np.where(cond3, 0, disturbance_dens)
                    d2 = np.where(cond1, np.subtract(disturbance_dens, deadzone), d1)
                    disturbance_dens = np.where(cond2, np.add(disturbance_dens, deadzone), d2)

                    frame_disturbance[frame] = min(frame_disturbance[frame], pseudo_Lp(disturbance_dens, D_POW_F))
                    disturbance_dens = multiply_with_asymmetry_factor(disturbance_dens, frame,
                                                                      pitch_pow_dens_ref, pitch_pow_dens_deg)
                    frame_disturbance_asym_add[frame] = \
                        min(frame_disturbance_asym_add[frame], pseudo_Lp(disturbance_dens, A_POW_F))

            deg_data = untweaked_deg

    if stop_frame + 1 > 1000:
        frame_index = np.arange(stop_frame + 1)
        n = math.floor((maxNsamples - 2 * SEARCHBUFFER * Downsample) / (Nf / 2)) - 1
        timeWeightFactor = (n - 1000) / 5500
        timeWeightFactor = min(timeWeightFactor, 0.5)
        time_weight = (1.0 - timeWeightFactor) + timeWeightFactor * frame_index / n

    h = np.power((total_power_ref + 1e5) / 1e7, 0.04)
    frame_disturbance = np.divide(frame_disturbance, h)
    frame_disturbance_asym_add = np.divide(frame_disturbance_asym_add, h)
    frame_disturbance = np.minimum(frame_disturbance, 45)
    frame_disturbance_asym_add = np.minimum(frame_disturbance_asym_add, 45)

    d_indicator = Lpq_weight(start_frame, stop_frame, D_POW_S, D_POW_T, frame_disturbance, time_weight)
    a_indicator = Lpq_weight(start_frame, stop_frame, A_POW_S, A_POW_T, frame_disturbance_asym_add, time_weight)

    pesq_mos = 4.5 - D_WEIGHT * d_indicator - A_WEIGHT * a_indicator

    if Plot_Frame > 0:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax = Axes3D(ax)
        X1, Y1, Z1 = np.meshgrid(np.arange(stop_frame + 1), centre_of_band_hz, D_disturbance)
        ax.plot_surface(X1, Y1, Z1)
        ax.title('disturbance')
        bx = fig.add_subplot(122)
        bx = Axes3D(bx)
        X2, Y2, Z2 = np.meshgrid(np.arange(stop_frame + 1), centre_of_band_hz, DA_disturbance)
        bx.plot_surface(X2, Y2, Z2)
        bx.title('disturbance after asymmetry processing')
        plt.show()

    return pesq_mos


def Lpq_weight(start_frame, stop_frame, power_syllable, power_time, frame_disturbance, time_weight):
    global NUMBER_OF_PSQM_FRAMES_PER_SYLLABE

    result_time = 0
    total_time_weight_time = 0
    for start_frame_of_syllable in range(start_frame, stop_frame + 1, int(NUMBER_OF_PSQM_FRAMES_PER_SYLLABE / 2)):
        result_syllable = 0
        count_syllable = 0
        for frame in range(start_frame_of_syllable, start_frame_of_syllable + NUMBER_OF_PSQM_FRAMES_PER_SYLLABE):
            if frame <= stop_frame:
                h = frame_disturbance[frame]
                result_syllable = result_syllable + pow(h, power_syllable)
            count_syllable = count_syllable + 1

        result_syllable = result_syllable / count_syllable
        result_syllable = pow(result_syllable, (1 / power_syllable))

        result_time += pow(time_weight[start_frame_of_syllable - start_frame] * result_syllable, power_time)
        total_time_weight_time += pow(time_weight[start_frame_of_syllable - start_frame], power_time)

    result_time = result_time / total_time_weight_time
    result_time = pow(result_time, 1 / power_time)

    return result_time


def compute_delay(start_sample, stop_sample, search_range, time_series1, time_series2):
    n = stop_sample - start_sample + 1
    power_of_2 = pow(2, math.ceil(math.log2(2 * n)))

    power1 = pow_of(time_series1, start_sample, stop_sample, n) * n / power_of_2
    power2 = pow_of(time_series2, start_sample, stop_sample, n) * n / power_of_2
    normalization = math.sqrt(power1 * power2)

    if (power1 <= 1e-6) or (power2 <= 1e-6):
        max_correlation = 0
        best_delay = 0

    x1 = np.zeros(power_of_2)
    x2 = np.zeros(power_of_2)
    y = np.zeros(power_of_2)
    x1[0: n] = np.fabs(time_series1[start_sample - 1: stop_sample])
    x2[0: n] = np.fabs(time_series2[start_sample - 1: stop_sample])

    x1_fft = fft(x1, power_of_2) / power_of_2
    x2_fft = fft(x2, power_of_2)
    x1_fft_conj = np.conj(x1_fft)
    y = ifft(np.multiply(x1_fft_conj, x2_fft), power_of_2)

    best_delay = 0
    max_correlation = 0
    for i in range(-search_range, 0):
        h = abs(y[i + power_of_2]) / normalization
        if h > max_correlation:
            max_correlation = h
            best_delay = i
    for i in range(search_range):
        h = abs(y[i]) / normalization
        if h > max_correlation:
            max_correlation = h
            best_delay = i
    best_delay = best_delay - 1
    return best_delay, max_correlation


def multiply_with_asymmetry_factor(disturbance_dens, frame, pitch_pow_dens_ref, pitch_pow_dens_deg):
    global Nb

    ratio = np.divide((pitch_pow_dens_deg[frame, :] + 50), (pitch_pow_dens_ref[frame, :] + 50))
    h = np.power(ratio, 1.2)
    h = np.minimum(h, 12)
    h = np.where((h < 3), 0, h)
    mod_disturbance_dens = np.multiply(disturbance_dens, h)

    return mod_disturbance_dens


def intensity_warping_of(frame, pitch_pow_dens):
    global abs_thresh_power, Sl, Nb, centre_of_band_bark
    ZWICKER_POWER = 0.23

    input = pitch_pow_dens[frame, :]
    h = np.where(np.less(centre_of_band_bark, 4), 6 / (centre_of_band_bark + 2), 1)
    h = np.minimum(h, 2)
    h = np.power(h, 0.15)
    modified_zwicker_power = np.multiply(ZWICKER_POWER, h)
    cond = np.greater(input, abs_thresh_power)
    loudness_dens_new = np.multiply(np.power(abs_thresh_power / 0.5, modified_zwicker_power),
                                    (np.power(0.5 + 0.5 * input / abs_thresh_power, modified_zwicker_power) - 1))
    loudness_dens = np.where(cond, loudness_dens_new, 0)
    loudness_dens = np.multiply(loudness_dens, Sl)

    return loudness_dens


def pseudo_Lp(x, p):
    global Nb, width_of_band_bark

    h = np.abs(x[1: Nb])
    w = width_of_band_bark[1: Nb]
    prod = np.multiply(h, w)
    result = np.sum(np.power(prod, p))
    totalWeight = np.sum(w)
    result = np.power(result / totalWeight, 1 / p)
    result = np.multiply(result, totalWeight)
    return result


def freq_resp_compensation(number_of_frames, pitch_pow_dens_ref, avg_pitch_pow_dens_ref,
                           avg_pitch_pow_dens_deg, constant):
    global Nb

    x = (avg_pitch_pow_dens_deg + constant) / (avg_pitch_pow_dens_ref + constant)
    x = np.minimum(x, 100.0)
    x = np.maximum(x, 0.01)
    mod_pitch_pow_dens_ref = np.multiply(pitch_pow_dens_ref, x)

    return mod_pitch_pow_dens_ref


def time_avg_audible_of(number_of_frames, silent, pitch_pow_dens, total_number_of_frames):
    global Nb, abs_thresh_power

    find_silent = np.squeeze(np.argwhere(silent == 0))
    h = pitch_pow_dens[find_silent]
    to_compare = np.multiply(100, abs_thresh_power)
    result = np.where(np.greater(h, to_compare), h, 0)
    avg_pitch_pow_dens = np.divide(np.sum(result, axis=0), total_number_of_frames)

    return avg_pitch_pow_dens


def short_term_fft(Nf, data, Whanning, start_sample):
    x1 = np.multiply(data[start_sample - 1: start_sample + Nf - 1], Whanning)
    x1_fft = fft(x1)
    hz_spectrum = np.square(np.abs(x1_fft[0: int(Nf / 2)]))
    hz_spectrum[0] = 0
    return hz_spectrum


def freq_warping(hz_spectrum, Nb, frame):
    global nr_of_hz_bands_per_bark_band, pow_dens_correction_factor
    global Sp

    hz_band = 0
    sum_total = np.empty(Nb)
    for bark_band in range(Nb):
        n = nr_of_hz_bands_per_bark_band[bark_band]
        sum = 0
        for i in range(n):
            sum = sum + hz_spectrum[hz_band]
            hz_band = hz_band + 1
        sum_total[bark_band] = sum
    pitch_pow_dens = sum_total * pow_dens_correction_factor * Sp
    return pitch_pow_dens


def total_audible(frame, pitch_pow_dens, factor):
    global Nb, abs_thresh_power

    h = pitch_pow_dens[frame, 1: Nb]
    threshold = np.multiply(factor, abs_thresh_power[1: Nb])
    total_audible_pow = np.where(np.greater(h, threshold), h, 0)
    total_audible_pow = np.sum(total_audible_pow)

    return total_audible_pow


def pow_of(data, start_point, end_point, divisor):
    x = data[start_point - 1: end_point]
    power = np.dot(x, x) / divisor
    return power


def setup_global(sampling_rate):
    global Downsample, InIIR_Hsos, InIIR_Nsos, Align_Nfft
    global DATAPADDING_MSECS, SEARCHBUFFER, Fs, MINSPEECHLGTH, JOINSPEECHLGTH
    global Nutterances, Largest_uttsize, Nsurf_samples, Crude_DelayEst
    global Crude_DelayConf, UttSearch_Start, UttSearch_End, Utt_DelayEst
    global Utt_Delay, Utt_DelayConf, Utt_Start, Utt_End
    global MAXNUTTERANCES, WHOLE_SIGNAL
    global pesq_mos, subj_mos, cond_nr, MINUTTLENGTH
    global CALIBRATE, Nfmax, Nb, Sl, Sp
    global nr_of_hz_bands_per_bark_band, centre_of_band_bark
    global width_of_band_hz, centre_of_band_hz, width_of_band_bark
    global pow_dens_correction_factor, abs_thresh_power

    CALIBRATE = 0
    Nfmax = 512

    MAXNUTTERANCES = 50
    MINUTTLENGTH = 50
    WHOLE_SIGNAL = -1
    UttSearch_Start = np.zeros(MAXNUTTERANCES, dtype=int)
    UttSearch_End = np.zeros(MAXNUTTERANCES, dtype=int)
    Utt_DelayEst = np.zeros(MAXNUTTERANCES, dtype=int)
    Utt_DelayEst = np.zeros(MAXNUTTERANCES, dtype=int)
    Utt_Delay = np.zeros(MAXNUTTERANCES, dtype=int)
    Utt_DelayConf = np.zeros(MAXNUTTERANCES)
    Utt_Start = np.zeros(MAXNUTTERANCES, dtype=int)
    Utt_End = np.zeros(MAXNUTTERANCES, dtype=int)

    DATAPADDING_MSECS = 320
    SEARCHBUFFER = 75
    MINSPEECHLGTH = 4
    JOINSPEECHLGTH = 50

    global WB_InIIR_Nsos, WB_InIIR_Hsos
    if sampling_rate == 8000:
        WB_InIIR_Nsos = 1
        WB_InIIR_Hsos = np.array([[2.6657628, -5.3315255, 2.6657628, -1.8890331, 0.89487434]])
    elif sampling_rate == 16000:
        WB_InIIR_Nsos = 1
        WB_InIIR_Hsos = np.array([[2.740826, -5.4816519, 2.740826, -1.9444777, 0.94597794]])
    else:
        print('Error: Unsupported sampling rate.')

    Sp_16k = 6.910853e-006
    Sl_16k = 1.866055e-001
    fs_16k = 16000
    Downsample_16k = 64
    Align_Nfft_16k = 1024
    InIIR_Nsos_16k = 12
    InIIR_Hsos_16k = np.array([[0.325631521, -0.086782860, -0.238848661, -1.079416490, 0.434583902],
                               [0.403961804, -0.556985881, 0.153024077, -0.415115835, 0.696590244],
                               [4.736162769, 3.287251046, 1.753289019, -1.859599046, 0.876284034],
                               [0.365373469, 0.000000000, 0.000000000, -0.634626531, 0.000000000],
                               [0.884811506, 0.000000000, 0.000000000, -0.256725271, 0.141536777],
                               [0.723593055, -1.447186099, 0.723593044, -1.129587469, 0.657232737],
                               [1.644910855, -1.817280902, 1.249658063, -1.778403899, 0.801724355],
                               [0.633692689, -0.284644314, -0.319789663, 0.000000000, 0.000000000],
                               [1.032763031, 0.268428979, 0.602913323, 0.000000000, 0.000000000],
                               [1.001616361, -0.823749013, 0.439731942, -0.885778255, 0.000000000],
                               [0.752472096, -0.375388990, 0.188977609, -0.077258216, 0.247230734],
                               [1.023700575, 0.001661628, 0.521284240, -0.183867259, 0.354324187]])

    Sp_8k = 2.764344e-5
    Sl_8k = 1.866055e-1
    fs_8k = 8000
    Downsample_8k = 32
    Align_Nfft_8k = 512
    InIIR_Nsos_8k = 8
    InIIR_Hsos_8k = np.array([[0.885535424, -0.885535424, 0.000000000, -0.771070709, 0.000000000],
                              [0.895092588, 1.292907193, 0.449260174, 1.268869037, 0.442025372],
                              [4.049527940, -7.865190042, 3.815662102, -1.746859852, 0.786305963],
                              [0.500002353, -0.500002353, 0.000000000, 0.000000000, 0.000000000],
                              [0.565002834, -0.241585934, -0.306009671, 0.259688659, 0.249979657],
                              [2.115237288, 0.919935084, 1.141240051, -1.587313419, 0.665935315],
                              [0.912224584, -0.224397719, -0.641121413, -0.246029464, -0.556720590],
                              [0.444617727, -0.307589321, 0.141638062, -0.996391149, 0.502251622]])

    nr_of_hz_bands_per_bark_band_8k = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
                                                1, 1, 1, 1, 2, 1, 1, 2, 2, 2,
                                                2, 2, 2, 2, 2, 3, 3, 3, 3, 4,
                                                3, 4, 5, 4, 5, 6, 6, 7, 8, 9,
                                                9, 11])

    centre_of_band_bark_8k = np.array([0.078672, 0.316341, 0.636559, 0.961246, 1.290450,
                                       1.624217, 1.962597, 2.305636, 2.653383, 3.005889,
                                       3.363201, 3.725371, 4.092449, 4.464486, 4.841533,
                                       5.223642, 5.610866, 6.003256, 6.400869, 6.803755,
                                       7.211971, 7.625571, 8.044611, 8.469146, 8.899232,
                                       9.334927, 9.776288, 10.223374, 10.676242, 11.134952,
                                       11.599563, 12.070135, 12.546731, 13.029408, 13.518232,
                                       14.013264, 14.514566, 15.022202, 15.536238, 16.056736,
                                       16.583761, 17.117382])

    centre_of_band_hz_8k = np.array([7.867213, 31.634144, 63.655895, 96.124611, 129.044968,
                                     162.421738, 196.259659, 230.563568, 265.338348, 300.588867,
                                     336.320129, 372.537140, 409.244934, 446.448578, 484.568604,
                                     526.600586, 570.303833, 619.423340, 672.121643, 728.525696,
                                     785.675964, 846.835693, 909.691650, 977.063293, 1049.861694,
                                     1129.635986, 1217.257568, 1312.109497, 1412.501465, 1517.999390,
                                     1628.894165, 1746.194336, 1871.568848, 2008.776123, 2158.979248,
                                     2326.743164, 2513.787109, 2722.488770, 2952.586670, 3205.835449,
                                     3492.679932, 3820.219238])

    width_of_band_bark_8k = np.array([0.157344, 0.317994, 0.322441, 0.326934, 0.331474,
                                      0.336061, 0.340697, 0.345381, 0.350114, 0.354897,
                                      0.359729, 0.364611, 0.369544, 0.374529, 0.379565,
                                      0.384653, 0.389794, 0.394989, 0.400236, 0.405538,
                                      0.410894, 0.416306, 0.421773, 0.427297, 0.432877,
                                      0.438514, 0.444209, 0.449962, 0.455774, 0.461645,
                                      0.467577, 0.473569, 0.479621, 0.485736, 0.491912,
                                      0.498151, 0.504454, 0.510819, 0.517250, 0.523745,
                                      0.530308, 0.536934])

    width_of_band_hz_8k = np.array([15.734426, 31.799433, 32.244064, 32.693359, 33.147385,
                                    33.606140, 34.069702, 34.538116, 35.011429, 35.489655,
                                    35.972870, 36.461121, 36.954407, 37.452911, 40.269653,
                                    42.311859, 45.992554, 51.348511, 55.040527, 56.775208,
                                    58.699402, 62.445862, 64.820923, 69.195374, 76.745667,
                                    84.016235, 90.825684, 97.931152, 103.348877, 107.801880,
                                    113.552246, 121.490601, 130.420410, 143.431763, 158.486816,
                                    176.872803, 198.314697, 219.549561, 240.600098, 268.702393,
                                    306.060059, 349.937012])

    pow_dens_correction_factor_8k = np.array([100.000000, 99.999992, 100.000000, 100.000008, 100.000008,
                                              100.000015, 99.999992, 99.999969, 50.000027, 100.000000,
                                              99.999969, 100.000015, 99.999947, 100.000061, 53.047077,
                                              110.000046, 117.991989, 65.000000, 68.760147, 69.999931,
                                              71.428818, 75.000038, 76.843384, 80.968781, 88.646126,
                                              63.864388, 68.155350, 72.547775, 75.584831, 58.379192,
                                              80.950836, 64.135651, 54.384785, 73.821884, 64.437073,
                                              59.176456, 65.521278, 61.399822, 58.144047, 57.004543,
                                              64.126297, 59.248363])

    abs_thresh_power_8k = np.array([51286152, 2454709.500, 70794.593750,
                                    4897.788574, 1174.897705, 389.045166,
                                    104.712860, 45.708820, 17.782795,
                                    9.772372, 4.897789, 3.090296,
                                    1.905461, 1.258925, 0.977237,
                                    0.724436, 0.562341, 0.457088,
                                    0.389045, 0.331131, 0.295121,
                                    0.269153, 0.257040, 0.251189,
                                    0.251189, 0.251189, 0.251189,
                                    0.263027, 0.288403, 0.309030,
                                    0.338844, 0.371535, 0.398107,
                                    0.436516, 0.467735, 0.489779,
                                    0.501187, 0.501187, 0.512861,
                                    0.524807, 0.524807, 0.524807])

    nr_of_hz_bands_per_bark_band_16k = np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 1,
                                                 1, 1, 1, 1, 2, 1, 1, 2, 2, 2,
                                                 2, 2, 2, 2, 2, 3, 3, 3, 3, 4,
                                                 3, 4, 5, 4, 5, 6, 6, 7, 8, 9,
                                                 9, 12, 12, 15, 16, 18, 21, 25, 20])

    centre_of_band_bark_16k = np.array([0.078672, 0.316341, 0.636559, 0.961246, 1.290450,
                                        1.624217, 1.962597, 2.305636, 2.653383, 3.005889,
                                        3.363201, 3.725371, 4.092449, 4.464486, 4.841533,
                                        5.223642, 5.610866, 6.003256, 6.400869, 6.803755,
                                        7.211971, 7.625571, 8.044611, 8.469146, 8.899232,
                                        9.334927, 9.776288, 10.223374, 10.676242, 11.134952,
                                        11.599563, 12.070135, 12.546731, 13.029408, 13.518232,
                                        14.013264, 14.514566, 15.022202, 15.536238, 16.056736,
                                        16.583761, 17.117382, 17.657663, 18.204674, 18.758478,
                                        19.319147, 19.886751, 20.461355, 21.043034])

    centre_of_band_hz_16k = np.array([7.867213, 31.634144, 63.655895, 96.124611, 129.044968,
                                      162.421738, 196.259659, 230.563568, 265.338348, 300.588867,
                                      336.320129, 372.537140, 409.244934, 446.448578, 484.568604,
                                      526.600586, 570.303833, 619.423340, 672.121643, 728.525696,
                                      785.675964, 846.835693, 909.691650, 977.063293, 1049.861694,
                                      1129.635986, 1217.257568, 1312.109497, 1412.501465, 1517.999390,
                                      1628.894165, 1746.194336, 1871.568848, 2008.776123, 2158.979248,
                                      2326.743164, 2513.787109, 2722.488770, 2952.586670, 3205.835449,
                                      3492.679932, 3820.219238, 4193.938477, 4619.846191, 5100.437012,
                                      5636.199219, 6234.313477, 6946.734863, 7796.473633])

    width_of_band_bark_16k = np.array([0.157344, 0.317994, 0.322441, 0.326934, 0.331474,
                                       0.336061, 0.340697, 0.345381, 0.350114, 0.354897,
                                       0.359729, 0.364611, 0.369544, 0.374529, 0.379565,
                                       0.384653, 0.389794, 0.394989, 0.400236, 0.405538,
                                       0.410894, 0.416306, 0.421773, 0.427297, 0.432877,
                                       0.438514, 0.444209, 0.449962, 0.455774, 0.461645,
                                       0.467577, 0.473569, 0.479621, 0.485736, 0.491912,
                                       0.498151, 0.504454, 0.510819, 0.517250, 0.523745,
                                       0.530308, 0.536934, 0.543629, 0.550390, 0.557220,
                                       0.564119, 0.571085, 0.578125, 0.585232])

    width_of_band_hz_16k = np.array([15.734426, 31.799433, 32.244064, 32.693359,
                                     33.147385, 33.606140, 34.069702, 34.538116,
                                     35.011429, 35.489655, 35.972870, 36.461121,
                                     36.954407, 37.452911, 40.269653, 42.311859,
                                     45.992554, 51.348511, 55.040527, 56.775208,
                                     58.699402, 62.445862, 64.820923, 69.195374,
                                     76.745667, 84.016235, 90.825684, 97.931152,
                                     103.348877, 107.801880, 113.552246, 121.490601,
                                     130.420410, 143.431763, 158.486816, 176.872803,
                                     198.314697, 219.549561, 240.600098, 268.702393,
                                     306.060059, 349.937012, 398.686279, 454.713867,
                                     506.841797, 564.863770, 637.261230, 794.717285,
                                     931.068359])

    pow_dens_correction_factor_16k = np.array([100.000000, 99.999992, 100.000000, 100.000008,
                                               100.000008, 100.000015, 99.999992, 99.999969,
                                               50.000027, 100.000000, 99.999969, 100.000015,
                                               99.999947, 100.000061, 53.047077, 110.000046,
                                               117.991989, 65.000000, 68.760147, 69.999931,
                                               71.428818, 75.000038, 76.843384, 80.968781,
                                               88.646126, 63.864388, 68.155350, 72.547775,
                                               75.584831, 58.379192, 80.950836, 64.135651,
                                               54.384785, 73.821884, 64.437073, 59.176456,
                                               65.521278, 61.399822, 58.144047, 57.004543,
                                               64.126297, 54.311001, 61.114979, 55.077751,
                                               56.849335, 55.628868, 53.137054, 54.985844,
                                               79.546974])

    abs_thresh_power_16k = np.array([51286152.00, 2454709.500, 70794.593750,
                                     4897.788574, 1174.897705, 389.045166,
                                     104.712860, 45.708820, 17.782795,
                                     9.772372, 4.897789, 3.090296,
                                     1.905461, 1.258925, 0.977237,
                                     0.724436, 0.562341, 0.457088,
                                     0.389045, 0.331131, 0.295121,
                                     0.269153, 0.257040, 0.251189,
                                     0.251189, 0.251189, 0.251189,
                                     0.263027, 0.288403, 0.309030,
                                     0.338844, 0.371535, 0.398107,
                                     0.436516, 0.467735, 0.489779,
                                     0.501187, 0.501187, 0.512861,
                                     0.524807, 0.524807, 0.524807,
                                     0.512861, 0.478630, 0.426580,
                                     0.371535, 0.363078, 0.416869,
                                     0.537032])
    if sampling_rate == fs_16k:
        Downsample = Downsample_16k
        InIIR_Hsos = InIIR_Hsos_16k
        InIIR_Nsos = InIIR_Nsos_16k
        Align_Nfft = Align_Nfft_16k
        Fs = fs_16k

        Nb = 49
        Sl = Sl_16k
        Sp = Sp_16k
        nr_of_hz_bands_per_bark_band = nr_of_hz_bands_per_bark_band_16k
        centre_of_band_bark = centre_of_band_bark_16k
        centre_of_band_hz = centre_of_band_hz_16k
        width_of_band_bark = width_of_band_bark_16k
        width_of_band_hz = width_of_band_hz_16k
        pow_dens_correction_factor = pow_dens_correction_factor_16k
        abs_thresh_power = abs_thresh_power_16k
        return

    if sampling_rate == fs_8k:
        Downsample = Downsample_8k
        InIIR_Hsos = InIIR_Hsos_8k
        InIIR_Nsos = InIIR_Nsos_8k
        Align_Nfft = Align_Nfft_8k
        Fs = fs_8k

        Nb = 42
        Sl = Sl_8k
        Sp = Sp_8k
        nr_of_hz_bands_per_bark_band = nr_of_hz_bands_per_bark_band_8k
        centre_of_band_bark = centre_of_band_bark_8k
        centre_of_band_hz = centre_of_band_hz_8k
        width_of_band_bark = width_of_band_bark_8k
        width_of_band_hz = width_of_band_hz_8k
        pow_dens_correction_factor = pow_dens_correction_factor_8k
        abs_thresh_power = abs_thresh_power_8k
        return


def utterance_locate(ref_data, ref_Nsamples, ref_VAD, ref_logVAD, deg_data, deg_Nsamples, deg_VAD, deg_logVAD):
    global Nutterances, Utt_Delay, Utt_DelayConf, Utt_Start, Utt_End, Utt_DelayEst

    id_searchwindows(ref_VAD, ref_Nsamples, deg_VAD, deg_Nsamples)
    for Utt_id in range(1, Nutterances + 1):
        crude_align(ref_logVAD, ref_Nsamples, deg_logVAD, deg_Nsamples, Utt_id)
        time_align(ref_data, ref_Nsamples, deg_data, deg_Nsamples, Utt_id)

    id_utterances(ref_Nsamples, ref_VAD, deg_Nsamples)
    utterance_split(ref_data, ref_Nsamples, ref_VAD, ref_logVAD, deg_data, deg_Nsamples, deg_VAD, deg_logVAD)


def utterance_split(ref_data, ref_Nsamples, ref_VAD, ref_logVAD, deg_data, deg_Nsamples, deg_VAD, deg_logVAD):
    global Nutterances, MAXNUTTERANCES, Downsample, SEARCHBUFFER
    global Utt_DelayEst, Utt_Delay, Utt_DelayConf, UttSearch_Start
    global Utt_Start, Utt_End, Largest_uttsize, UttSearch_End
    global Best_ED1, Best_D1, Best_DC1, Best_ED2, Best_D2, Best_DC2, Best_BP

    Utt_id = 1
    while (Utt_id <= Nutterances) and (Nutterances <= MAXNUTTERANCES):
        Utt_DelayEst_l = Utt_DelayEst[Utt_id - 1]
        Utt_Delay_l = Utt_Delay[Utt_id - 1]
        Utt_DelayConf_l = Utt_DelayConf[Utt_id - 1]
        Utt_Start_l = Utt_Start[Utt_id - 1]
        Utt_End_l = Utt_End[Utt_id - 1]

        Utt_SpeechStart = Utt_Start_l
        Utt_SpeechStart = max(1, Utt_SpeechStart)
        while (Utt_SpeechStart < Utt_End_l) and (ref_VAD[Utt_SpeechStart - 1] <= 0.0):
            Utt_SpeechStart = Utt_SpeechStart + 1
        Utt_SpeechEnd = Utt_End_l
        while (Utt_SpeechEnd > Utt_Start_l) and (ref_VAD[Utt_SpeechEnd - 1] <= 0):
            Utt_SpeechEnd = Utt_SpeechEnd - 1
        Utt_SpeechEnd = Utt_SpeechEnd + 1
        Utt_Len = Utt_SpeechEnd - Utt_SpeechStart

        if Utt_Len >= 200:
            split_align(ref_data, ref_Nsamples, ref_VAD, ref_logVAD, deg_data, deg_Nsamples, deg_VAD, deg_logVAD,
                        Utt_Start_l, Utt_SpeechStart, Utt_SpeechEnd, Utt_End_l, Utt_DelayEst_l, Utt_DelayConf_l)
            if (Best_DC1 > Utt_DelayConf_l) and (Best_DC2 > Utt_DelayConf_l):
                for step in range(Nutterances, Utt_id, -1):
                    Utt_DelayEst[step] = Utt_DelayEst[step - 1]
                    Utt_Delay[step] = Utt_Delay[step - 1]
                    Utt_DelayConf[step] = Utt_DelayConf[step - 1]
                    Utt_Start[step] = Utt_Start[step - 1]
                    Utt_End[step] = Utt_End[step - 1]
                    UttSearch_Start[step] = Utt_Start[step - 1]
                    UttSearch_End[step] = Utt_End[step - 1]

                Nutterances = Nutterances + 1

                Utt_DelayEst[Utt_id - 1] = Best_ED1
                Utt_Delay[Utt_id - 1] = Best_D1
                Utt_DelayConf[Utt_id - 1] = Best_DC1

                Utt_DelayEst[Utt_id] = Best_ED2
                Utt_Delay[Utt_id] = Best_D2
                Utt_DelayConf[Utt_id] = Best_DC2

                UttSearch_Start[Utt_id] = UttSearch_Start[Utt_id - 1]
                UttSearch_End[Utt_id] = UttSearch_End[Utt_id - 1]
                if Best_D2 < Best_D1:
                    Utt_Start[Utt_id - 1] = Utt_Start_l
                    Utt_End[Utt_id - 1] = Best_BP
                    Utt_Start[Utt_id] = Best_BP
                    Utt_End[Utt_id] = Utt_End_l
                else:
                    Utt_Start[Utt_id - 1] = Utt_Start_l
                    Utt_End[Utt_id - 1] = Best_BP + math.floor((Best_D2 - Best_D1) / (2 * Downsample))
                    Utt_Start[Utt_id] = Best_BP - math.floor((Best_D2 - Best_D1) / (2 * Downsample))
                    Utt_End[Utt_id] = Utt_End_l

                if (Utt_Start[Utt_id - 1] - SEARCHBUFFER - 1) * Downsample + 1 + Best_D1 < 0:
                    Utt_Start[Utt_id - 1] = SEARCHBUFFER + 1 + math.floor((Downsample - 1 - Best_D1) / Downsample)

                if ((Utt_End[Utt_id] - 1) * Downsample + 1 + Best_D2) > (deg_Nsamples - SEARCHBUFFER * Downsample):
                    Utt_End[Utt_id] = math.floor((deg_Nsamples - Best_D2) / Downsample) - SEARCHBUFFER + 1
            else:
                Utt_id = Utt_id + 1
        else:
            Utt_id = Utt_id + 1

    Largest_uttsize = np.max(np.subtract(Utt_End, Utt_Start))


def split_align(ref_data, ref_Nsamples, ref_VAD, ref_logVAD, deg_data, deg_Nsamples, deg_VAD, deg_logVAD,
                Utt_Start_l, Utt_SpeechStart, Utt_SpeechEnd, Utt_End_l, Utt_DelayEst_l, Utt_DelayConf_l):
    global MAXNUTTERANCES, Align_Nfft, Downsample, Window
    global Utt_DelayEst, Utt_Delay, UttSearch_Start, UttSearch_End
    global Best_ED1, Best_D1, Best_DC1, Best_ED2, Best_D2, Best_DC2, Best_BP

    Utt_BPs = np.zeros(41)
    Utt_ED1 = np.zeros(41)
    Utt_ED2 = np.zeros(41)
    Utt_D1 = np.zeros(41)
    Utt_D2 = np.zeros(41)
    Utt_DC1 = np.zeros(41)
    Utt_DC2 = np.zeros(41)

    Utt_Len = Utt_SpeechEnd - Utt_SpeechStart
    Utt_Test = MAXNUTTERANCES
    Best_DC1 = 0.0
    Best_DC2 = 0.0
    kernel = int(Align_Nfft / 64)
    Delta = Align_Nfft / (4 * Downsample)
    Step = math.floor((0.801 * Utt_Len + 40 * Delta - 1) / (40 * Delta))
    Step = Step * Delta

    Pad = max(math.floor(Utt_Len / 10), 75)

    Utt_BPs[0] = Utt_SpeechStart + Pad
    N_BPs = 1

    while True:
        N_BPs = N_BPs + 1
        Utt_BPs[N_BPs - 1] = Utt_BPs[N_BPs - 2] + Step
        if not ((Utt_BPs[N_BPs - 1] <= (Utt_SpeechEnd - Pad)) and (N_BPs <= 40)):
            break

    if N_BPs <= 1:
        return

    for bp in range(1, N_BPs):
        Utt_DelayEst[Utt_Test - 1] = Utt_DelayEst_l
        UttSearch_Start[Utt_Test - 1] = Utt_Start_l
        UttSearch_End[Utt_Test - 1] = Utt_BPs[bp - 1]

        crude_align(ref_logVAD, ref_Nsamples, deg_logVAD, deg_Nsamples, MAXNUTTERANCES)
        Utt_ED1[bp - 1] = Utt_Delay[Utt_Test - 1]

        Utt_DelayEst[Utt_Test - 1] = Utt_DelayEst_l
        UttSearch_Start[Utt_Test - 1] = Utt_BPs[bp - 1]
        UttSearch_End[Utt_Test - 1] = Utt_End_l

        crude_align(ref_logVAD, ref_Nsamples, deg_logVAD, deg_Nsamples, MAXNUTTERANCES)
        Utt_ED2[bp - 1] = Utt_Delay[Utt_Test - 1]

    Utt_DC1[0: N_BPs - 1] = -2.0

    while True:
        bp = 1
        while (bp <= N_BPs - 1) and (Utt_DC1[bp - 1] > -2.0):
            bp = bp + 1
        if bp >= N_BPs:
            break

        estdelay = int(Utt_ED1[bp - 1])
        H = np.zeros(Align_Nfft)
        Hsum = 0.0

        startr = (Utt_Start_l - 1) * Downsample + 1
        startd = startr + estdelay

        if startd < 0:
            startr = -estdelay + 1
            startd = 1

        startr = max(1, startr)
        startd = max(1, startd)

        while ((startd + Align_Nfft) <= 1 + deg_Nsamples) and \
                ((startr + Align_Nfft) <= (1 + (Utt_BPs[bp - 1] - 1) * Downsample)):
            X1 = np.multiply(ref_data[startr - 1: startr + Align_Nfft - 1], Window)
            X2 = np.multiply(deg_data[startd - 1: startd + Align_Nfft - 1], Window)

            X1_fft = fft(X1, Align_Nfft)
            X2_fft = fft(X2, Align_Nfft)
            X1 = ifft(np.multiply(np.conj(X1_fft), X2_fft), Align_Nfft)
            X1 = np.abs(X1)
            v_max = np.max(X1) * 0.99
            n_max = (v_max ** 0.125) / kernel

            cond = np.greater(X1, v_max)
            Hsum = Hsum + n_max * kernel * np.count_nonzero(cond)
            k = np.arange(1 - kernel, kernel)
            count_in = np.argwhere(cond)
            H[np.remainder(count_in + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))
            # for j in count_in:
            #     H[np.remainder(j + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))

            startr = startr + int(Align_Nfft / 4)
            startd = startd + int(Align_Nfft / 4)

        v_max = np.max(H)
        I_max = np.argmax(H)
        if I_max >= (Align_Nfft / 2):
            I_max = I_max - Align_Nfft

        Utt_D1[bp - 1] = estdelay + I_max
        if Hsum > 0.0:
            Utt_DC1[bp - 1] = v_max / Hsum
        else:
            Utt_DC1[bp - 1] = 0.0

        while bp < (N_BPs - 1):
            bp = bp + 1
            if (Utt_ED1[bp - 1] == estdelay) and (Utt_DC1[bp - 1] <= -2.0):
                while ((startd + Align_Nfft) <= 1 + deg_Nsamples) and \
                        ((startr + Align_Nfft) <= ((Utt_BPs[bp - 1] - 1) * Downsample + 1)):
                    X1 = np.multiply(ref_data[startr - 1: startr + Align_Nfft - 1], Window)
                    X2 = np.multiply(deg_data[startd - 1: startd + Align_Nfft - 1], Window)

                    X1_fft = fft(X1, Align_Nfft)
                    X2_fft = fft(X2, Align_Nfft)
                    X1 = ifft(np.multiply(np.conj(X1_fft), X2_fft), Align_Nfft)

                    X1 = np.abs(X1)
                    v_max = 0.99 * np.max(X1)
                    n_max = (v_max ** 0.125) / kernel

                    cond = np.greater(X1, v_max)
                    Hsum = Hsum + n_max * kernel * np.count_nonzero(cond)
                    k = np.arange(1 - kernel, kernel)
                    count_in = np.argwhere(cond)
                    H[np.remainder(count_in + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))
                    # for j in count_in:
                    #     H[np.remainder(j + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))

                    startr = startr + int(Align_Nfft / 4)
                    startd = startd + int(Align_Nfft / 4)

                v_max = np.max(H)
                I_max = np.argmax(H)
                if I_max >= (Align_Nfft / 2):
                    I_max = I_max - Align_Nfft

                Utt_D1[bp - 1] = estdelay + I_max
                if Hsum > 0.0:
                    Utt_DC1[bp - 1] = v_max / Hsum
                else:
                    Utt_DC1[bp - 1] = 0.0

    bp = np.arange(N_BPs - 1)
    Utt_DC2[bp] = np.where(np.greater(Utt_DC1[bp], Utt_DelayConf_l), -2.0, 0.0)

    while True:
        bp = N_BPs - 1
        while (bp >= 1) and (Utt_DC2[bp - 1] > -2.0):
            bp = bp - 1
        if bp < 1:
            break

        estdelay = int(Utt_ED2[bp - 1])
        H = np.zeros(Align_Nfft)
        Hsum = 0.0

        startr = (int(Utt_End_l) - 1) * Downsample + 1 - Align_Nfft
        startd = startr + estdelay

        if (startd + Align_Nfft) > deg_Nsamples + 1:
            startd = deg_Nsamples - Align_Nfft + 1
            startr = startd - estdelay

        while (startd >= 1) and (startr >= (Utt_BPs[bp - 1] - 1) * Downsample + 1):
            X1 = np.multiply(ref_data[startr - 1: startr + Align_Nfft - 1], Window)
            X2 = np.multiply(deg_data[startd - 1: startd + Align_Nfft - 1], Window)

            X1_fft = fft(X1, Align_Nfft)
            X2_fft = fft(X2, Align_Nfft)

            X1 = ifft(np.multiply(np.conj(X1_fft), X2_fft), Align_Nfft)
            X1 = np.abs(X1)

            v_max = np.max(X1) * 0.99
            n_max = (v_max ** 0.125) / kernel

            cond = np.greater(X1, v_max)
            Hsum = Hsum + n_max * kernel * np.count_nonzero(cond)
            k = np.arange(1 - kernel, kernel)
            count_in = np.argwhere(cond)
            H[np.remainder(count_in + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))
            # for j in count_in:
            #     H[np.remainder(j + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))

            startr = startr - int(Align_Nfft / 4)
            startd = startd - int(Align_Nfft / 4)

        v_max = np.max(H)
        I_max = np.argmax(H)
        if I_max >= (Align_Nfft / 2):
            I_max = I_max - Align_Nfft

        Utt_D2[bp - 1] = estdelay + I_max
        if Hsum > 0.0:
            Utt_DC2[bp - 1] = v_max / Hsum
        else:
            Utt_DC2[bp - 1] = 0.0

        while bp > 1:
            bp = bp - 1
            if (Utt_ED2[bp - 1] == estdelay) and (Utt_DC2[bp - 1] <= -2.0):
                while (startd >= 1) and (startr >= (Utt_BPs[bp - 1] - 1) * Downsample + 1):
                    X1 = np.multiply(ref_data[startr - 1: startr + Align_Nfft - 1], Window)
                    X2 = np.multiply(deg_data[startd - 1: startd + Align_Nfft - 1], Window)
                    X1_fft_conj = np.conj(fft(X1, Align_Nfft))
                    X2_fft = fft(X2, Align_Nfft)
                    X1 = ifft(np.multiply(X1_fft_conj, X2_fft), Align_Nfft)

                    X1 = np.abs(X1)
                    v_max = np.max(X1) * 0.99
                    n_max = (v_max ** 0.125) / kernel

                    cond = np.greater(X1, v_max)
                    Hsum = Hsum + n_max * kernel * np.count_nonzero(cond)
                    k = np.arange(1 - kernel, kernel)
                    count_in = np.argwhere(cond)
                    H[np.remainder(count_in + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))
                    # for j in count_in:
                    #     H[np.remainder(j + k + Align_Nfft, Align_Nfft)] += n_max * (kernel - np.fabs(k))

                    startr = startr - int(Align_Nfft / 4)
                    startd = startd - int(Align_Nfft / 4)

                v_max = np.max(H)
                I_max = np.argmax(H)
                if I_max >= (Align_Nfft / 2):
                    I_max = I_max - Align_Nfft

                Utt_D2[bp - 1] = estdelay + I_max
                if Hsum > 0.0:
                    Utt_DC2[bp - 1] = v_max / Hsum
                else:
                    Utt_DC2[bp - 1] = 0.0

    for bp in range(1, N_BPs):
        if (np.fabs(Utt_D2[bp - 1] - Utt_D1[bp - 1]) >= Downsample) \
                and ((Utt_DC1[bp - 1] + Utt_DC2[bp - 1]) > (Best_DC1 + Best_DC2)) \
                and (Utt_DC1[bp - 1] > Utt_DelayConf_l) and (Utt_DC2[bp - 1] > Utt_DelayConf_l):
            Best_ED1 = Utt_ED1[bp - 1]
            Best_D1 = Utt_D1[bp - 1]
            Best_DC1 = Utt_DC1[bp - 1]
            Best_ED2 = Utt_ED2[bp - 1]
            Best_D2 = Utt_D2[bp - 1]
            Best_DC2 = Utt_DC2[bp - 1]
            Best_BP = Utt_BPs[bp - 1]


def time_align(ref_data, ref_Nsamples, deg_data, deg_Nsamples, Utt_id):
    global Utt_DelayEst, Utt_Delay, Utt_DelayConf, UttSearch_Start, UttSearch_End
    global Align_Nfft, Downsample, Window

    estdelay = Utt_DelayEst[Utt_id - 1]
    H = np.zeros(Align_Nfft)

    startr = (UttSearch_Start[Utt_id - 1] - 1) * Downsample + 1
    startd = startr + estdelay
    if startd < 0:
        startr = 1 - estdelay
        startd = 1

    while ((startd + Align_Nfft) <= deg_Nsamples) and \
            ((startr + Align_Nfft) <= ((UttSearch_End[Utt_id - 1] - 1) * Downsample)):
        X1 = np.multiply(ref_data[startr - 1: startr + Align_Nfft - 1], Window)
        X2 = np.multiply(deg_data[startd - 1: startd + Align_Nfft - 1], Window)

        X1_fft = fft(X1, Align_Nfft)
        X2_fft = fft(X2, Align_Nfft)
        X1 = ifft(np.multiply(X1_fft.conj(), X2_fft), Align_Nfft)

        X1 = np.abs(X1)
        v_max = np.multiply(np.max(X1), 0.99)

        H = np.where(np.greater(X1, v_max), H + v_max ** 0.125, H)

        startr = startr + int(Align_Nfft / 4)
        startd = startd + int(Align_Nfft / 4)

    X1 = H
    X2 = np.zeros(Align_Nfft)
    Hsum = np.sum(H)

    X2[0] = 1.0
    kernel = int(Align_Nfft / 64)
    to_count = 1 - np.arange(1, kernel) / kernel
    X2[1: kernel] = to_count
    X2[(Align_Nfft - kernel): (Align_Nfft - 1)] = np.flipud(to_count)

    X1_fft = fft(X1, Align_Nfft)
    X2_fft = fft(X2, Align_Nfft)

    X1 = ifft(np.multiply(X1_fft, X2_fft), Align_Nfft)

    if Hsum > 0:
        H = np.abs(X1) / Hsum
    else:
        H = 0.0

    v_max = np.max(H)
    I_max = np.argmax(H)
    if I_max >= (Align_Nfft / 2):
        I_max = I_max - Align_Nfft

    Utt_Delay[Utt_id - 1] = estdelay + I_max
    Utt_DelayConf[Utt_id - 1] = v_max  # confidence

