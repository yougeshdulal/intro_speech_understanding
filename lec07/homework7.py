import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.
    '''
    N = int(0.5 * Fs)          # half-second worth of samples
    n = np.arange(N)

    f_root = f
    f_third = f * (2 ** (4/12))   # 4 semitones up
    f_fifth = f * (2 ** (7/12))   # 7 semitones up

    x = (np.cos(2*np.pi*f_root*n/Fs) +
         np.cos(2*np.pi*f_third*n/Fs) +
         np.cos(2*np.pi*f_fifth*n/Fs))

    # optional normalization so the waveform stays in [-1, 1]
    x = x / 3.0
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    W[k,n] = cos(2*pi*k*n/N) - j*sin(2*pi*k*n/N) = exp(-j*2*pi*k*n/N)
    '''
    k = np.arange(N).reshape(-1, 1)   # column vector
    n = np.arange(N).reshape(1, -1)   # row vector
    W = np.exp(-1j * 2 * np.pi * k * n / N).astype(complex)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x (in Hz), sorted f1 < f2 < f3.
    '''
<<<<<<< HEAD
    x = np.asarray(x)
    N = len(x)

    W = dft_matrix(N)
    X = W.dot(x)

    # Use only non-negative frequencies (0 .. Fs/2)
    K = N // 2 + 1
    mag = np.abs(X[:K])
    mag[0] = 0.0  # ignore DC

    # indices of 3 largest magnitudes
    idx = np.argpartition(mag, -3)[-3:]
    freqs = idx * (Fs / N)

    freqs_sorted = np.sort(freqs)
    return float(freqs_sorted[0]), float(freqs_sorted[1]), float(freqs_sorted[2])
=======
    raise RuntimeError("You need to write this part")
>>>>>>> 778ea79f2d4b46b64dd323950ad92fda1870da4b
