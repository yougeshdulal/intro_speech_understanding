import numpy as np

def voiced_excitation(duration, F0, Fs):
    '''
    Create voiced speeech excitation.
    
    @param:
    duration (scalar) - length of the excitation, in samples
    F0 (scalar) - pitch frequency, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    excitation (np.ndarray) - the excitation signal, such that
      excitation[n] = -1 if n is an integer multiple of int(np.round(Fs/F0))
      excitation[n] = 0 otherwise
    '''
    N = int(duration)
    excitation = np.zeros(N, dtype=float)
    P = int(np.round(Fs / F0))
    if P <= 0:
        return excitation
    excitation[::P] = -1.0
    return excitation

def resonator(x, F, BW, Fs):
    '''
    Generate the output of a resonator.
    
    @param:
    x (np.ndarray(N)) - the excitation signal
    F (scalar) - resonant frequency, in Hertz
    BW (scalar) - resonant bandwidth, in Hertz
    Fs (scalar) - sampling frequency, in samples/second
    
    @returns:
    y (np.ndarray(N)) - resonant output
    '''
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    y = np.zeros(N, dtype=float)
    r = np.exp(-np.pi * BW / Fs)
    theta = 2.0 * np.pi * F / Fs
    a1 = 2.0 * r * np.cos(theta)
    a2 = -(r * r)
    y1 = 0.0
    y2 = 0.0
    for n in range(N):
        yn = x[n] + a1 * y1 + a2 * y2
        y[n] = yn
        y2 = y1
        y1 = yn
    return y

def synthesize_vowel(duration,F0,F1,F2,F3,F4,BW1,BW2,BW3,BW4,Fs):
    '''
    Synthesize a vowel.
    
    @param:
    duration (scalar) - duration in samples
    F0 (scalar) - pitch frequency in Hertz
    F1 (scalar) - first formant frequency in Hertz
    F2 (scalar) - second formant frequency in Hertz
    F3 (scalar) - third formant frequency in Hertz
    F4 (scalar) - fourth formant frequency in Hertz
    BW1 (scalar) - first formant bandwidth in Hertz
    BW2 (scalar) - second formant bandwidth in Hertz
    BW3 (scalar) - third formant bandwidth in Hertz
    BW4 (scalar) - fourth formant bandwidth in Hertz
    Fs (scalar) - sampling frequency in samples/second
    
    @returns:
    speech (np.ndarray(samples)) - synthesized vowel
    '''
    excitation = voiced_excitation(duration, F0, Fs)
    y = resonator(excitation, F1, BW1, Fs)
    y = resonator(y, F2, BW2, Fs)
    y = resonator(y, F3, BW3, Fs)
    y = resonator(y, F4, BW4, Fs)
    return y


