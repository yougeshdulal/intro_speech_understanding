import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''

    waveform = np.asarray(waveform)
    N = len(waveform)


    if frame_length <= 0 or step <= 0:
        raise ValueError("frame_length and step must be positive integers")

  
    if N < frame_length:
        return np.zeros((0, frame_length), dtype=waveform.dtype)

 
    num_frames = 1 + (N - frame_length) // step


    frames = np.zeros((num_frames, frame_length), dtype=waveform.dtype)

   
    for i in range(num_frames):
        start = i * step              
        end = start + frame_length      
        frames[i, :] = waveform[start:end]

    return frames

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
   
    frames = np.asarray(frames)

    
    mstft = np.abs(np.fft.fft(frames, axis=1))

    return mstft

def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    stft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
    '''
   
    mstft = np.asarray(mstft)

    if mstft.size == 0:
        return mstft.astype(float)

 
    floor = 0.001 * np.amax(mstft)

  
    mstft_clipped = np.maximum(floor, mstft)

    
    spectrogram = 20 * np.log10(mstft_clipped)

    
    max_db = np.amax(spectrogram)
    spectrogram = np.maximum(spectrogram, max_db - 60.0)

    return spectrogram
