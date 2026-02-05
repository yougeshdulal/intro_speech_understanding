import numpy as np
import librosa

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes,order+1) - linear predictive coefficients from each frames
    excitation (nframes,frame_length) - linear prediction excitation frames
      (only the last frame_skip samples in each frame need to be valid)
    '''
    speech = np.asarray(speech, dtype=float)
    frame_length = int(frame_length)
    frame_skip = int(frame_skip)
    order = int(order)
    if frame_length <= 0 or frame_skip <= 0:
        raise ValueError("frame_length and frame_skip must be positive")
    if order < 0:
        raise ValueError("order must be non-negative")

    nframes = 1 + int(np.floor((len(speech) - frame_length) / frame_skip)) if len(speech) >= frame_length else 0
    A = np.zeros((nframes, order + 1), dtype=float)
    excitation = np.zeros((nframes, frame_length), dtype=float)

    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start:start + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        a = librosa.lpc(frame.astype(float), order)
        A[i, :] = a
        e = librosa.lfilter(a, [1.0], frame)
        excitation[i, -frame_skip:] = e[-frame_skip:]

    return A, excitation

def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes,order+1) - linear predictive coefficients from each frames
    frame_skip (1) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    e = np.asarray(e, dtype=float)
    A = np.asarray(A, dtype=float)
    frame_skip = int(frame_skip)
    nframes = A.shape[0]
    duration = nframes * frame_skip
    if e.shape[0] < duration:
        e = np.pad(e, (0, duration - e.shape[0]))
    else:
        e = e[:duration]

    synthesis = np.zeros(duration, dtype=float)
    order = A.shape[1] - 1
    zi = np.zeros(order, dtype=float)

    for i in range(nframes):
        seg = e[i * frame_skip:(i + 1) * frame_skip]
        y, zi = librosa.lfilter([1.0], A[i, :], seg, zi=zi)
        synthesis[i * frame_skip:(i + 1) * frame_skip] = y

    return synthesis

def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes,frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes*frame_skip) - excitation for the robot voice
    '''
    excitation = np.asarray(excitation, dtype=float)
    T0 = int(T0)
    frame_skip = int(frame_skip)
    nframes = excitation.shape[0]
    gain = np.sqrt(np.mean(excitation[:, -frame_skip:] ** 2, axis=1) + 1e-12)

    e_robot = np.zeros(nframes * frame_skip, dtype=float)
    if T0 <= 0:
        return gain, e_robot

    for n in range(0, e_robot.shape[0], T0):
        e_robot[n] = 1.0

    for i in range(nframes):
        e_robot[i * frame_skip:(i + 1) * frame_skip] *= gain[i]

    return gain, e_robot
