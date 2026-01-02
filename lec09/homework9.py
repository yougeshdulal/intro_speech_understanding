import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(round(0.025 * Fs))
    step = int(round(0.010 * Fs))

    waveform = np.asarray(waveform)
    N = len(waveform)

    if N < frame_length:
        return []

    num_frames = 1 + (N - frame_length) // step
    energies = np.zeros(num_frames, dtype=float)

    for i in range(num_frames):
        start = i * step
        frame = waveform[start:start + frame_length].astype(float)
        energies[i] = np.sum(frame * frame)

    max_energy = np.max(energies)
    if max_energy == 0:
        return []

    thr = 0.10 * max_energy
    speech_frames = np.where(energies > thr)[0]
    if speech_frames.size == 0:
        return []

    segments = []
    start_f = speech_frames[0]
    prev_f = speech_frames[0]

    for f in speech_frames[1:]:
        if f == prev_f + 1:
            prev_f = f
        else:
            start_sample = start_f * step
            end_sample = prev_f * step + frame_length
            end_sample = min(end_sample, N)
            segments.append(waveform[start_sample:end_sample])
            start_f = f
            prev_f = f

    start_sample = start_f * step
    end_sample = prev_f * step + frame_length
    end_sample = min(end_sample, N)
    segments.append(waveform[start_sample:end_sample])

    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    frame_length = int(round(0.004 * Fs))
    step = int(round(0.002 * Fs))
    half_bins = frame_length // 2 + 1

    models = []
    for seg in segments:
        seg = np.asarray(seg)
        if len(seg) < frame_length:
            continue

        seg_pe = np.empty_like(seg, dtype=float)
        seg_pe[0] = float(seg[0])
        seg_pe[1:] = seg[1:] - 0.97 * seg[:-1]

        if len(seg_pe) < frame_length:
            continue

        num_frames = 1 + (len(seg_pe) - frame_length) // step
        if num_frames <= 0:
            continue

        frames = np.zeros((num_frames, frame_length), dtype=float)
        for i in range(num_frames):
            start = i * step
            frames[i, :] = seg_pe[start:start + frame_length]

        mstft = np.abs(np.fft.fft(frames, axis=1))
        floor = 0.001 * np.amax(mstft)
        mstft_clipped = np.maximum(floor, mstft)
        spec_db = 20.0 * np.log10(mstft_clipped)
        max_db = np.amax(spec_db)
        spec_db = np.maximum(spec_db, max_db - 60.0)

        low_spec = spec_db[:, :half_bins]
        model = np.mean(low_spec, axis=0)
        models.append(model)

    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    Y = len(models)
    K = len(test_models)

    sims = np.zeros((Y, K), dtype=float)

    for y in range(Y):
        m = np.asarray(models[y], dtype=float)
        m_norm = np.linalg.norm(m)
        for k in range(K):
            t = np.asarray(test_models[k], dtype=float)
            t_norm = np.linalg.norm(t)
            denom = m_norm * t_norm
            sims[y, k] = 0.0 if denom == 0 else float(np.dot(m, t) / denom)

    test_outputs = []
    for k in range(K):
        best_y = int(np.argmax(sims[:, k])) if Y > 0 else 0
        test_outputs.append(labels[best_y] if Y > 0 else "")

    return sims, test_outputs