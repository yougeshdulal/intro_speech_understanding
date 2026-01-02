import numpy as np
import torch, torch.nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step,
        then keep only the low-frequency half (the non-aliased half).
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    
    '''
    waveform = np.asarray(waveform)
    N = len(waveform)

    fl_feat = int(round(0.004 * Fs))
    st_feat = int(round(0.002 * Fs))
    fl_vad = int(round(0.025 * Fs))
    st_vad = int(round(0.010 * Fs))

    if N < max(fl_feat, fl_vad) or fl_feat <= 0 or st_feat <= 0 or fl_vad <= 0 or st_vad <= 0:
        return np.zeros((0, max(1, fl_feat // 2 + 1)), dtype=float), np.zeros((0,), dtype=int)

    pe = np.empty(N, dtype=float)
    pe[0] = float(waveform[0])
    pe[1:] = waveform[1:] - 0.97 * waveform[:-1]

    nf_feat = 1 + (N - fl_feat) // st_feat
    idx_feat = (np.arange(nf_feat)[:, None] * st_feat) + np.arange(fl_feat)[None, :]
    frames_feat = pe[idx_feat]
    mstft = np.abs(np.fft.fft(frames_feat, axis=1))

    floor = 0.001 * np.amax(mstft)
    mstft = np.maximum(floor, mstft)
    spec = 20.0 * np.log10(mstft)
    max_db = np.amax(spec)
    spec = np.maximum(spec, max_db - 60.0)

    half_bins = fl_feat // 2 + 1
    features = spec[:, :half_bins].astype(float)

    nf_vad = 1 + (N - fl_vad) // st_vad
    idx_vad = (np.arange(nf_vad)[:, None] * st_vad) + np.arange(fl_vad)[None, :]
    frames_vad = waveform[idx_vad].astype(float)
    energies = np.sum(frames_vad * frames_vad, axis=1)

    max_energy = np.max(energies)
    vad_labels = np.zeros(nf_vad, dtype=int)
    if max_energy > 0:
        speech = energies > (0.10 * max_energy)
        seg_id = 0
        i = 0
        while i < nf_vad:
            if speech[i]:
                seg_id += 1
                while i < nf_vad and speech[i]:
                    vad_labels[i] = seg_id
                    i += 1
            else:
                i += 1

    labels = np.repeat(vad_labels, 5)
    if len(labels) >= nf_feat:
        labels = labels[:nf_feat]
    else:
        labels = np.pad(labels, (0, nf_feat - len(labels)), constant_values=0)

    return features, labels.astype(int)

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors:
        Pre-emphasize the signal, then compute the spectrogram with a 4ms frame length and 2ms step.
    labels (NFRAMES) - numpy array of labels (integers):
        Calculate VAD with a 25ms window and 10ms skip. Find start time and end time of each segment.
        Then give every non-silent segment a different label.  Repeat each label five times.
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training

    The model should be Sequential(LayerNorm, Linear), 
    input dimension = NFEATS = number of columns in "features",
    output dimension = 1 + max(labels)

    The lossvalues should be computed using a CrossEntropy loss.
    '''
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    nfeats = features.shape[1] if features.ndim == 2 else 0
    nlabels = int(1 + np.max(labels)) if labels.size > 0 else 1

    model = torch.nn.Sequential(
        torch.nn.LayerNorm(nfeats),
        torch.nn.Linear(nfeats, nlabels)
    )

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    loss_fn = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    lossvalues = np.zeros(int(iterations), dtype=float)

    for i in range(int(iterations)):
        opt.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        lossvalues[i] = float(loss.detach().cpu().item())

    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    features = np.asarray(features, dtype=np.float32)
    X = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X)
        probabilities = torch.softmax(logits, dim=1).detach().numpy()
    return probabilities
