"""EOT predictor.

Usage: python predict.py --data_dir <folder> --out predictions.csv

For each annotated pause it extracts prosodic features from the audio
STRICTLY BEFORE pause_start (causality: we never read a single sample at or
after pause_start, and never use pause_end/duration of the current pause).
It then scores the pause with a gradient-boosting model trained on the
provided english+hindi data (model.pkl, shipped next to this file; if
missing, it is retrained from train_model.py logic is unavailable -> error).
"""
import argparse
import csv
import os
import pickle

import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


# ---------------------------------------------------------------- audio utils
def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    f0 = np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)
    # 3-point median filter on voiced frames to kill octave glitches
    out = f0.copy()
    for i in range(1, len(f0) - 1):
        w = f0[i - 1:i + 2]
        wv = w[w > 0]
        if f0[i] > 0 and len(wv) >= 2:
            out[i] = float(np.median(wv))
    return out


def hz_to_semitones(f0_hz, ref_hz):
    return 12.0 * np.log2(np.maximum(f0_hz, 1e-3) / max(ref_hz, 1e-3))


def spectral_stats(seg, sr):
    """Centroid (Hz) and tilt (dB low band minus high band) of a segment."""
    if len(seg) < 64:
        return 0.0, 0.0
    w = seg * np.hanning(len(seg))
    spec = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), 1.0 / sr)
    tot = spec.sum() + 1e-12
    centroid = float((freqs * spec).sum() / tot)
    lo = spec[(freqs >= 60) & (freqs < 1000)].sum() + 1e-12
    hi = spec[(freqs >= 1000) & (freqs < 5000)].sum() + 1e-12
    tilt = float(10 * np.log10(lo / hi))
    return centroid, tilt


def mel_filterbank(sr, n_fft, n_mels=12, fmin=100.0, fmax=4000.0):
    def hz2mel(h): return 2595.0 * np.log10(1 + h / 700.0)
    def mel2hz(m): return 700.0 * (10 ** (m / 2595.0) - 1)
    mels = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    hz = mel2hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        if c > l: fb[i, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c: fb[i, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = {}
def logmel(seg, sr, n_mels=12):
    """Mean log-mel spectrum of a segment."""
    n_fft = 512
    if len(seg) < n_fft:
        return np.zeros(n_mels, dtype=np.float32)
    if sr not in _FB:
        _FB[sr] = mel_filterbank(sr, n_fft, n_mels)
    hop = 160
    n = 1 + (len(seg) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    fr = seg[idx] * np.hanning(n_fft)[None, :]
    spec = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    mel = spec @ _FB[sr].T
    return np.log10(mel + 1e-10).mean(axis=0).astype(np.float32)


# ---------------------------------------------------------------- features
N_FEATURES = 57
HOP_S = HOP_MS / 1000.0


def extract_features(x, sr, pause_start, prior_pauses):
    """Features from audio[0 : pause_start] only.

    prior_pauses: list of (start, end) of pauses in the same turn whose
    pause_end < pause_start of the current pause (i.e., already finished,
    fully in the past -> causal to use).
    """
    end = int(pause_start * sr)
    audio = x[:end]                       # everything so far (causal)
    seg = audio[max(0, end - int(1.5 * sr)):]   # last 1.5 s before pause
    f = np.zeros(N_FEATURES, dtype=np.float32)
    if len(seg) < sr // 20:
        return f

    # --- energy shape into the pause
    e = frame_energy_db(seg, sr)
    if len(e) == 0:
        return f
    peak = float(np.max(e))
    f[0] = float(np.mean(e[-10:])) - peak          # final 100ms energy vs peak
    f[1] = float(np.mean(e[-30:])) - peak          # final 300ms vs peak
    n = min(50, len(e))
    if n >= 5:                                     # dB/frame slope, last 500ms
        t = np.arange(n)
        f[2] = float(np.polyfit(t, e[-n:], 1)[0])
    f[3] = float(np.std(e[-30:]))                  # energy variability at end

    # --- pitch: speaker-normalized final contour
    f0_all = f0_contour(audio[max(0, end - int(6.0 * sr)):], sr)  # up to 6 s of context
    voiced_all = f0_all[f0_all > 0]
    med = float(np.median(voiced_all)) if len(voiced_all) >= 5 else 150.0
    f0 = f0_contour(seg, sr)
    v = f0 > 0
    f[4] = float(np.mean(v))                       # voiced fraction last 1.5s
    if v.any():
        idx = np.where(v)[0]
        # final voiced run
        run_end = idx[-1]
        run_start = run_end
        while run_start - 1 in idx:
            run_start -= 1
        run = f0[run_start:run_end + 1]
        run = run[run > 0]
        f[5] = (run_end - run_start + 1) * HOP_S   # final voiced run length (final lengthening)
        f[6] = (len(f0) - 1 - run_end) * HOP_S     # unvoiced tail before pause (trailing consonant/breath)
        st = hz_to_semitones(run, med)
        f[7] = float(st[-1])                       # final pitch rel. speaker median (st)
        f[8] = float(np.mean(st[-min(5, len(st)):]))
        if len(st) >= 4:                           # F0 slope over final run (st/s)
            t = np.arange(len(st)) * HOP_S
            f[9] = float(np.polyfit(t, st, 1)[0])
        f[10] = float(st.max() - st.min())         # pitch range in final run
        # last 300 ms of voiced frames anywhere in seg
        tail = f0[-30:]
        tv = tail[tail > 0]
        if len(tv):
            f[11] = float(np.mean(hz_to_semitones(tv, med)))
        # pitch slope across the whole last 1.5 s (voiced frames, robust)
        vi = np.where(v)[0]
        if len(vi) >= 6:
            st_all = hz_to_semitones(f0[vi], med)
            f[12] = float(np.polyfit(vi * HOP_S, st_all, 1)[0])

    # --- spectral character of the very end (trailing fricatives, breathiness)
    c200, t200 = spectral_stats(seg[-int(0.2 * sr):], sr)
    f[13] = c200 / 1000.0
    f[14] = t200
    zc = np.mean(np.abs(np.diff(np.signbit(seg[-int(0.2 * sr):]).astype(np.int8))))
    f[15] = float(zc)

    # --- turn-position / timing context (all strictly past)
    f[16] = min(pause_start, 30.0)                 # time elapsed in turn
    f[17] = float(len(prior_pauses))               # pauses already taken
    if prior_pauses:
        last_s, last_e = prior_pauses[-1]
        f[18] = min(pause_start - last_e, 15.0)    # speech burst length since last pause
        f[19] = min(last_e - last_s, 3.0)          # duration of previous pause
        sp = pause_start - sum(pe - ps for ps, pe in prior_pauses)
        f[20] = min(sp, 30.0)                      # net speech time so far
    else:
        f[18] = min(pause_start, 15.0)
        f[20] = min(pause_start, 30.0)
    # energy of final 300ms relative to the whole-turn-so-far loudness
    e_all = frame_energy_db(audio[max(0, end - int(6.0 * sr)):], sr)
    if len(e_all) >= 10:
        f[21] = float(np.mean(e[-30:])) - float(np.percentile(e_all, 90))
    f[22] = med / 100.0                            # speaker pitch register
    f[23] = float(np.mean(e[-5:])) - peak          # final 50ms vs peak

    # --- extra prosody
    if v.any():
        st_run = hz_to_semitones(run, med) if len(run) else np.zeros(1)
        if len(st_run) >= 2:
            f[24] = float(st_run[-1] - st_run[0])          # F0 fall over final run
            k3 = min(3, len(st_run))
            f[25] = float(np.mean(st_run[-k3:]) - np.max(st_run))  # end vs peak of run
        # normalized final lengthening: run length vs median voiced run in seg
        runs = []
        cur = 0
        for vv in v:
            if vv: cur += 1
            elif cur: runs.append(cur); cur = 0
        if cur: runs.append(cur)
        if len(runs) >= 2:
            f[26] = runs[-1] / (np.median(runs) + 1e-6)
    # syllable-rate proxy: energy peaks per second in last 1.5 s
    if len(e) >= 10:
        em = e - np.median(e)
        peaks = 0
        for i in range(1, len(em) - 1):
            if em[i] > em[i-1] and em[i] >= em[i+1] and em[i] > 3.0:
                peaks += 1
        f[27] = peaks / (len(e) * HOP_S)
        half = len(e) // 2
        f[28] = float(np.mean(e[half:]) - np.mean(e[:half]))  # energy trend halves
        f[29] = float(np.mean(e[-30:] < (peak - 25)))          # near-silent frac last 300ms
        n2 = min(20, len(e))
        t2 = np.arange(n2)
        f[30] = float(np.polyfit(t2, e[-n2:], 1)[0])           # slope last 200ms
    c500, t500 = spectral_stats(seg[-int(0.5 * sr):], sr)
    f[31] = c500 / 1000.0
    f[32] = t500

    # --- log-mel snapshot of the final 300 ms, and its delta vs the prior 300 ms
    m_last = logmel(seg[-int(0.3 * sr):], sr)
    m_prev = logmel(seg[-int(0.6 * sr):-int(0.3 * sr)], sr) if len(seg) >= int(0.6 * sr) else m_last
    # normalize out channel/loudness: subtract each snapshot's own mean
    f[33:45] = m_last - m_last.mean()
    f[45:57] = m_last - m_prev

    return f


def build_matrix(data_dir, rows):
    cache = {}
    # group rows per turn, sorted by pause_start, to compute prior pauses causally
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    for t in by_turn.values():
        t.sort(key=lambda r: float(r["pause_start"]))
    X, keys = [], []
    for tid, trs in by_turn.items():
        prior = []
        for r in trs:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            ps = float(r["pause_start"])
            X.append(extract_features(x, sr, ps, list(prior)))
            keys.append((r["turn_id"], r["pause_index"]))
            # this pause is now in the past for the NEXT pause of the turn
            prior.append((ps, float(r["pause_end"])))
    return np.array(X, dtype=np.float32), keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    X, keys = build_matrix(args.data_dir, rows)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    with open(model_path, "rb") as fh:
        models = pickle.load(fh)          # list of fitted classifiers
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pe in zip(keys, p):
            w.writerow([tid, pi, f"{pe:.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
