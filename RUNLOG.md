# RUNLOG

All scores = `score.py` mean response delay (ms) @ ≤5% interrupted turns.
Model scores are **out-of-fold** (5-fold GroupKFold, split by turn) — in-sample
numbers on the training folders are meaningless (~100 ms) so we never log them.

| # | run | English | Hindi | change & why |
|---|-----|--------:|------:|--------------|
| 1 | silence-only baseline (`baseline.py`) | **1600** | **850** | Reference. AUC 0.51/0.50. Hindi holds are short (p90 = 0.63 s) so a pure 850 ms timer already works there; English holds are long (p90 = 1.23 s) so silence alone hits the 1.6 s timeout. |
| 2 | 24 prosodic features + GBM, trained on en+hi combined | **1264** | **797** | Energy decay into the pause, speaker-normalized final F0 (level/slope/fall), final voiced-run length, unvoiced tail, spectral tilt/centroid of last 200 ms, turn-position context (elapsed time, #prior pauses, burst length). Combined-language training beat per-language (per-lang: en 1320 / hi 856) — 496 pauses is too few to split. |
| 3 | + median-filtered F0, F0-fall, syllable-rate proxy, near-silence fraction (33 feats) | 1246 | 857 | Cleaned octave glitches; mixed result — Hindi latency is noisy at this data size. |
| 4 | + duration-weighted hold samples (w = 0.2 + 4·dur) | 1246 | **820→764** (sweep) | Insight: a hold can only cause a false cutoff if it outlasts the action delay, so long holds are the only ones that matter. Weighted training pushes the classifier to suppress exactly those. Weight sweep chose (0.2, 4.0). |
| 5 | + 24 log-mel features: mean-normalized snapshot of final 300 ms + delta vs prior 300 ms (57 feats) | 1270 | **720** | Gives the model a crude "what sound am I ending on" cue (trailing fillers/fricatives vs full stops). Big Hindi win. |
| 6 | model sweep: GBM depth-4 / depth-2 / HistGBM | 1210–1285 | 687–767 | Each single config wins one language and loses the other — classic small-data variance. |
| 7 | **final: blend of 9 models** (3 seeds × {GBM-d4, GBM-d2, HistGBM}), duration-weighted | **1250** | **742** | Averaging trades a little peak performance per language for robustness on the unseen (mostly Hindi) test set. Shipped as `model.pkl`. |

Net vs baseline: English 1600 → 1250 ms (−22%), Hindi 850 → 742 ms (−13%),
OOF AUC 0.63 (en) / 0.76 (hi).
