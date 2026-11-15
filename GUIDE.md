# Neural Information-Preserving EEG Denoising — Project Guide (Review 2)

Thesis: *Neural Information-Preserving EEG Denoising Using Adaptive Artifact Separation* (Aditya Bhoyar, 225EC3024, NIT Rourkela)

---
## 0. What is finished and what is not (be honest with the panel)

| Part | Status |
|---|---|
| Baselines: ICA, NLMS, notch, low-pass straw-man, wavelet | ICA/NLMS/notch/low-pass **run and tested** on semi-synthetic data. Wavelet needs `PyWavelets` and was not run. |
| Stage 1: artifact models (EOG, EMG, motion, 50 Hz) with spatial/spectral/temporal signatures | **Implemented and tested** (`artifacts.py`) |
| **NIPI** (7 components + aggregation + weight-robustness + sanity tests) | **Implemented and tested** (`nipi.py`, `tests/`) |
| Stage 2 deep model (graph prior + attention + NIPI-aware losses + feedback loop) | **Code written, syntax-checked only. Not trained**, because the build sandbox has no PyTorch. You must run `train.py` and fix any shape or bug on your machine. |
| Real-data pipeline (Siena, PhysioNet MI, Biosource) | `data_real.py` written, **not run**. Channel-name matching may need small edits for your files. |
| Stage 3 physiological validation (NIPI + task + connectivity) | NIPI is done. Final tables on real data come after you run Steps 2–4 below. |

Do not show any number that you did not produce yourself on your own machine.

---
## 1. Core idea, in one slide

Conventional denoising solves `min ‖y − x_clean‖` (artifact energy / RMSE).
Proposed: `min  Artifact energy  −  β · Neural-information preserved`, where the second term is measured by **NIPI**.

Information-theoretic form: let `x = s + n` (neural signal `s`, artifact `n`) and `y = f(x)`.
By the data-processing inequality `I(s;y) ≤ I(s;x)`, so a denoiser can never add information. The best case is to lose none:

`max_f  I(s; y) − β · I(n; y)`   (an information-bottleneck-style objective)

- `I(s;y)` is lower-bounded in practice by **InfoNCE**: `I ≥ log B − L_NCE` (`losses.infonce`).
- `I(n;y)` is controlled by the artifact-residual / reconstruction terms.

> **Wording warning.** The problem statement says "*provably* preserving". You do not have a general proof. Say: "**quantifiably** preserving, with a mutual-information lower bound and a measurable index (NIPI)". Claiming a proof invites a hard question you cannot answer. Honest framing is safer and still strong.

---
## 2. How NIPI is created (this is the main contribution)

**Inputs:** clean reference `x`, denoised `y`, both `(epochs N, channels 19, samples T)`, labels, fs.
**Why a clean reference is needed:** information preservation can only be measured against what was there. Real EEG has no ground truth, so use the **semi-synthetic protocol** (as in EEGdenoiseNet): pick artifact-free epochs → treat them as truth → add artifacts at known SNR → denoise → compare.
(Fix for Review 1: you planned RMSE/correlation on *real* contaminated data; that has no ground truth. Use semi-synthetic for all numbers, and real data only for qualitative plots.)

**Common rule:** every component gives a relative error `r`, converted to a score `s = exp(−r / r0)` in (0,1]; 1 = perfectly preserved.

| # | Component (PS wording) | What is computed | Code |
|---|---|---|---|
| 1 | Spectral consistency | mean abs log-ratio of band power (δ θ α β γ), Welch | `spectral_consistency` |
| 2 | Phase-locking preservation | (a) phase fidelity \|⟨e^{i(φx−φy)}⟩\| per channel in θ/α/β; (b) relative error of inter-channel PLV matrices; geometric mean of (a),(b) | `phase_preservation` |
| 3 | Connectivity retention | relative error of band amplitude-envelope-correlation (AEC) matrices | `connectivity_retention` |
| 4 | Covariance (Riemannian) geometry | affine-invariant distance `√Σ ln²λᵢ`, λ = generalised eigenvalues of (C_y, C_x) | `covariance_geometry` |
| 5 | ERP morphology | relative L2 error of class-averaged ERP + single-trial waveform correlation in the post-stimulus window | `erp_morphology` |
| 6 | Entropy preservation | relative change of permutation entropy and spectral entropy (r0 = 0.1) | `entropy_preservation` |
| 7 | Task discriminability retention | `(acc_den − chance)/(acc_clean − chance)`, 5-fold CV logistic regression on log band power | `task_retention` |

**Aggregation:** `NIPI = ∏ s_k^{w_k}`, weighted **geometric** mean, default `w_k = 1/7`.
Why geometric: it is non-compensatory. Destroying phase cannot be hidden by keeping entropy.
**Always report the 7 sub-scores next to the single number.**

**Companion metrics (important):**
- ARR = `1 − ‖y−x‖² / ‖noisy−x‖²` (artifact removal), plus RMSE, CC, SNR gain.
- `Joint = harmonic mean(ARR, NIPI)`. Doing nothing (ARR 0) and over-smoothing (NIPI ≈ 0) both score ≈ 0.

**Validation of the metric itself** (`python tests/test_nipi_properties.py`, all pass):
1. identity → NIPI = 1.
2. more noise → NIPI strictly decreases (0.57 → 0.42 → 0.29 → 0.19 for SNR 10, 5, 0, −5 dB).
3. 8 Hz low-pass (looks "clean") → NIPI < 0.6.
4. amplitude shrink ×0.5 → phase stays > 0.95, spectral/ERP drop (the components are separable).
5. 40 ms time shift → phase drops, spectrum unaffected.
6. white noise → task and NIPI collapse.

**Still to do for a convincing thesis chapter** (1–2 days of work each):
- Weight sensitivity: `rank_stability()` (already coded). Demo gave mean Kendall τ = 0.84 (min 0.20). Report it, and discuss which pair of methods swaps.
- Correlation of NIPI with downstream BCI accuracy across methods and SNRs (Spearman).
- Ablation: drop each component and show the ranking changes.
- Test on real MI data (ERD/ERS in place of ERP if no stimulus-locked response).

---
## 3. Proposed method (3 stages)

| Stage | What | Where |
|---|---|---|
| 1 Artifact characterisation | Explicit EOG/EMG/motion/50 Hz generators (spatial weights, bands, durations). Network has an artifact-type head and an electrode-graph prior (`[x, Âx, Â²x]`). | `artifacts.py`, `model.py` |
| 2 Neural-dynamics preservation | Predict artifact `n̂`, output `y = x − n̂`. SE + temporal self-attention. Losses that are differentiable surrogates of NIPI parts: log band power, phase cosine, band connectivity, covariance, graph Dirichlet energy (GSP), InfoNCE. | `model.py`, `losses.py` |
| 3 Physiological validation | NIPI, task retention, connectivity, baseline comparison. **Feedback loop:** after each epoch, NIPI components update the loss weights `λ_k ← λ_k · exp(0.2 (1 − s_k))`, clipped to [0.05, 5]. This is the "feedback loop" on your methodology slide. | `train.py` |

Honest notes:
- The covariance loss is a Euclidean surrogate of the Riemannian distance. The *metric* uses the true one.
- The model sees clean targets only in semi-synthetic training (supervised). For real data this is the standard compromise.
- Do not call this "provable". Call it "information-regularised".

---
## 4. How to run on your system

```bash
# 0. Python 3.10+; create env
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install numpy scipy scikit-learn matplotlib pandas torch mne PyWavelets

# 1. METRIC + BASELINES (works now, CPU, ~40 s) -> outputs/phase1_results.csv, phase1_nipi_vs_arr.png
python tests/test_nipi_properties.py
python demo_nipi.py

# 2. BUILD CLEAN REAL EPOCHS (edit channel names if it errors)
python data_real.py --edf S001R04.edf S001R08.edf S001R12.edf --out data/mi_clean.npz --mode mi

# 3. TRAIN + FINAL COMPARISON  (smoke test first!)
python train.py --synthetic --epochs 3          # must run end-to-end before anything else
python train.py --synthetic --epochs 40         # GPU recommended
python train.py --data data/mi_clean.npz --epochs 60
# -> outputs/best_nipdnet.pt, outputs/final_comparison.csv
```

---
## 5. Step-by-step plan (do in this order; each step gives one thing to show)

| Step | Goal | Done when |
|---|---|---|
| 1 | Run Step 1 above; save the figure | tests pass, figure created |
| 2 | Wavelet baseline (`pip install PyWavelets`), rerun demo | wavelet row in table |
| 3 | `python train.py --synthetic --epochs 3` and fix errors | no crash |
| 4 | Full synthetic training; compare with baselines | Proposed row in `final_comparison.csv` |
| 5 | Real MI epochs via `data_real.py`; retrain | real-data table |
| 6 | Ablations: remove one loss term at a time (set its λ = 0) | table: NIPI vs removed term |
| 7 | Weight sensitivity + NIPI vs BCI accuracy correlation | 2 plots |
| 8 | Add EMD, DAE and WGAN baselines | complete baseline set |

**Prompts for other AI tools (paste the file named in brackets as context):**
1. *"Run this PyTorch file on CPU with a 3-epoch smoke test and fix any shape or dtype error without changing the loss definitions."* [`model.py`, `losses.py`, `train.py`]
2. *"Add an EMD baseline using PyEMD with the same `f(noisy, refs, fs)` signature, and a denoising autoencoder baseline in PyTorch."* [`baselines.py`]
3. *"Adapt `data_real.py` to my EDF channel names (paste `raw.ch_names`) and plot 3 epochs."* [`data_real.py`]
4. *"Add ablation and SNR-sweep plots from `outputs/final_comparison.csv`."*
5. Rule for every prompt: **do not change `nipi.py` or `tests/`**. Metric changes must be your decision.

---
## 6. Likely panel questions

- **Is NIPI valid?** Axiom tests (Section 2), component separability, weight robustness (τ), correlation with BCI accuracy (to do).
- **Why a clean reference?** Information loss is only defined relative to the original. Semi-synthetic protocol, same as the literature.
- **Why geometric mean?** Non-compensatory. Sub-scores are always shown.
- **Why not just accuracy of a classifier?** It is one component (task); it misses phase, connectivity and ERP shape.
- **Synthetic vs real?** Synthetic EEG is used only to test the metric. Reported results use real clean epochs plus simulated artifacts.
- **Is ICA's low NIPI fair?** Demo ICA uses my automatic rejection rule (kurtosis + frontal weight, HF power, 50 Hz power). In the thesis, compare against a stronger rule (e.g. ICLabel) so the baseline is not a straw-man.
- **"Provably preserving"?** Information-theoretic bound (DPI, InfoNCE lower bound) plus measured NIPI; not a general proof.

## 7. Limitations to state yourself
- Semi-synthetic artifacts may not match real ones; verify qualitatively on real contaminated recordings.
- The simulator's results are demonstration only.
- NIPI scale constants (r0) are design choices; the sensitivity analysis covers weights and should be extended to r0.
- ERP component needs a stimulus-locked response; for MI use ERD/ERS.
