# HW5 implementation and measured evidence

Repository: https://github.com/vhamz/mf-movie-recommender
Demo: https://vhamz.github.io/mf-movie-recommender/

Run `python3 -m http.server 8055` here and open http://localhost:8055/.
Run tests, Train, Recommend Top-10, and Evaluate Recall@10. Training is
synchronous in the supplied interface and can keep the tab busy temporarily.
TensorFlow.js 4.22.0 loads from the supplied CDN script.

The six student functions implement observed-only mini-batch Adam training,
16-dimensional factors, mean plus dot-product prediction, validation checkpoint
restoration, training-catalog Top-K with train/validation filtering, and macro
Recall@10. Initialization seeds are 42/43; shuffle seed is 2026.
L2 = reg * mean(||p_u||^2 + ||q_i||^2) over batch observations.

One starter fixture was corrected: a rank-11 recall test requires eleven eligible
training candidates, but the original fixture supplied only two after masking.
Movies 4-12 are now training observations of another user. No application data,
HTML or CSS was changed.

## Measured default run

Chrome WebGL backend, 20 epochs, learning rate 0.01, L2 0.05, batch size 64.
Best epoch: 2. Validation RMSE: 1.002885. Test RMSE: 1.043254.
Mean baseline test RMSE: 1.228900. Macro Recall@10: 0.020015618.
Evaluated recall users: 861. Exclusions: 13 distinct unseen relevant movie ids,
82 users without an eligible relevant test movie.
Full-split RMSE uses training-mean fallback for movies absent from training.
Warm-only test excludes 53 ratings and gives MF RMSE 1.038503 (baseline 1.225719).
User 405 has test ratings but no eligible relevant test movie; their Top-10 is
shown as a cautionary case, and they are excluded from the recall average.

All 11 supplied self-checks pass after correcting the fixture. A separate toy
check proves best-weight restoration and stable own-model tensor counts.
Physical GPU acceleration is not inferred merely from the WebGL backend name.

## Independent verification

`evidence/full-training.json` contains all 20 epoch metrics from the browser.
`evidence/results.json` includes the restored checkpoint weights, metrics,
two recommendation lists, and a fresh 2-epoch rerun. That rerun reproduced the
20-epoch run's restored test RMSE within 1e-6. Its timing is not full-run timing.
`evidence/extra-checks.json` records the checkpoint and memory toy experiment.

With Python 3 and NumPy installed, run `python3 evidence/verify.py`.
It rebuilds the chronological splits, recalculates baseline and factor RMSE,
recomputes macro recall and checks recommendation filtering from exported weights.
Backend floating-point differences can slightly alter metrics or tie ordering.

Dataset SHA-256:
u.data: 06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490
u.item: 553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701
