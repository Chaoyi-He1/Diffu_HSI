# Task 2 report
Implemented data_loader/sensor_forward.py (TILE, N_CHANNELS, mosaic_mask, sigma_from_snr, measure, guidance_operator, apply_H, apply_HT) exactly per brief; b = H @ 1 (= 1/2 R^T A^+ 1).
Tests: tests/test_sensor_forward.py verbatim from brief, 7 passed.
RED: pytest -> ImportError: cannot import name 'sensor_forward' (module missing, expected).
GREEN: `7 passed in 1.05s`.
Files: the two above only. Self-review: no over-building; noise added before masking so off-mask exactly 0. Concerns: none.
