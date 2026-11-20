# Task 15c report
Commit d2df32b. Files: scripts/sensor/sd_eval.py, tests/test_sd_eval.py.
- check_models: --uncond accepts prediction_type in (v, x0), else ValueError naming the file and value; --cond still v only. prov['uncond'] gains prediction_type.
- run_eval: separate trainer_B built with the checkpoint's type for draw_guided; C keeps the 'v' trainer. provenance['b_prediction_type'] (None when B not requested; 'v' default when provenance lacks it).
- Tests: added test_B_accepts_x0_prediction_prior_and_records_it, test_B_refuses_eps_prediction_prior. RED: x0 test failed with ValueError "prediction_type 'x0' vs required 'v'" (eps test already passed, since the old guard refused it). GREEN: tests/test_sd_eval.py + tests/test_guidance.py = 17 passed.
- Adjusted the existing exact provenance key-set assertion to include b_prediction_type (unavoidable, new key).
- Concern: the x0 test uses a Shrink stub (0.5*x_t), since a constant x0 output gives guidance no gradient path (Const(0.0) fails under x0; fine under v). No hard-coded 'v' in model/guidance.py.
