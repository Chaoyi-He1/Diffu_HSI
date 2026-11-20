# Task 1 report
Implemented data_loader/sensor_model.py (DEVICES, FILES, legacy_axis, raw_matrix, load_sensor; numpy+scipy only), fake_sensor_root fixture in tests/conftest.py, tests/test_sensor_model.py (brief's tests verbatim).
RED: pytest tests/test_sensor_model.py -> ImportError: cannot import name 'sensor_model' (expected, module absent).
GREEN: same command -> 5 passed (including the dataset-marked equality with the legacy loader at 1e-12). tests/test_linear_estimate.py + tests/test_residual_dataset.py -> 18 passed.
Files: data_loader/sensor_model.py (new), tests/test_sensor_model.py (new), tests/conftest.py (fixture appended).
Self-review: matches brief; no existing module touched; pyc files not staged. Concerns: none.
