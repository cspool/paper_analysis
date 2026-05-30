# H. Calibration Data Sensitivity

We test robustness to calibration data quantity and quality (Table [F\)](#page-17-1).

Performance is stable across calibration sizes from 50k to 960k tokens (45.4–45.8%). Similarly, calibration data quality shows no clear correlation with accuracy: using Google homepage HTML (low quality) achieves 46.2%, comparable to ShareGPT chat data (46.7%). This confirms that the Q/K statistics captured during calibration are model-intrinsic properties, robust to the choice of calibration data.

