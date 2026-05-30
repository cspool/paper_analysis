# <span id="page-16-0"></span>**C.6 Calibration Context Analysis**

We find that the quality of the calibration data is paramount. Figure [C.2](#page-17-0) shows that including the model's full reasoning output in the calibration context improves accuracy across all benchmarks compared to using only the input problems. Qualitatively, using only inputs can induce repetitive generation loops (see Figure [E.1](#page-20-0) for an extreme case), whereas adding the reasoning outputs stabilizes expert utility patterns and provides richer context for measuring decisive expert activations.

