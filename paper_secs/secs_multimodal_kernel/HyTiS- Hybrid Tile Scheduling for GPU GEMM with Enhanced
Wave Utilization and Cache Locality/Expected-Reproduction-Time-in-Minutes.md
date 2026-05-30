# Expected Reproduction Time (in Minutes)

On an H100 GPU, the total estimated computational time for this artifact is approximately 50 hours, including 42 hours for 1.<sup>1</sup> and 8 hours for 1.2. The computational time for 1.<sup>3</sup> is encompassed within that of 1.1, as the necessary data were already collected during the experiments in 1.1. On an A100 GPU, the total computational time is approximately 30% higher than on the H100.

To support faster, targeted evaluation, we additionally provide an option to restrict the benchmark scope to a subset of configurations that expose wave quantization effects, such as the (, 1024, 4096) shape region. By setting environment variable MNs=0, MNe=120, and TASK\_ID=3 when executing run\_tasks.py, users can evaluate 120 representative cases using only the PySIM method, reducing the runtime from 42 hours to approximately 1 hours. These options are fully documented in the revised artifact to facilitate more efficient evaluation.

