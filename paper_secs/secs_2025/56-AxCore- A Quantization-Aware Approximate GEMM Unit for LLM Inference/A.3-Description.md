# A.3 Description

A.3.1 How to access. We archive the source code at [https://doi.org/](https://doi.org/10.5281/zenodo.16895417) [10.5281/zenodo.16895417.](https://doi.org/10.5281/zenodo.16895417) We recommend you access our GitHub repository<https://github.com/CLab-HKUST-GZ/micro58-axcore> for the latest version.

A.3.2 Hardware dependencies. We evaluate the models with our server equipped with four NVIDIA RTX 6000 Ada GPUs (48GB).

A.3.3 Software dependencies. The experiments rely on the following software components.

- Ubuntu 22.04.5 LTS
- Python 3.9

- <span id="page-14-0"></span>• PyTorch 2.5.1
- Conda 25.1.1
- GCC 11.4.0
- CUDA 12.4
- Cacti 7.0

A.3.4 Data sets. We evaluate perplexity on the WikiText-2 dataset. For zero-shot evaluations, we employ a suite of benchmarks, including ARC-e, HellaSwag, PiQA, and Winogrande. Additionally, the Pile dataset is used during the calibration of AxCore to mitigate overfitting.

A.3.5 Models. We evaluate a suite of foundation models from the Hugging Face Hub. For perplexity measurements, we use OPT-2.7B, OPT-6.7B, OPT-13B, OPT-30B, LLaMA2-7B, and LLaMA2-70B. For the zero-shot performance evaluation, we then focus on the two largest models: OPT-30B and LLaMA2-70B.

