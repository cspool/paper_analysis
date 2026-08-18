# *C. Description*

- *1) How to access:* We archive the source code at zenodo: [https://doi.org/10.5281/zenodo.19449314.](https://doi.org/10.5281/zenodo.19449314) We also recommend you to access our GitHub repository for the latest version: [https://github.com/CLab-HKUST-GZ/isca53-unicore.](https://github.com/CLab-HKUST-GZ/isca53-unicore)
- *2) Hardware dependencies:* We evaluate the models with our server equipped with four NVIDIA RTX 6000 Ada GPUs (48 GB).
- *3) Software dependencies:* The experiments rely on the following software components.
  - Ubuntu 22.04.5 LTS
  - Python 3.10.18
  - PyTorch 2.6.0
  - Conda 25.1.1
  - GCC 11.4.0
  - CUDA 12.4
- *4) Data sets:* We evaluate perplexity on the WikiText-2 dataset. For zero-shot evaluations, we employ a suite of benchmarks, including ARC-e, HellaSwag, PiQA, and Winogrande.
- *5) Models:* We evaluate a suite of foundation models from the Hugging Face Hub. For perplexity measurements, we use OPT-6.7B, Llama-2-7B, Llama-2-70B, Llama-3-8B, Qwen3- 8B, Qwen3-14B. For the zero-shot performance evaluation, we focus on the two models: Llama-3-8B and Qwen3-8B.

