# A Appendix

This appendix provides additional implementation insights and discusses potential future directions for DFVG. We envision extending this work to support multi-round speculative decoding and broader LLM families.

## A.1 Abstract

This artifact contains the implementation of DFVG, a heterogeneous speculative decoding architecture for Large Language Models that leverages FPGA-GPU collaboration. The artifact includes: (1) FPGA-based draft model implementation with custom micro-architecture designed in Verilog HDL, (2) GPU-based verification framework with TreeSort-Verify mechanism, (3) runtime coordination system for cross-device communication, and (4) evaluation scripts and datasets for reproducing the key results demonstrating up to 3.26× speedup and 5.8× energy efficiency improvement over existing approaches.

#### A.2 Artifact check-list (meta-information)

- Algorithm: Speculative decoding, hardware-aware dynamic draft generation, TreeSort-Verify
- Program: DFVG system implementation in Verilog HDL and C++
- Compilation: Xilinx Vivado 2024.1, GCC 9.4.0, CUDA 12.1
- Transformations: Token tree reordering, block-parallel attention computation
- Binary: FPGA bitstream, GPU kernels, host controller executable
- Model: LLaMA-7B, OPT-13B, Qwen3-8B, Vicuna-7B and corresponding draft models
- Data set: MT-Bench, Translation, Summarization, QA, Math reasoning, RAG datasets
- Run-time environment: Ubuntu 20.04, CUDA 12.1, Xilinx Runtime (XRT) 2024.1
- Hardware: AMD V80/U200 FPGA, NVIDIA RTX 4090/A100 GPU, Intel Xeon 4310 CPU

- Run-time state: KV-cache management, cross-device synchronization
- Execution: Heterogeneous pipeline with FPGA draft generation and GPU verification
- Metrics: Latency speedup, energy efficiency, token throughput, acceptance rate
- Output: Performance logs, resource utilization reports, energy consumption data
- Experiments: End-to-end performance, ablation studies, scalability analysis
- How much disk space required (approximately)?: 50GB (models, datasets, logs)
- How much time is needed to prepare workflow (approximately)?: 2-3 hours
- How much time is needed to complete experiments (approximately)?: 8-12 hours
- Publicly available?: Yes, upon paper acceptance
- Code licenses (if publicly available)?: MIT License
- Data licenses (if publicly available)?: Model-specific licenses (Apache 2.0, MIT)
- Workflow automation framework used?: Custom Python scripts with job scheduling
- Archived (provide DOI)?: Will be provided upon acceptance

## A.3 Description

A.3.1 How to access. The source code is at: [https://github.](https://github.com/ShaoqiangLu/DFVG) [com/ShaoqiangLu/DFVG](https://github.com/ShaoqiangLu/DFVG)

## A.3.2 Hardware dependencies.

- FPGA:AMD Versal V80 or U200 FPGA with HBM/DDR memory
- GPU: NVIDIA RTX 4090 or A100 GPU with at least 24GB memory
- CPU: Intel Xeon or equivalent with PCIe Gen4 support
- Memory: At least 64GB system RAM
- Storage: 100GB available disk space for models and datasets
- Connectivity: PCIe Gen4 x16 slots for FPGA-GPU communication

## A.3.3 Software dependencies.

- OS: Ubuntu 20.04 LTS or later
- FPGA Tools: Xilinx Vivado 2024.1, Xilinx Runtime (XRT) 2024.1
- GPU Tools: CUDA 12.1, cuDNN 8.9, NVIDIA Driver 530+
- Compilers: GCC 9.4.0, Python 3.8+
- Libraries: PyTorch 2.0+, Transformers 4.30+, NumPy, NCCL
- Monitoring: NVIDIA-SMI, Xilinx XRT utilities

## A.3.4 Data sets.

- MT-Bench: Multi-turn conversation benchmark
- Translation: WMT translation tasks
- Summarization: CNN/DailyMail, XSum datasets

- Question Answering: SQuAD, Natural Questions
- Math Reasoning: GSM8K, MATH datasets
- RAG: Retrieval-Augmented Generation tasks

#### A.3.5 Models.

- Target Models: LLaMA-7B, OPT-13B, Qwen3-8B, Vicuna-7B
- Draft Models: LLaMA-160M, OPT-125M, Qwen3-0.6B, Vicuna-160M
- Formats: HuggingFace Transformers format, quantized variants

#### A.4 Installation

1. Clone Repository:

git clone https://github.com/ShaoqiangLu/DFVG

2. Install Dependencies:

sudo apt update sudo apt install build-essential cmake pip install -r requirements.txt

3. Setup FPGA Environment:

source /opt/xilinx/xrt/setup.sh export XILINX\_VIVADO=/opt/Xilinx/Vivado/2024.1

4. Build FPGA Bitstream:

cd fpga/ make synthesize make implement

5. Compile GPU Kernels:

cd gpu/ make all

6. Download Models:

python scripts/download\_models.py

## A.5 Experiment workflow

- 1. Environment Setup: Initialize FPGA and GPU devices
- 2. Model Loading: Load target and draft models onto respective devices
- 3. Baseline Measurement: Run autoregressive and standard speculative decoding
- 4. DFVG Evaluation: Execute DFVG with various configurations
- 5. Performance Analysis: Collect latency, throughput, and energy metrics
- 6. Ablation Studies: Evaluate individual component contributions

Example execution:

python scripts/run\_experiments.py --config configs/llama7b.yaml python scripts/collect\_results.py --output results/

#### A.6 Evaluation and expected results

## Key Performance Metrics:

- Latency Speedup: 2.44×–3.26× improvement over autoregressive baseline
- Energy Efficiency: 4.33×–5.79× improvement in energy per token
- Token Throughput: 100-200 tokens/second depending on model size
- Acceptance Rate: 75%-85% for dynamic draft generation
- Resource Utilization: 83% LUT, 79% FF utilization on FPGA

## Expected Output Files:

- performance\_summary.json: Overall speedup and efficiency metrics
- energy\_analysis.csv: Detailed energy consumption breakdown
- ablation\_results.json: Component-wise performance contributions
- resource\_utilization.log: FPGA and GPU resource usage

Reproducibility: Results should be within ±5% of reported values due to hardware variations and thermal conditions.

#### A.7 Experiment customization

## Configuration Parameters:

- Draft Length: Modify configs/draft\_params.yaml
- Batch Size: Adjust BATCH\_SIZE in configuration files
- Model Selection: Change TARGET\_MODEL and DRAFT\_MODEL
- Hardware Mapping: Modify device assignments in device\_config.yaml

## Adding New Models:

python scripts/add\_model.py --target new\_model --draft new\_draft

#### Custom Datasets:

python scripts/prepare\_dataset.py --input custom\_data.json

