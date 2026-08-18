# Install

./Ascend-cann-toolkit\_8.2.RC1.alpha002\_linux-aarch64.run --\ install --install-path=/your/path

./Ascend-cann-kernels-910b\_8.2.RC1.alpha002\_linux-aarch64.\ run --install --install-path=/your/path source /your/path/ascend-toolkit/set\_env.sh

*2) Configure the Conda environment:* Create a Python 3.9 environment and install NPU-specific PyTorch and dependencies:

conda create -n enec python=3.9 -y conda activate enec

pip install pandas numpy==1.24.3 transformers==4.30.0 jinja2 \ decorator attrs psutil absl-py cloudpickle ml-dtypes scipy \ tornado pyyaml

wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp39-\ cp39-manylinux\_2\_17\_aarch64.manylinux2014\_aarch64.\ whl

pip install torch-2.5.1-cp39-cp39-manylinux\_2\_17\_aarch64.\ manylinux2014\_aarch64.whl

wget https://gitee.com/ascend/pytorch/releases/download/v7 .1.0.2-pytorch2.5.1/torch\_npu-2.5.1.post3-cp39-cp39-\ manylinux\_2\_17\_aarch64.manylinux2014\_aarch64.whl

pip install torch\_npu-2.5.1.post3-cp39-cp39-manylinux\_2\_17\_\ aarch64.manylinux2014\_aarch64.whl

*3) Verify the environment:* Run a simple NPU tensor operation to confirm correct setup:

python3 -c "import torch; import torch\_npu; a = torch.randn(3, 4).npu(); print(a + a)"

If the output is normal, the environment is normal.

*4) Build:* Clone the repository and run build\_csrc.sh (1 hour).

git clone https://github.com/jinwuyang/ENEC\_ISCA\_AE.git chmod 777 -R ENEC\_ISCA\_AE cd ENEC\_ISCA\_AE bash build\_csrc.sh

## *E. Experiment workflow*

*1) Data Preparation:* Execute data\_prepare.sh to download and split the model weights. By default, the script only downloads and processes Qwen3-32B (1 hour). To test other models (e.g., DeepSeek-LLM-7B, Falcon-40B), simply uncomment the corresponding lines in data\_prepare.sh.

bash data\_prepare.sh

*2) Performance Testing:* Run compressor\_test.sh to measure the compression ratio and throughput. This script automates parameter searching, compression/decompression profiling, and global analysis. At the end of the execution, it also outputs the end-to-end inference results (2 hours).

source /your/path/ascend-toolkit/set\_env.sh bash compressor\_test.sh

## *F. Evaluation and expected results*

*1) Optimal parameter search results:* The following results show the expected outputs for the Qwen3-32B model:

#### BF16 Model Compression Results

========================================

File Processed: hyperparams\_results.csv Total Elements: 32,761,446,400

------------------------------------------------- Original BF16 Size: 62487.50 MB ENEC Compressed Size: 46265.99 MB Compression Ratio (CR): 1.35x

Model Avg Bit-width: 11.8465 bits/element Exponent Avg Bit-width: 3.8465 bits/element

Formula Avg CR\*: 1.35 x

The optimal parameter search results are organized within the param\_search\_enec/ directory. Each model subfolder (e.g., BF16/Qwen3-32B) provides:

- hyperparams\_results.csv: An exhaustive list of optimal parameters for every model tensor.
- param\_combinations\_stats.txt: A comprehensive statistical report of the search results.
- *2) Compression Ratio and Throughput:* The file summary\_enec.csv summarizes the compression ratio, compression throughput, and decompression throughput of ENEC on 11 models, corresponding to Table [II](#page-8-2) and Figure [9](#page-9-0) in the paper. The expected results for these 11 models are presented as follows:

| model_name dtype<br>compression_ratio_CR |       |       |  |  |  |  |  |  |  |
|------------------------------------------|-------|-------|--|--|--|--|--|--|--|
| compress_throughput_GBps                 |       |       |  |  |  |  |  |  |  |
| decompress_throughput_GBps               |       |       |  |  |  |  |  |  |  |
| Llama-3.1-8B-Instruct<br>BF16            | 1.36  |       |  |  |  |  |  |  |  |
| 376.8                                    | 219.4 |       |  |  |  |  |  |  |  |
| Qwen3-32B<br>BF16                        | 1.35  | 366.3 |  |  |  |  |  |  |  |
| 217.1                                    |       |       |  |  |  |  |  |  |  |
| Qwen3-8B<br>BF16                         | 1.36  | 388.1 |  |  |  |  |  |  |  |
| 222.7                                    |       |       |  |  |  |  |  |  |  |
| deepseek-llm-7b-base<br>BF16             | 1.37  |       |  |  |  |  |  |  |  |
| 391.2                                    | 223.1 |       |  |  |  |  |  |  |  |
| falcon-40b<br>BF16                       | 1.37  | 369.1 |  |  |  |  |  |  |  |
| 217.2                                    |       |       |  |  |  |  |  |  |  |
| falcon-7b<br>BF16                        | 1.36  | 364.6 |  |  |  |  |  |  |  |
| 215.8                                    |       |       |  |  |  |  |  |  |  |
| CapybaraHermes-2.5-Mistral-7B            | FP16  | 1.12  |  |  |  |  |  |  |  |
| 317.0                                    | 195.5 |       |  |  |  |  |  |  |  |
| stable-video-diffusion-img2vid           | FP16  | 1.09  |  |  |  |  |  |  |  |
| 223.4                                    | 148.0 |       |  |  |  |  |  |  |  |
| OLMo-1B-hf<br>FP32                       | 1.15  | 538.6 |  |  |  |  |  |  |  |
| 348.7                                    |       |       |  |  |  |  |  |  |  |
| bert-base-uncased<br>FP32                | 1.15  |       |  |  |  |  |  |  |  |
| 329.1                                    | 252.8 |       |  |  |  |  |  |  |  |
| wav2vec2-large-xlsr-53-english           | FP32  | 1.15  |  |  |  |  |  |  |  |
| 372.1                                    | 254.6 |       |  |  |  |  |  |  |  |
|                                          |       |       |  |  |  |  |  |  |  |

*3) End-to-End Inference Latency:* Figure [10](#page-10-1) in the paper shows the end-to-end inference latency and speedup over the baseline (uncompressed with CPU offloading) for both Qwen3-32B and Falcon-40B under different batch sizes. For brevity, we only present the results for Qwen3-32B with batch size = 1. The expected results are presented as follows:

[Inference: Qwen3-32B]

Configuration: size=61.02 GB, throughput=217.05 GB/s

baseline TTFT: 2.36064 s baseline TPOT: 1.1951 s

ENEC TTFT: 0.556342 s (Speedup: 4.24x) ENEC TPOT: 0.361142 s (Speedup: 3.31x) Result saved to: Latency\_Qwen3-32B\_BF16.csv