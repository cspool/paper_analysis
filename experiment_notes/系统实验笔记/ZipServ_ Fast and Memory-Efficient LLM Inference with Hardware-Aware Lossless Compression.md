## ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression

- 属于Serving调度的实现是什么？实验比较什么？
  提出stage-aware inference strategy：根据prefill（compute-bound）和decode（memory-bound）阶段的不同特性选择不同执行路径。decode阶段使用fused ZipGEMM kernel，实现"load-compressed, compute-decompressed"执行模型——直接从DRAM读取压缩权重，在register file内解压并直接送入Tensor Core计算，消除中间global memory buffer，提高compute intensity约50%（相比标准GEMM）。prefill阶段使用decoupled pipeline——先用高效decompression kernel解压到global memory，再用cuBLAS_TC做高吞吐GEMM，利用prefill高算术强度摊销解压开销（<4% overhead）。实验比较end-to-end latency和throughput，对比vLLM、Transformers和DFloat11；还比较memory consumption和KV cache扩展能力。

- 硬件平台是什么，配置是什么。
  (1) 1× RTX4090跑LLaMA3.1-8B；(2) 2× L40S跑Mistral-24B (tensor parallelism)；(3) 4× L40S跑LLaMA3.1-70B (tensor parallelism)。batch size 8/32，output length 128/256/512/1024/2048 tokens。

- 开源Serving框架是什么。修改了什么。
  基于vLLM扩展，约1.0K行Python glue code通过PyBind11集成自定义CUDA kernel（ZipGEMM + Decompression kernel）。修改：(1) vLLM model loader：支持加载TCA-TBE压缩格式的权重；(2) linear execution module：根据prefill/decode阶段选择不同kernel——decode用fused ZipGEMM，prefill用decoupled decompression+cuBLAS_TC；(3) weight memory management：压缩权重从14.96GB降至10.83GB (LLaMA3.1-8B)，释放的内存自动分配给KV cache (PagedAttention)，KV cache从5.07GB扩展到8.60GB（1.70×增加）。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  已开源：https://github.com/HPMLL/ZipServ_ASPLOS26.git。约3.5K行代码（2.5K CUDA/C++ + 1.0K Python）。使用流程：
  1. 环境准备：CUDA 12.4，PyTorch，vLLM
  2. 克隆仓库编译：`git clone https://github.com/HPMLL/ZipServ_ASPLOS26.git && cd ZipServ && mkdir build && cd build && cmake .. && make` 生成ZipGEMM .so
  3. 离线压缩模型：`python compress.py --model Llama-3.1-8B-Instruct --output compressed_model/` 产生TCA-TBE格式
  4. 加载到vLLM推理：压缩权重以PyBind11调用CUDA kernel，decode阶段自动使用fused ZipGEMM
  5. 端到端效果：LLaMA3.1-8B在RTX4090上，output 2048 tokens batch 32下，throughput 1105 tok/s（1.66× over vLLM），平均1.22× end-to-end加速；内存节省25-29%。例如Mistral-24B weight从43.92GB压缩至31.30GB (28.7% reduction)，释放的~12.6GB内存用于扩展KV cache支持更大batch size或更长context。

