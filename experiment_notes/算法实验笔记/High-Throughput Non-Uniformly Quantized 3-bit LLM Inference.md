## High-Throughput Non-Uniformly Quantized 3-bit LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出Quantix，一个面向non-uniform 3-bit weight-only quantized LLM inference的GPU执行框架。Quantix本身不是新量化算法，而是将已有non-uniform量化方案（SqueezeLLM、Any-Precision、Bitsandbytes等）的压缩权重高效映射到GPU的执行层优化。核心设计：(1) Hardware-aligned bit shuffling：离线将3-bit weight index拆分为1-bit和2-bit两个矩阵（bit dividing），再按Tensor Core tile访问模式重排为连续segments（bit mapping），使内存访问coalesced且无需padding/spanning；(2) Fused dequantization-matmul kernel：在CUDA cores上完成in-register dequantization（bit concatenation重建3-bit index→centroid lookup生成FP16 weight），直接喂给Tensor Cores做matrix-multiplication；(3) Hierarchical software pipeline：inter-tile层用shared memory double buffering重叠global memory prefetch与计算，intra-tile层用register double buffering重叠dequantization与MMA；(4) Split-K parallelization + 128-bit vectorized memory access。实验比较kernel-level speedup（vs FP16 cuBLAS、SqueezeLLM、Any-Precision LLM、GPTQ）和end-to-end throughput（vs SqueezeLLM、FP16、GPTQ、Marlin），并做ablation study评估各优化组件贡献。

- 硬件平台是什么，配置是什么。
  主要平台：NVIDIA L40 GPU（面向LLM inference优化），NVIDIA A100 GPU。kernel benchmark主要在L40上进行。hardware utilization分析使用NVIDIA Nsight。端到端实验在单张A100和双L40 GPU上测试。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA家族（LLaMA-13B/33B/65B, LLaMA2-7B/13B）、OPT家族（OPT-30B/66B/175B）、Vicuna-13B。kernel benchmark从LLaMA和OPT linear layers提取真实weight矩阵shapes，batch size 1-512。端到端实验Vicuna-13B、OPT-30B、LLaMA-65B，单A100和双L40，prompt length固定128 tokens，output length 128-1024 tokens。accuracy评估使用WikiText-2 perplexity和5-shot MMLU（lm-eval harness），LLaMA2-7B和LLaMA2-13B。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：https://github.com/yuang-chen/Quantix-PPoPP26。Quantix算法pipeline：
  1. 离线阶段：(a) 使用SqueezeLLM/Any-Precision/Bitsandbytes等non-uniform quantization对原始FP16 weight做K-means clustering，得到3-bit index矩阵Wq和每行centroids C；(b) Quantix bit dividing将每个3-bit index拆成1-bit matrix Wq,1和2-bit matrix Wq,2，使32个1-bit元素恰好填32-bit word，32个2-bit元素恰好填64-bit word；(c) Quantix bit mapping按64×64 warp tile→16×16 Tensor Core tile的层次，将每个thread负责的元素收集为连续linear segments W1'和W2'，使在线kernel可用128-bit cp.async指令一次抓取。
  2. 在线推理：(a) Fused kernel初始化时prefetch初始warp tiles到shared memory；(b) 主循环中inter-tile层用cp.async异步预取下一K-tile的W1'/W2'/A到shared memory；(c) intra-tile层从shared memory load subtile到registers→CUDA cores做in-register dequantization：bit concatenation [1-bit]+[2-bit]→3-bit index→shift+mask提取→centroid lookup得到FP16 W†→Tensor Cores执行MMA (A×W†)；(d) 两层double buffering使prefetch/dequant/matmul三级重叠；(e) Split-K将K维度切分并行计算partial sums，最后reduction合并。
  3. 例如LLaMA-65B在A100上用3-bit Quantix，batch size 16、token length 128时相对SqueezeLLM最高11.46× speedup；3-bit Quantix让LLaMA-65B可在单GPU运行（FP16无法）。
