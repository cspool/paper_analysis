## RWKV__Reinventing_RNNs_for_the_Transformer_Era

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Custom CUDA kernel 用于 WKV 计算。由于 WKV 的逐元素时间依赖计算（`wkv_t = Σe^{-(t-1-i)w+k_i}⊙v_i / Σe^{-(t-1-i)w+k_i}`）在标准深度学习框架中是串行的，论文开发了自定义CUDA kernel来在GPU上高效执行这一计算。其余部分（矩阵乘法、逐点运算）本身已可并行化。
  - 实验比较：论文未对kernel本身进行独立性能benchmark。整体推理性能在Section 6和Appendix K中通过比较RWKV与Transformer家族（BLOOM、OPT、GPT-Neo、Pythia）在文本生成中的时延（s）和内存（CPU RAM、GPU VRAM）来间接评估。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A100 80 GB
  - CPU: x86
  - 推理精度: float32（使用HuggingFace Transformers）

- 评估性能的软件/脚本是什么。修改了什么。
  - 推理框架: HuggingFace Transformers (Wolf et al., 2020)
  - 论文对比了RWKV各规模模型（169M-14B）与BLOOM (560M-3B)、OPT (125M-13B)、GPT-Neo (125M-2.7B)、Pythia (160M-12B)在相同硬件上的文本生成推理时延和内存使用。
  - 论文未说明完全使用CUDA kernel进行推理时的benchmark；HuggingFace Transformers路径可能使用PyTorch原生实现而非自定义CUDA kernel。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/BlinkDL/RWKV-LM（包含CUDA kernel的C++/CUDA实现，位于仓库的cuda/目录）
  - 论文说明：自定义CUDA kernel将WKV的串行扫描在GPU上并行化。kernel的工作原理：WKV计算公式涉及指数加权移动平均（指数衰减权重e^{-w}乘以历史K,V值），这本质上是一个前缀和扫描操作。标准PyTorch实现因每次计算wkv_t需要遍历1..t-1而低效。CUDA kernel通过work-efficient parallel scan算法（如Blelloch scan），将串行O(T)的前缀和分解为O(log T)并行步骤，在GPU上并发处理batch和channel维度。
  - Kernel输入到输出过程：
    1. 输入：k_tensor [B, T, d], v_tensor [B, T, d], w [d]（通道级时间衰减）, u [d]（当前token bonus）
    2. CUDA kernel在GPU上为每个(batch, channel)对执行并行前缀扫描
    3. 输出：wkv_tensor [B, T, d]——每个位置t的加权平均V值
    4. 后续output gating: o_t = W_o @ (σ(r_t) ⊙ wkv_t)，通过标准cuBLAS矩阵乘法完成
  - 论文还指出可以使用更高级的parallel scan方法（Martin and Cundy, 2017）对极长序列进一步并行化。
