## Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  - 实现：Pre-attention same-layer expert prediction 与 prefetching pipeline。核心设计：(1) 在 pre-attention norm 后 clone hidden state 到 CPU，CPU 执行预测器推理（0.15ms），与 GPU 上的 self-attention（0.74-1.13ms）和 post-attention norm（0.08-0.13ms）并行；(2) 预测正确的 expert 在 attention 执行期间从 memory 预取到 GPU（每 expert 0.7-1.6ms），pipeline 到达 expert selection 时直接可用；(3) 预测错误时触发 emergency loading（5.6-8.3ms per expert from disk），但可被 expert computation（6.2-10.3ms）部分隐藏。支持三种部署策略：standard（精确 Top-K）、over-provisioning（多 load experts 换更高 hit rate）、top-1（边缘设备仅并行 load 1 个 expert）。
  - 实验比较：对比 FATE (cross-layer prediction, 78.79% accuracy)、DuoServe-MoE (54-67% top-2 accuracy)、HOB-BIT (55% cache hit rate)。比较指标：Exact-match accuracy、Over-provisioning accuracy、Top-1 accuracy。Expert loading latency 对比：disk→GPU vs memory→GPU，predictor overhead vs attention timing window。
- 硬件平台是什么，配置是什么。
  - 训练：NVIDIA TITAN RTX 24GB GPU。Profiling：Tesla V100-SXM2-32GB、NVIDIA A100-PCIE-40GB、NVIDIA A100 80GB PCIe。预测推理可在 CPU-only 上运行（无 GPU 要求）。
- 开源Serving框架是什么。修改了什么。
  - 论文未在现有开源 serving 框架（如 vLLM、SGLang）上修改。其 prefetching pipeline 设计是框架无关的：pre-attention norm → [fork: GPU self-attention || CPU predictor → expert prefetch] → post-attention norm → expert selection（hit: use prefetched; miss: emergency load）→ expert computation。论文给出了 parallel execution pipeline（Fig.8a-c）的时序设计，但未说明集成到具体 serving 框架的实现细节。
- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - 未开源。Serving pipeline 全流程（DeepSeek-V2-Lite on A100-80GB）：
    - 输入：token 序列经过 embedding → 进入 MoE layer l。
    - Pre-Attention Norm (0.075ms)：RMSNorm 产生 hidden state X。
    - Fork：(a) GPU path——self-attention (0.739ms) → post-attention norm (0.080ms)；(b) CPU path——clone X → CPU predictor f_l(X) (0.15ms) → TopK(s, k) → 发起 expert prefetch 请求。
    - Expert Selection (0.102ms)：Ground-truth router 计算 g = Softmax(W_g · X')，TopK(g, k)。与预测 Ŷ 比对。
    - Hit case：prefetched experts 已在 GPU memory 就绪 → 直接进入 expert computation。
    - Miss case：紧急从 disk/CPU memory load miss 的 experts (single expert: 5.6ms from disk / 0.7ms from memory)，可与已就绪 experts 的 computation 重叠。
    - Expert Computation (6.8ms)：对已加载 experts 执行 FFN 计算。
    - 总 latency savings：1000-token inference session，93.03% accuracy → 569-1352ms total saved（vs FATE 78.79% accuracy）。
