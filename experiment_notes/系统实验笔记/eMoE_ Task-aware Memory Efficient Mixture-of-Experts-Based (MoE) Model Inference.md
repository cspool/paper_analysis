## eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

- 属于Serving调度的实现是什么？实验比较什么？
  eMoE 是一个面向 MoE-based LLM 的记忆体高效推理系统，由四个协同组件构成：(1) **Expert Prediction**：使用 BERT-XLNet（0.108B 参数）基于历史 expert 路由分布预测未来的 expert 序列，支持逐层预测（eMoE-L，用 prev-layer expert 预测 next-layer expert）和全层预测（eMoE-A，用上一条 prompt 的 expert 分布预测当前全部层 expert）；(2) **Periodic Expert Invocation**：每 p 条 prompt（实验确定 p=40）调用一次预测器，reuse expert 减少加载开销而基本不影响 perplexity；(3) **Task-aware Expert Loading**：利用不同任务对 token-to-expert routing accuracy 的敏感度差异，仅对敏感任务加载预测 expert，对分类/对比任务跳过预测以降低加载延迟；(4) **Task-aware Request Scheduling**：联合考虑用户 SLO、profiled 任务特定生成长度、expert 加载延迟进行贪心调度，最小化端到端推理延迟。

  实验比较对象：vLLM、DeepSpeedFastGen（端到端延迟）、Pre-gatedMoE、MoEInfinity、Random（记忆体消耗与准确率）、量化模型 4-bit/8-bit（记忆体 vs 准确率 trade-off）。

- 硬件平台是什么，配置是什么。
  Intel Xeon 处理器 + 128GB host memory，4× Nvidia A100 Tensor Core GPU（40GB 设备内存）。推理延迟实验生成 synthetic request trace（Poisson 分布 + multinomial task 分布），最大生成 token 数设为 1000 以避免 OOM。

- 开源Serving框架是什么。修改了什么。
  基于 **DeepSpeed-FastGen**（github.com/microsoft/DeepSpeed-MII）作为推理引擎，包装 HuggingFace Transformers 模型。对 HuggingFace 模型代码做了以下扩展：
  1. 为每个 MoE 层维护 Python multiprocessing lock，将 MoE layer 计算包装在 lock 内部以同步 expert 加载与计算；
  2. 为每个 MoE 层的 expert 加载包装 CUDA event，MoE 层同步该 event 以防使用 stale model weights；
  3. 异步 expert 加载：通过 `torch.Tensor.copy_(non_blocking=True)` 从 host 到 device 传输 expert，与 non-expert layer 计算重叠；
  4. 条件加载：当前 MoE 层的 expert 加载以前一层加载完成为条件，防止 PCIe 带宽饱和。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未明确提供 eMoE 的独立开源代码仓库。实现基于 DeepSpeed-FastGen（开源）和 HuggingFace Transformers（开源）。专家预测器使用 BERT-XLNet（HuggingFace pretrained），PyTorch 实现，运行在独立进程中。

  **eMoE 推理全流程（Full Pipeline）**：
  ```
  === eMoE Serving 全流程 ===

  Input: inference requests 流 (prompt + task type)，4× A100 40GB GPU + 128GB CPU host memory

  Step 1 ─ Task Type Extraction (CPU)
    For each incoming request:
      Parse input token → match profiled keywords → assign task type
      (SUM/CLSFY/QA/COMP/CONV)

  Step 2 ─ Task-aware Request Scheduling (CPU, 见 Algorithm 1)
    Input: waiting queue Qw, scheduled queue Qs, max tokens Tmax
    1. Sort Qw by SLO stringiness (first-token latency deadline, 升序)
    2. For each request R in Qw:
       if R.inputTokens + Qs.totalTokens < Tmax:
         for each scheduled request S:
           compute t_i = ΔE + (W + n_i·G_i)·c + r_i  (Eq. 3)
           if S.expectedLatency < S.SLO: schedule R
    其中 ΔE = profiled expert loading latency
         G_i = task-specific profiled token生成数（运行时递减）
         c = average expert computation+communication latency per input token

  Step 3 ─ Expert Prediction (GPU, 每 p=40 prompts 调用一次)
    eMoE-A (all-layer):
      Input: expert sequence of previous prompt → [e1^r1, e2^r1, ..., em^r1]
        where e_i = [k expert indices] for top-k routing
      XLNet predicts: [e1^r2, e2^r2, ..., em^r2] for current prompt
    eMoE-L (layer-by-layer):
      Input: expert sequence of layer i-1 → e_{i-1}^r1
      XLNet predicts: e_i^r1 for layer i
    Memory: 0.24%-1.3% of MoE model memory

  Step 4 ─ Task-aware Expert Loading (GPU)
    For each MoE layer:
      1. Compute N_i = (Σ W_j + T·W_o) · s · f_i  (Eq. 2)
         where s ∈ {0,1}: task sensitivity to routing accuracy at this layer
               f_i: predicted routing frequency for expert i
               T: # running requests of this task type
      2. Sum N_i across all task types → expected tokens per expert
      3. Sort experts by expected tokens (descending) → pick top L
         (L set by memory budget)
      4. Load new experts: compare predicted vs already on GPU
         → torch.Tensor.copy_(non_blocking=True) for new experts
         → move unpredicted experts to CPU
      5. Condition: wait for previous MoE layer's loading CUDA event
         → prevents PCIe saturation

  Step 5 ─ Inference Engine (DeepSpeed-FastGen, GPU)
    For each transformer layer:
      a. Self-Attention: dense inference (standard HuggingFace forward)
      b. MoE Layer:
         - Acquire multiprocessing lock on this MoE layer
         - Wait for CUDA event (expert loading complete, prevent stale weights)
         - Router gate: compute gating scores → top-k expert selection
         - Expert FFN: execute only loaded experts
         - Token routing: if expert not on GPU, route to next top-k on GPU
         - Release lock
      c. Continue with loaded experts for subsequent prompts (until next p-th)

  Step 6 ─ Periodic Expert Re-invocation
    Maintain request index counter: 0, 1, 2, ...
    When index % p == 0:
      → goto Step 3 (Expert Prediction)
      → goto Step 4 (Expert Loading)
    Else:
      → reuse currently loaded experts on GPU

  Output: generated token sequences → return to client

  === Key Trade-offs ===
  - Memory vs Accuracy: 60% experts loaded → 98.2%-98.8% accuracy
                     80% experts loaded → 99.6%-99.7% accuracy
  - eMoE-A time overhead: ~0.381s (OpenMoE) / ~0.334s (Mixtral) per predictor call
    eMoE-L time overhead: ~1.387s (OpenMoE) / ~4.211s (Mixtral) per predictor call
    Amortized over 40 prompts: 0.47%-3.11% of avg inference time per request
  ```

  性能结果：
  - 记忆体：减少 **up to 80%** vs Baseline（所有 expert 在 GPU）
  - 延迟：降低 **up to 17%** vs DeepSpeedFastGen（Mixtral-8x22B 最大）
  - Prompt 长度：支持 **40× longer** prompts（Mixtral-8x7B, 512→20480）
  - Batch size：支持 **4.5× larger** batches（Mixtral, 4→18）
  - Throughput：**1.5× higher** tokens/second vs Baseline
  - Accuracy：60% experts loaded → 98.2%-98.8%，80% experts → 99.6%-99.7%
  - vs 量化：eMoE-A accuracy 98.2% vs 量化8-bit 95.2% vs 量化4-bit 91.5%（Mixtral-8x7B）
  - eMoE-A vs eMoE-L: 精度相近，eMoE-A 开销更小（见 §5.7）
