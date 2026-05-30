## Priority-Aware Preemptive Scheduling for Mixed-Priority Workloads in MoE Inference

- baseline方法是什么？
  - Baseline 是 HuggingFace TGI（Text Generation Inference）——生产级 LLM 推理引擎，采用 iteration-level scheduling 和 continuous batching。Baseline 的核心问题：(a) priority-oblivious：无法区分 LS（latency-sensitive）和 BE（best-effort）请求，以 FCFS (first-come-first-served) 策略对待所有请求；(b) run-to-completion semantics：每个 decode batch 固定后必须跑完所有 N 层才返回 Scheduler，中途无法插入新请求；(c) Head-of-Line (HOL) Blocking：当 BE 请求先占据 batch 时，LS 请求必须等待当前 iteration 的完整 300-400ms decode iteration 完成后才能被调度；(d) 无 inner-layer state tracking：模型内部只看到 concatenated tensors，无法区分 individual sequence，导致 inner-layer preemption 需要昂贵的 tensor split-merge 操作且可能破坏数据流对齐。
  - 全栈执行例子（Baseline: HF TGI, Mixtral 8×7B, 4-bit 量化, batch_size=32, single A100 80GB）：
    - **算法层**：Mixtral 8×7B MoE 模型：每层 self-attention → router (gating network predicts top-k=2 experts) → selected expert FFN → combine。4-bit GPTQ quantization + FP16 compute precision。无算法层面的 priority 区分——所有 token 经过相同 pipeline。
    - **系统框架层**：HF TGI continuous batching：Job 到达 → Scheduler 接收 → 在 iteration boundary（所有 N 层执行完成后）决定 batch composition。如果 decode batch 有空位且新 job 到达，Scheduler 停止 decode、执行新 job 的 prefill、扩展 batch、然后继续 decode。Batch 内所有 job 的 tensors 被 pad+concatenate 为单一 tensor，model 内部无法区分 individual sequence。Scheduler 仅在 iteration boundary 获得控制权（即每 300-400ms 一次机会）。
    - **编译框架层**：论文未明确说明。HF TGI 使用 PyTorch 原生执行或 torch.compile，无专项编译修改。
    - **kernel调度层**：论文未明确说明。使用标准 PyTorch CUDA kernel（attention: scaled dot-product attention via cuBLAS; expert FFN: GEMM via cuBLAS）。
    - **硬件架构层**：单卡 Nvidia A100 80GB, PCIe 4.0, dual-socket Intel Xeon Gold 6336Y。GPU 执行全流程：prefill phase（compute-bound, 并行处理所有 input tokens）→ decode phase（memory-bound, 逐 token 迭代，每次 iteration 约 300-400ms for 32 layers × (attention + router + expert FFN)）。

  - Baseline 核心缺陷根因：iteration-level granularity (300-400ms per iteration) + FCFS policy 导致即使 LS 请求在 BE batch 执行的第 1 层就到达，也必须等待全部 32 层完成（300-400ms 级延迟）才能被调度。每个 iteration 内无抢占能力——Scheduler 在 iteration boundary 才取回控制权，且模型内部无 per-sequence state 支持任意点恢复。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QLLM——在 HF TGI 基础上实现 expert-level priority-aware preemptive scheduling。核心创新：(1) Per-Expert FIFO Queues 打破 layer-wise barrier，使每个 expert 可独立处理 token；(2) Scheduler 通过 closed-loop feedback 在任意 layer 的 router 后 preempt BE batch；(3) Sequence/Batch Facade Pattern 支持 zero-copy individual sequence 状态保存/恢复；(4) Unified Dynamic KV Cache 避免大 tensor split-merge。
  - 全栈执行例子（QLLM, Mixtral 8×7B, 4-bit 量化, batch_size=32, single A100 80GB）：
    - **算法层（解决"无法区分 priority"）**：在 MoE layer 内插入 per-expert FIFO queues——router 输出的 top-k=2 expert selection 将 sequence reference push 进对应 expert 队列。LS sequence 优先入队（通过 policy 在 preempt 时先处理 LS）。QLLM 区分 fully processed tokens（两个 expert 都输出完毕）和 partially processed tokens（只完成一个 expert），确保只有完整 token 的 hidden state 进入下一层。
    - **系统框架层（解决"FCFS + HOL blocking + run-to-completion"）**：① Scheduler 新增 Dispatcher（按 priority 分发到 4 个优先级队列：LS_PrefillQueue, LS_DecodeQueue, BE_PrefillQueue, BE_DecodeQueue）和 Batch Engine（Algorithm 1：LS_Decode > LS_Prefill + 填充 BE > BE_Decode > BE_Prefill）。② Closed-loop feedback controller：Inference Engine 在每个 attention 和 router stage 后回调 Scheduler。当 LS 到达时，Scheduler 立即发送 preempt 信号——无需等待 iteration boundary。③ Expert-level preemption：在 layer L 的 router 后，BE batch 暂停，其 partial state 通过 Sequence 对象的独立 tensor 原地保存。Engine 立即转向执行 LS prefill + decode，完成后动态将 LS 加入当前 batch，BE 从 preemption point 恢复（无 recomputation）。④ Facade Pattern Batch：对外呈现为单一 concat tensor，对内维护 per-sequence 独立 tensor——model 无感知 batch composition 变化，但系统可在任意时刻修改 individual sequence 状态。
    - **编译框架层**：论文未明确说明。QLLM 继承 HF TGI 的 PyTorch 执行路径，不涉及编译框架修改。
    - **kernel调度层**：论文未明确说明。preemption 的额外开销来自 per-sequence state tracking (routing_weights, hidden_states 的读写)，不涉及新 GPU kernel。
    - **硬件架构层**：同一 A100 80GB GPU。关键变化：per-expert queuing 将 "batch 必须同步跑完全部 layers" 解耦为 "各 expert 独立处理其队列中的 token"——在 A100 的 CUDA stream 层面，expert FFN kernel 仍是串行执行的（同一 GPU），但 LS token 优先被选中执行：当 Scheduler 在 layer 1 的 router 后 preempt BE、prefill LS、恢复 BE 时，GPU 上的实际计算序列变为 attention_l1 → router_l1 → (暂停 BE expert FFN) → LS prefill (all layers) → LS decode iteration → (恢复 BE expert FFN_l1) → attention_l2 → ...。preemption latency 远小于 300-400ms iteration 时间（因为只在单层内切换而非等待整轮 iteration）。
  - 解决 Baseline 缺陷的方式总结：
    1. **针对"priority-oblivious FCFS"**：将队列按 priority 分为 LS/BE × prefill/decode 四级，Batch Engine 严格优先 LS。LS 不因早到达的 BE 而被排队阻塞。
    2. **针对"iteration-level granularity (300-400ms)"**：Closed-loop feedback 在每层 attention/router 后给 Scheduler 控制权，preempt 可在任意 layer 触发——LS 到达后的等待时间从 300-400ms 降低到当前 layer 执行时间（~10ms for a single layer）。
    3. **针对"run-to-completion (无法中途插入)"**：Expert-level preemption + Sequence 对象独立状态保存——BE batch 在任意 layer 可暂停，LS 执行完毕后动态合并恢复，无需 recomputation。
    4. **针对"inner-layer state tracking 困难"**：Facade Pattern 的 Batch/Sequence 抽象——上层 model 看到的是 concat tensor（兼容现有代码），下层系统维护 per-sequence 独立 tensor，支持零拷贝 individual update。
