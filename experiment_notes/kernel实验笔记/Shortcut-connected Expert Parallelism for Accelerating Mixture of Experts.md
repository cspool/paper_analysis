## Shortcut-connected Expert Parallelism for Accelerating Mixture of Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：ScMoE的**自适应算子调度（Adaptive Operator Scheduling）**实现专家并行中的通信-计算重叠，核心调度机制：(1) **双流并行架构**：ScMoE将MoE操作从backbone网络完全解耦后，执行两条独立CUDA stream——Shared Expert stream（处理当前层表示的计算路径）和MoE stream（处理前一层表示的通信+计算路径）；(2) **自适应专家计算定位**：MoE stream中gate routing和encode算子调度到最早可行位置，decode算子延迟到最后位置以最大化重叠窗口，核心挑战是将expert computation插入Shared Expert stream的4个候选位置（①②③④）之一（见图6），目标函数为 min_K(|ΣCOMP_pre - T_disp| + |ΣCOMP_post - T_comb|)，其中T_disp和T_comb分别是All-to-All Dispatch/Combine的通信时间，根据实际模型和硬件配置的性能数据自适应选择最优位置；(3) **异步All-to-All通信**：使用异步All-to-All通信算子，在CUDA stream间实现通信与计算的并行；(4) **与pipeline的兼容组合**：当通信时间超出重叠窗口时，ScMoE策略可与传统pipeline策略叠加——先利用ScMoE的扩展窗口隐藏部分通信，剩余部分通过pipeline以fine-grained chunk隐藏。与传统pipeline对比：pipeline策略将tokens均匀切分为chunks并行处理但无法重叠首尾chunk的通信（受限于prologue/epilogue bubbles），ScMoE直接消除这些限制实现100%通信隐藏。
  - 实验比较：(a) Overhead breakdown：各MoE架构（Standard top-2/top-1 + pipeline、Shared-Expert、ScMoE）在三种硬件配置下的通信/计算时间分解，ScMoE在8×A30-PCIe重叠70%通信、8×A800-NVLink完全重叠、16×A800-NVLink（cross-node）完全重叠；(b) 加速对比：ScMoE在8×A30-PCIe vs pipelined standard top-2 MoE 提升42%、vs pipelined top-1 MoE 提升15%、vs Shared-Expert MoE 提升27%；(c) 端到端训练/推理speedup：各模型+各硬件配置下的wall-clock加速比。

- 后端平台是什么，配置是什么。
  - GPU：8×NVIDIA A30-PCIe（PCIe互联，高通信开销，All-to-All占总时间60%）；8×NVIDIA A800-NVLink（NVLink互联，低通信开销，All-to-All占15%）；16×NVIDIA A800-NVLink across 2 nodes（节点间Ethernet互联）。单卡A30-PCIe用于memory-limited inference实验。
  - CPU-GPU通信：PCIe 4.0（expert offloading场景）。
  - 软件：PyTorch + CUDA streams，Tutel MoE + Fairseq框架。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于Tutel MoE框架和Fairseq训练框架。修改内容：(1) 实现MoE模块的双CUDA stream架构——将shared expert计算放在主stream，gate-routed expert的通信+计算放在独立stream；(2) 实现自适应算子调度器——基于profiled T_disp/T_comb和T_comp数据，计算min_K目标函数选择最优expert computation位置，运行时动态配置；(3) 实现异步All-to-All通信接口（NCCL异步模式）；(4) 实现与Tutel pipeline策略的组合模式开关。评估方法：测量每个Block-MLP+Block-MoE对的实际wall-clock时间（通过CUDA event timing），计算speedup ratio和通信重叠比例。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未提供独立的ScMoE调度器开源代码仓库。基于Tutel (https://github.com/microsoft/Tutel) 实现。
  - 评估原理与调度全过程（ScMoE Pos-2, 8×A30-PCIe, training one iteration）：
    ```
    输入：microbatch tokens [B, S, d] 分布在8 GPUs (DP+EP混合)
    
    # === Block-MLP (前一层，GPU主Stream) ===
    ├─ [Kernel] MultiHead_MLP Forward: QKV projections + attention + output proj
    │   输入: H_{l-1} [B, S, d] → 输出: H_l^{MH} [B, S, d]
    └─ [Kernel] MLP Forward: gate_proj → activation → up_proj → down_proj
        输入: H_l^{MH} → 输出: H_l^{MLP} [B, S, d]
    
    # === 调度决策（CPU侧，基于profiled数据） ===
    读取已profile的: T_Atten, T_SE, T_MLP, T_disp, T_comb
    计算overlap_window = T_Atten + T_SE + T_MLP (Pos-2)
    计算4个候选位置K∈{1,2,3,4}的cost:
      cost(K) = |Σ_{i=1}^{K-1} COMP_i - T_disp| + |Σ_{i=K+1}^{4} COMP_i - T_comb|
    选择 argmin_K cost(K) → 设expert computation位置
    
    # === Block-MoE (当前层，双CUDA stream) ===
    
    # MoE Stream (独立stream，与主stream并行):
    ├─ [Kernel] Gate Routing: H_l^{MH} @ W_gate + noise → TopK softmax
    │   → top-1 expert index per token (可在Block-MLP执行时提前调度)
    ├─ [Kernel] Input Encode: 聚合token data到连续layout
    ├─ [Communication] Async All-to-All Dispatch: 将tokens发送到目标expert所在GPU
    │   (与主Stream的Attention + Shared Expert重叠)
    ├─ [Kernel] Expert Computation (在调度器选择的位置插入):
    │   expert FFN: gate_proj(H_l^{MH}) → SiLU ⊙ up_proj(H_l^{MH}) → down_proj
    ├─ [Communication] Async All-to-All Combine: 将expert输出发回原始GPU
    │   (与主Stream的后续计算重叠)
    └─ [Kernel] Output Decode: 恢复token原始顺序
    
    # Shared Expert Stream (主GPU stream):
    ├─ [Kernel] MultiHead_MoE Forward (与MoE stream的gate+encode+dispatch并行)
    │   输入: H_l^{MLP} → 输出: H_{l+1}^{MH}
    ├─ [Kernel] Shared Expert Forward (与MoE stream的expert comp+combine并行)
    │   SE^{(l+1)}(H_{l+1}^{MH}): gate_proj → SiLU → up_proj → down_proj
    └─ [Kernel] Merge: coef * se_out + gate_weight * expert_out + residual
    
    # 通信重叠效果（8×A30-PCIe, T_disp+T_comb ≈ 60% of MoE time）:
    # overlap_window ≈ T_Atten + T_SE + T_MLP ≈ 70% of total → 70%通信被隐藏
    # 剩余30%通信无法隐藏 → 可叠加pipeline策略进一步隐藏
    
    性能输出：
    ├─ per Block-MLP+Block-MoE pair wall-clock时间 (ms)
    ├─ 通信重叠比例 = (T_disp+T_comb - 实际暴露的通信时间) / (T_disp+T_comb)
    ├─ 端到端speedup = T_baseline / T_ScMoE
    └─ 各component时间分解 (CUDA profiler timeline)
    ```
  - 关键技术点：(a) 自适应调度关键：expert computation位置的选择直接影响重叠效果——选在T_comp_pre最接近T_disp的位置使dispatch通信几乎完全隐藏，选在T_comp_post最接近T_comb的位置使combine通信同理；(b) ScMoE在通信时间≤约50%总MoE时间时可实现完全重叠（公式上下界保证）；(c) 与pipeline对比的核心优势：pipeline受限于prologue/epilogue不能被重叠的首尾传输，ScMoE无此限制。
