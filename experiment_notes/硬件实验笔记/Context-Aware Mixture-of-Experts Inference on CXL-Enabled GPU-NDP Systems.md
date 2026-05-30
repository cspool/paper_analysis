## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- 属于硬件架构的实现是什么？实验比较什么？
  实现是 **CXL-attached NDP (Near-Data Processing) device 模拟与 GPU-NDP 异构系统架构**。核心硬件架构：1× H100 GPU + 1× DDR-based CXL-NDP device 通过 PCIe Gen4 ×16 互联。NDP device 配备 64×(4×4) systolic arrays @ 1 GHz 作为计算单元，512 GB DDR 内存提供 512 GB/s 内部带宽。系统将 MoE experts 分为 GPU-resident hot experts（FP16 全精度）和 NDP-resident cold experts（1-4 bit 量化），利用 NDP 的近数据执行能力将传统的 "Parameter Movement"（移动 expert 权重）转化为 "Activation Movement"（仅移动激活值），通过 CXL 协议实现 GPU↔NDP 的 activation 传输。NDP 计算单元受限于 tight power/area budget，低精度量化是使 NDP 不成为瓶颈的关键使能技术。

  实验比较：
  - vs MoNDE [18] (相同 GPU-NDP 硬件配置，但 context-agnostic expert placement)：Ours-3bit 6.6-8.3× 端到端加速，Ours-2bit 7.9-10.6×
  - vs HOBBIT [31] (GPU-only 混合精度 offloading)：Ours-2bit 达 18-19× speedup
  - NDP 侧 latency：Ours-3bit ~5× reduction, Ours-2bit ~8× reduction (量化降低 NDP compute pressure)
  - Mixtral-8×7B: 每层 4 GPU experts + 4 NDP experts；Mixtral-8×22B: 每层 2 GPU experts + 6 NDP experts

- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  **Ramulator** [19]：CMU SAFARI 组的快速可扩展 DRAM 模拟器。GitHub: https://github.com/CMU-SAFARI/ramulator。论文基于 Ramulator 构建 NDP system simulator，采用与 MoNDE [18] 相同的方法学。

- 模拟器模拟什么的性能，修改了什么。
  模拟 GPU-NDP 异构系统的端到端推理性能，包括：
  - GPU 侧：H100 的 expert FFN 计算延迟（FP16 GEMM on tensor cores）
  - NDP 侧：DDR-based NDP 设备的 expert 计算延迟（量化精度下的 systolic array 矩阵乘法）
  - 互联：PCIe Gen4 ×16 的 GPU↔NDP activation/weight 传输延迟
  - 量化开销：GPTQ 量化/反量化延迟、不同 bitwidth (1/2/3/4 bit) 对 NDP 计算 throughput 的影响
  
  修改内容：
  1. **NDP 设备建模**：在 Ramulator 上增加 DDR-based NDP compute unit 模拟——64×(4×4) systolic arrays @ 1 GHz，建模量化精度下的计算吞吐（4-bit 计算 vs 1-bit 计算的 throughput 差异）
  2. **CXL 互联建模**：GPU↔NDP 通过 PCIe Gen4 ×16 的 activation movement 延迟建模，对比 parameter movement 的带宽消耗
  3. **Expert Placement 调度模拟**：prefill-stage statistics collection + single migration + decoding with fixed placement 的完整时序模拟
  4. **Multi-precision NDP execution**：模拟 NDP 在混合精度（不同 expert 不同 bitwidth）下的 per-expert 计算延迟和 pipeline overlap

- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？至少具体到模拟器模拟性能的原理和模拟器输入到性能输出的全过程。
  论文未公开代码。Ramulator 开源: https://github.com/CMU-SAFARI/ramulator。NDP 扩展代码论文未明确说明是否公开。

  **GPU-NDP 系统模拟器工作原理与执行流程**：

  1. **输入配置**：
     - MoE 模型定义：层数 L、expert 数 E、hidden dim、FFN dim（如 Mixtral-8×7B: L=32, E=8, d=4096, d_ff=14336）
     - GPU 参数：H100 SMs=132, peak TFLOP/s=989.4, HBM=80GB, HBM BW
     - NDP 参数：systolic arrays=64×(4×4), clock=1 GHz, DDR BW=512 GB/s, capacity=512 GB
     - 互联：PCIe Gen4 ×16 BW
     - Bitwidth 配置：GPU experts FP16, NDP experts 1/2/3/4 bit
     - Workload：输入/输出 token 长度

  2. **Prefill 阶段模拟**：
     - Attention: FlashAttention latency model on H100 (基于 seq_len 和 batch size)
     - Router: 轻量级 Linear-Softmax-TopK 延迟（GPU 原生，可忽略）
     - Expert FFN (all on GPU during prefill): GEMM latency = (M×K×N × 2) / GPU_TFLOPs × efficiency_factor
     - 统计收集：counter update latency (negligible)

  3. **Expert Placement 决策模拟**：
     - 输入：(P_{l,e}, W_{l,e}) → S_{l,e} → top-K selection → 生成 GPU expert set H_l 和 NDP expert set C_l
     - Expert 迁移延迟：若 expert 需跨设备迁移，延迟 = expert_weight_size / PCIe_BW
     - 量化权重加载：NDP 从预缓存的 1/2/3/4-bit replicas 中选择对应 bitwidth 版本

  4. **Decoding 阶段模拟（per-step latency accumulation）**：
     ```
     for each decoding step:
         for each layer l:
             # GPU 部分
             gpu_time = attention_latency(seq_len) + router_latency
             
             # 检查选中 experts 的设备分布
             for e in selected_experts:
                 if e in H_l:  # GPU expert
                     gpu_time += gemm_latency_fp16(tokens, d, d_ff)
                 else:  # NDP expert
                     # Activation to NDP (parameter movement → activation movement)
                     pcie_tx += activation_size / PCIe_BW
                     # NDP systolic array computation
                     ndp_time += systolic_gemm_latency(tokens, d, d_ff, bits=b_{l,e})
                     # Result back to GPU
                     pcie_rx += activation_size / PCIe_BW
             
             # GPU 和 NDP 重叠执行
             layer_latency = max(gpu_time, pcie_tx + ndp_time + pcie_rx)
         
         step_latency = sum(layer_latency for all layers) + lm_head_latency
     ```

  5. **NDP Systolic Array 性能建模**：
     - 4×(4×4) = 16 MAC/array, 64 arrays = 1024 MACs/cycle @ 1 GHz = 1 TOPS (INT8 equivalent)
     - 不同 bitwidth 的有效吞吐：
       - 4-bit: weight/activation 各 4b → 接近 INT4 吞吐 ≈ 4× INT8 throughput（因位宽减半）
       - 1-bit: 二进制权重 → 计算变为 XNOR + popcount → throughput 进一步提升
     - GEMM latency = (M×K×N × bits_factor) / NDP_effective_TOPS

  6. **性能输出**：
     - End-to-end latency = prefill_latency + num_decoding_steps × step_latency
     - Decoding throughput = num_decoding_steps / total_decoding_time (tokens/s)
     - NDP-side latency breakdown：expert computation vs activation transfer
     - Speedup vs baseline = baseline_latency / proposed_latency

  7. **关键架构参数对性能的影响**：
     - NDP compute capacity (64×(4×4) arrays) + low-bitwidth quantization → NDP 从瓶颈变为可支撑组件
     - 512 GB/s NDP 内部带宽 >> PCIe Gen4 ×16 (~32 GB/s) → NDP 近数据执行的带宽优势
     - GPU HBM 80GB 容量 → 决定 K（GPU-resident experts/layer）的上限
