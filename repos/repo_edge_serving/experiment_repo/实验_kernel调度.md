## TZ-LLM

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：**TEE 内 NPU 数据平面驱动（Co-driver Design，§4.3）** 和 **动态安全内存管理（§4.2）**——在 TEE 中实现轻量级 NPU kernel 执行调度和安全内存运行时管理：
    1. **TEE NPU 数据平面驱动（~1K LoC user-mode）**：从 Rockchip NPU Linux 驱动（~60K LoC）中提取 NPU job 执行的最小闭包（数据平面）部署在 TEE 用户态，包括：(a) 初始化 job 执行上下文（I/O page table、register commands、input/output buffers，均位于安全内存）；(b) MMIO 操作启动 NPU job；(c) 安全中断处理（job completion interrupt）。控制平面（job 调度、电源管理、频率控制、Linux 设备框架依赖）保留在 REE 中。TEE OS 通过两个限制约束用户态 NPU 驱动：(1) 仅映射 NPU MMIO 区域到驱动地址空间，禁止访问其他安全设备；(2) 仅允许 NPU DMA 访问特定 TZASC 区域（job 执行上下文），禁止访问其他安全内存。
    2. **NPU 安全模式切换（§4.3）**：TEE 驱动按严格顺序执行 NPU 世界切换：(1) 配置 TZPC 隔离 NPU MMIO 从 REE；(2) 等待进行中的非安全 NPU job 完成；(3) 配置 TZASC 授权 NPU 访问安全内存区域。归还 NPU 时逆向操作。切换开销（TZASC+TZPC+GIC 配置+smc）占 TTFT 的 1.6%∼2.7% 和 decoding 时间的 2.3%∼5.7%。
    3. **动态 CMA 安全内存管理（§4.2）**：设计 "extend and shrink" 安全内存管理接口。`extend_allocated` → TEE OS 委托 REE TZ driver 从 Linux CMA 分配与已分配块相邻的物理连续内存 → verify contiguity → `extend_protected` → TZASC 扩展保护区域末端 → 映射到 TA 地址空间。`shrink` 从 TZASC 区域末端释放内存。CMA allocation 使用多线程可达 3.8 GB/s（4 threads），在 I/O 延迟下可隐藏分配开销。
    4. **安全中断路由（§3.2, §4.3）**：配置 GIC 将 NPU 中断从 REE 重新路由到 TEE OS，使 TEE NPU 驱动能直接接收 completion 中断，无需经过 REE 转发。
  - 实验比较：
    - NPU 分时复用场景下，TEE-REE NPU 分时 vs REE 内 NPU 分时的额外开销：NN 应用（YOLOv5、MobileNet）吞吐额外降低 ≤3.8%，LLM 吞吐额外降低 ≤3.0%
    - CMA allocation 对 REE 应用的干扰：Geekbench 分数降低 ≤6.7%（仅 prefill 阶段，解码阶段无影响）
    - CMA allocation 延迟 vs buddy system 分配延迟、vs S2PT 方案的性能对比

- 后端平台是什么，配置是什么。
  - 后端平台：**Rockchip RK3588 SoC**（Orange Pi 5 Plus）
    - CPU：4× Cortex-A76 @ 2.4GHz（big）+ 4× Cortex-A55 @ 1.8GHz（LITTLE），Armv8.2-A
    - NPU：3 核，最高 6 TOPS INT8 算力。配备 Rockchip NPU driver v0.9.8（https://github.com/airockchip/rknn-llm/tree/main/rknpu-driver）
    - 内存：16 GB LPDDR4X
    - 存储：1 TB NVMe SSD（PCIe 3.0 x4，顺序读 ~2 GB/s）
    - TrustZone 硬件：Arm TrustZone（TZASC-400、TZPC、GIC 安全扩展）
  - 软件栈：
    - TEE OS：OpenHarmony TEE（基础功能 17K LoC）
    - REE OS：OpenHarmony v4.1 + Linux v5.10
    - 推理框架：llama.cpp（作为 TA 运行在 TEE 中）
    - 加密：OpenSSL AES 解密

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件/脚本：
    1. `scripts/1-end-to-end-prefill.sh`：评估 TZ-LLM 及 baselines 在不同 benchmarks 下的 TTFT（对应论文 Figure 10）。约 60 compute-minutes。
    2. `scripts/2-end-to-end-decoding.sh`：评估 TZ-LLM 及 baselines 在不同模型下的 decoding speed（对应论文 Figure 11）。约 20 compute-minutes。
    3. `scripts/3-caching.sh`：评估 partial parameter caching 在不同 cache 比例和 prompt 长度下对 TTFT 的影响（对应论文 Figure 14）。约 60 compute-minutes。
    4. NPU 分时复用评估（论文 §7.3）：并发运行 YOLOv5（目标检测）和 MobileNet（图像分类）与 LLM 推理，测量 NPU 共享下的吞吐量。
    5. CMA 干扰评估（论文 §7.4）：并发运行 Geekbench 与 LLM prefill，测量 CMA allocation 对 REE 应用的性能影响。
    - 辅助工具：stress-ng（模拟 REE 内存压力，触发 CMA page migration）、Python3 + matplotlib（结果绘图和分析）
  - 修改内容：
    - **TEE OS**：+62 LoC（CMA page 内存映射管理）+ 50 LoC（动态 TZASC/TZPC 配置）
    - **REE Linux 内核**：+167 LoC（NPU 驱动 shadow job 调度）+ 197 LoC（TZ driver CMA allocation/deallocation）
    - **llama.cpp TA**：+1.2K LoC（流水线恢复）+ 1K LoC（NPU 数据平面驱动集成）
    - **NPU 驱动**：从 Rockchip NPU 驱动中提取数据平面闭包（job setup → MMIO launch → completion 处理），去除控制平面依赖（Linux device/memory/interrupt/power management 子系统）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源状态：**已开源**。Artifact DOI：https://doi.org/10.5281/zenodo.17213486。需 Orange Pi 5 Plus（RK3588）+ OpenHarmony 构建环境（Docker）+ USB-to-USB 连接线。
  - 全过程（以安全 NPU job 的 kernel 执行路径为例，Llama-3-8B 矩阵乘法算子）：
    1. **Job 初始化（TEE 侧）**：LLM TA 需要执行 attention 或 FFN 中的矩阵乘法。TEE NPU 驱动在安全内存中分配 job 执行上下文：(a) I/O page table——定义 NPU 的 DMA 地址映射，指向安全内存中的 weight buffer（已解密参数）和 activation buffer；(b) register commands——NPU 指令序列，描述矩阵乘法维度、数据布局（NHWC/NCHW）、量化参数；(c) input buffer——INT8 量化后的 activation 张量；(d) output buffer——预分配的输出张量内存。
    2. **Job 提交（smc 跨世界调用）**：TEE 驱动通过 smc 向 REE NPU 驱动提交一个 shadow job（仅含元数据、空执行上下文）。shadow job 被排入 REE 驱动的统一调度队列（与 YOLOv5、MobileNet 等 REE NPU jobs 混合调度）。TEE 驱动为每个 job 分配单调递增序列号防重放/重排序攻击。
    3. **NPU 控制权切换（REE→TEE）**：当 shadow job 被调度到：(a) REE 驱动停止调度新的非安全 job；(b) 通过 smc 通知 TEE 驱动接管 NPU；(c) TEE 驱动验证 shadow job 序列号和 job 状态（防重放/未授权启动）；(d) 配置 TZPC 寄存器——将 NPU MMIO 区域标记为安全设备，禁止 REE CPU 访问（~μs 级）；(e) 轮询 NPU 状态寄存器确认无进行中的非安全 job（或等待其完成）；(f) 配置 TZASC——将 job 执行上下文所在的 TZASC region 的 DMA 权限位设为允许 NPU 访问；(g) 配置 GIC——将 NPU 中断路由从 REE 切换到 TEE。
    4. **MMIO 启动 NPU job**：TEE 驱动通过 MMIO 写入 NPU 控制寄存器：(a) 写入 I/O page table 基地址；(b) 写入 register commands 基地址和长度；(c) 写入 input/output buffer 地址；(d) 写启动寄存器触发 NPU 开始执行。NPU 三核并行处理 INT8 矩阵乘法（每核独立从安全内存 DMA 读取权重和 activation → 乘加 → 写回安全内存）。
    5. **中断处理（NPU completion → TEE）**：NPU 完成 job 后触发硬件中断 → GIC 路由到 TEE OS 中断处理器 → TEE NPU 驱动中断处理函数被调用 → 读取 NPU 状态寄存器确认 job 成功完成 → 配置 TZASC 撤销 NPU 对安全内存的 DMA 权限 → 配置 GIC 将 NPU 中断路由回 REE → 配置 TZPC 将 NPU MMIO 恢复为非安全设备 → 通过 smc 通知 REE 驱动 shadow job 完成。
    6. **性能输出**：TEE 驱动记录每个 NPU job 的总耗时（初始化 + smc 通信 + TZASC/TZPC/GIC 配置 + MMIO 启动 + NPU 执行 + 中断处理 + 归还）。论文测量 NPU 分时复用总开销占 TTFT 的 1.6%∼2.7%（prefill，长 NPU 计算窗口摊薄）和 decoding 的 2.3%∼5.7%（解码单 batch 计算短，切换开销占比更高）。


## IntAttention Fully Integer Attention Pipeline for Edge LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于 Arm Compute Library (ACL) 的 IndexSoftmax 整数 kernel，替代 ARM CPU 上注意力 pipeline 中的 softmax 路径（dequantize → softmax → requantize）。核心 kernel 实现：
    1. **Clipping kernel**：对 int32 logits 做 max-subtraction + 裁剪，仅保留 c=6.6 范围内的有效 logits，利用 ARM NEON SIMD 做并行 max/clamp。
    2. **LUT Exponential kernel**：32-entry UINT8 LUT 完全驻留在 NEON 寄存器中，对裁剪后 logits 做并行查表（每个 lane 独立映射 logit → LUT index → UINT8 prob）。使用定点位移将 int32 logit 映射到 [0,31] 索引。
    3. **Integer Normalization kernel**：对每行的 UINT8 概率做整数归一化。使用 ACL 的 NEReduceMean + NEArithmeticOps 实现行内求和与除法。UINT8（256 个值）相较 INT8（128 个值）在相同 32B 预算下提供 4× 分辨率。
  - 对比对象与量化结果：
    - Softmax 路径在全 attention latency 中的占比：FP32 下为次要开销；FP16 下 softmax+cast 更明显；**INT8 下 dequantize→softmax→requantize 占 ≤65% attention latency**。
    - Kernel 速度结果：
      - RK3588S2（ARM Cortex-A76+A55，Armv8.2-A）：IntAttention vs FP16 attention **3.7× 加速**，vs INT8 Quant-Only（仅量化 GEMM、softmax 仍浮点）**2.0× 加速**。
      - Apple M2（ARM-based Apple Silicon）：IntAttention vs FP16 **2.8× 加速**，vs Quant-Only **2.4× 加速**。
    - Kernel 能量结果（RK3588S2 USB 功率计实测）：IntAttention energy vs FP16 **降低 61%**，vs INT8 Quant-Only **降低 37%**。

- 后端平台是什么，配置是什么。
  - 后端 CPU 平台：
    - RK3588S2 嵌入式开发板：ARM Cortex-A76（big core）+ Cortex-A55（LITTLE core），Armv8.2-A 架构。论文未明确说明核心数/频率，典型配置为 4×A76@2.4GHz + 4×A55@1.8GHz。
    - Apple M2 笔记本：Apple Silicon SoC，ARM ISA。论文未给出具体核心数/频率。
  - 硬件扩展：RK3588S2 内置 NPU（6 TOPS INT8），但论文**未使用 NPU**——所有实验在 ARM CPU 上运行。
  - 软件栈：
    - Arm Compute Library (ACL)：提供 INT8 GEMM kernel（NEMMConvLayer、NEGEMM）、reduce、arithmetic 等算子。论文按 https://github.com/WanliZhong/IntAttention 的 `add_impl_for_ACL.patch` 打补丁，在 ACL 中添加 IndexSoftmax 实现。
    - 编译工具链：clang++ + scons（ACL 构建系统）
    - OS：论文未明确说明（RK3588S2 典型运行 Linux/Android）

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件/脚本：
    1. `bench_speed.cpp`：C++ 延迟 benchmark。支持四种 pipeline（pipe 0=FP32, pipe 1=FP16, pipe 2=INT8 Quant-Only, pipe 3=IntAttention）。参数：`--L`（序列长度）、`--d`（head dim）、`--warmup`、`--runs`。编译时链接 patched ACL 静态库。
    2. `acc_llm.py`：Python 精度评估，基于 lm_eval harness。支持 `--model-name`（llama-3.2-1b/opt-1.3b/qwen3-1.7b）、`--method int_attention`。
    3. `acc_deit.py`：Python 视觉精度评估，基于 timm 的 DeiT/ViT/CaiT 模型，支持 `--model deit_base_patch16_224` + `--method int_attention`。
    4. `power_traces/`：USB 功率计导出的 CSV 能耗 raw data。
    5. `pysimulation/`：PyTorch 模拟代码，用于在 GPU 上模拟 IndexSoftmax 的整数行为（因 GPU 原生 INT32 矩阵乘支持有限，使用高精度浮点算术模拟）。
  - 对 ACL 的修改（`add_impl_for_ACL.patch`）：
    1. 新增 `NEIndexSoftmax` kernel：实现 `S32 → U8` 的 IndexSoftmax 全整数概率路径，替代原有的 `dequantize(INT8→FP32) → NEExpLayer → NEPoolingLayer(FP32 sum+div) → requantize(FP32→INT8)` 流程。
    2. 裁剪逻辑使用 NEElementwiseMax + NEComparison 找到有效 logits mask，仅对有效 logits 走 LUT + normalization。
    3. LUT 存储在 NEON 寄存器中（32×UINT8 = 32B），查表通过 NEON TBL 指令并行执行。
    4. 归一化使用定点乘加（而非浮点除法）：`prob_norm[i] = (prob[i] * 255 + sum_prob/2) / sum_prob`。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源状态：**已开源** https://github.com/WanliZhong/IntAttention。MLSys 2026 artifact evaluation 通过。
  - 评估原理与全过程（以 RK3588S2 上 benchmark IntAttention vs FP16 延迟为例）：

    ```
    === Step 1：环境准备（ArtifactEvaluation/ 目录） ===
    1. git clone --recurse-submodules https://github.com/WanliZhong/IntAttention
    2. cd IntAttention/ArtifactEvaluation
    3. 对 ACL 打补丁：
       cd ComputeLibrary && git apply ../add_impl_for_ACL.patch
    4. 构建 ACL：
       scons arch=arm64-v8.2-a neon=1 opencl=0 examples=0 \
             Werror=0 debug=0 asserts=0 standalone=1 \
             build_dir=build/int_attention \
             extra_cxx_flags="-march=armv8.2-a+fp16+dotprod"
    5. 编译 benchmark：
       clang++ -O3 -march=armv8.2-a+fp16+dotprod \
               -I ComputeLibrary/include -I ComputeLibrary \
               bench_speed.cpp \
               -L ComputeLibrary/build/int_attention \
               -larm_compute-static -larm_compute_core-static \
               -lpthread -o bench_speed_intattn

    === Step 2：运行延迟 Benchmark ===
    6. ./bench_speed_intattn --pipe 0 --L 1024 --d 128 --warmup 10 --runs 100
       # pipe 0 = FP32 baseline
    7. ./bench_speed_intattn --pipe 1 --L 1024 --d 128 --warmup 10 --runs 100
       # pipe 1 = FP16 baseline
    8. ./bench_speed_intattn --pipe 2 --L 1024 --d 128 --warmup 10 --runs 100
       # pipe 2 = INT8 Quant-Only (INT8 GEMM but FP32 softmax)
    9. ./bench_speed_intattn --pipe 3 --L 1024 --d 128 --warmup 10 --runs 100
       # pipe 3 = IntAttention (INT8 GEMM + IndexSoftmax)

    === 评估原理（bench_speed.cpp 内部） ===
    对每种 pipe，benchmark 一次 attention 的端到端延迟：
      - 分配 Q/K/V 输入张量（根据精度：FP32/FP16/INT8）
      - 如 pipe ≥ 2：Q/K/V 先量化至 INT8
      - warmup 轮：执行 attention 但不计时（预热 cache/TLB/分支预测）
      - 测量轮：std::chrono::high_resolution_clock 计时：
        a) QK^T GEMM（ACL NEGEMM，S8×S8→S32）
        b) Softmax 路径：
           pipe 0/1：dequant→FP32/FP16 exp+sum+div→requant
           pipe 2：   dequant→FP32 exp+sum+div→requant（与 FP 相同）
           pipe 3：   NEIndexSoftmax（clipping→LUT→integer norm, S32→U8）
        c) PV GEMM（ACL NEGEMM，U8×S8→S32 for pipe 3, S8×S8→S32 for pipe 2）
      - 输出：avg/median/min/max latency（μs）、GFLOPS

    === Step 3：运行精度 Benchmark（GPU 上） ===
    10. python acc_llm.py --model-name llama-3.2-1b --method int_attention
        # 加载 LLaMA-3.2-1B from HuggingFace
        # 替换每层 attention 的 softmax 为 IndexSoftmax（PyTorch 模拟）
        # 在 lm_eval 各 benchmark 上评估 WikiText/HellaSwag/LAMBADA 等

    === Step 4：能量测量（独立） ===
    11. USB 功率计串联在 RK3588S2 电源线上
        while 运行 step 6-9 各 1000 iterations：
          记录功率计 CSV (time, voltage_V, current_A, power_W)
        energy_mJ = mean(power_W) × latency_us / 1000
        输出 power_traces/*.csv
    ```

  - Kernel 输入到性能输出全过程（以 IntAttention pipe 3 一次 attention 为例）：
    ```
    输入：INT8 Q, K, V ∈ Z^{L×d}（S8，通过 ACL QuantizationLayer 量化得到）
    输出：INT32 O ∈ Z^{L×d}（S32 累加器）

    1. ACL NEGEMM::run(Q, K, S_buf)          → S32 logits buffer
    2. NEIndexSoftmax::run(S_buf, P_buf)      → UINT8 probability buffer
       ├── NEArithmeticOps::sub_max(S_buf)     // S_i -= max(S_row)
       ├── NEComparison::gt(S_buf, -c)         // mask = S > -c_thresh
       ├── NEArithmeticOps::apply_mask()       // 仅保留有效 logits
       ├── NEIndexSoftmax::lut_lookup()        // NEON TBL 并行查 LUT
       └── NEIndexSoftmax::integer_normalize() // 行内定点除
    3. ACL NEGEMM::run(P, V, O_buf)           → S32 output buffer

    关键性能指标：
    - softmax_ratio = t_softmax / t_total_attention  (≤65% for INT8 Quant-Only)
    - IntAttention 将 t_softmax 部分降至最低（纯整数，无类型转换）
    ```

## Scaling LLM Test-Time Compute with Mobile NPU on Smartphones

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在 Qualcomm Hexagon NPU 上实现了一套自定义 kernel，支持 LLM 推理中的 test-time scaling（并行采样/Beam Search）工作负载。核心 kernel 实现包括：
    1. **Mixed-Precision Dequantization GEMM Kernel**：W4A16 细粒度分组量化下的 INT4→FP16 动态反量化 GEMM kernel。包含三个优化层次：
       - **硬件感知 Tile 量化布局**：在量化前将权重预先排列为 HMX tile 布局（外层级 column-major tile 排序 + 每 tile 内每两行交错），使得反量化后的 FP16 权重可连续写入 TCM，消除 scatter 操作。
       - **量化组合并（Group Coalesce）**：将 8 个量化组（每组 32 元素）合并为一个 super-group，使 256 元素的 INT4 值精确填充一个 128-byte HVX 向量寄存器，最大化向量单元利用率。
       - **LUT 中心化反量化**：使用 HVX 的 `vlut16` 指令通过查表将 INT4 值直接转换为 FP16 值（含 qfloat 格式），避免传统的 unmask→unpack→convert 指令序列。Scale 广播也通过 vlut16 指令实现。
    2. **LUT-Based FP16 FlashAttention Kernel**：在 Hexagon NPU 上实现 FP16 FlashAttention（Algorithm 1），使用 HMX 做矩阵乘法、HVX 做 Softmax。核心创新：
       - **LUT-Based Fast Softmax**：利用 Safe Softmax 保证 exp 输入 ≤0，预计算 32768-entry FP16 LUT（64 KiB，存储在 TCM 中），使用 HVX 的 `vgather` 指令（单指令 gather 64 个 2-byte 元素）替换显式 exp 多项式计算。prefill 时 Softmax 加速 1.26–2.19× vs FP32 exp，up to 1.60× vs FP16 polynomial exp。
    3. **CPU-NPU 共享内存通信**：基于 FastRPC + rpcmem shared memory 实现 CPU 与 NPU 间的低延迟通信。NPU 侧通过轮询共享内存区域接收计算请求，消除默认 RPC 的额外延迟。
  - 实验比较：
    - Operator-level（GEMM dequantization）：Baseline（列主序量化 + scatter 写 TCM）vs HMX Layout vs HMX Layout + Group Coalesce（Ours）vs No Dequantization（上界）。Ours vs Baseline 加速 9.65–19.04×；Ours 仅比 No Dequantization 慢 27% on average。
    - Operator-level（Softmax）：FP32 polynomial exp vs FP16 polynomial exp vs LUT-based exp。LUT-based 加速 1.26–2.19× vs FP32，up to 1.60× vs FP16。
    - System-level（Decode throughput）：不同 batch size（1/2/4/8/16）、不同模型（Qwen2.5-1.5B/3B, Llama3.2-1B/3B）、不同设备（OnePlus Ace3/12/Ace5 Pro）下的端到端解码吞吐量。
    - System-level（Accuracy-Cost Trade-off）：Best-of-N 和 Beam Search 下 MATH500/GSM8K 的 pass@1 精度 vs 解码延迟。
    - Comparison：vs GPU-based llama.cpp OpenCL backend 和 QNN FP16（reference）。

- 后端平台是什么，配置是什么。
  - 后端 NPU 平台：Qualcomm Hexagon NPU，覆盖三代架构：
    | 设备 | SoC | NPU 架构 |
    |------|-----|----------|
    | OnePlus Ace3 | Snapdragon 8 Gen 2 | V73 |
    | OnePlus 12 | Snapdragon 8 Gen 3 | V75 |
    | OnePlus Ace5 Pro | Snapdragon 8 Elite | V79 |
  - NPU 内部硬件：
    - **HMX (Hexagon Matrix eXtension)**：FP16 tile 为 32×32（2 KiB/tile），支持 INT4/INT8/INT16/FP16。每 tile 支持 per-channel scale & bias。V73 上 FP16 GEMM 吞吐量 ~12 TFLOPS（实测）。
    - **HVX (Hexagon Vector eXtension)**：32 个 1024-bit 向量寄存器，4-6 个 HVX 单元/thread。V75 上单 thread FP16 GEMM 吞吐量 ~32.9 GFLOPS。支持 vgather、vlut16 等 SIMD 指令。
    - **Memory Subsystem**：1 MiB L2 cache + 8 MiB TCM（软件管理片上内存）。HVX scatter/gather 和所有 HMX 指令仅能访问 TCM。DMA 引擎提供 >60 GB/s DDR 读取带宽（vs HVX ~26 GB/s）。
  - 软件工具链：
    - Hexagon SDK v6.0.0.2（LLVM toolchain for Hexagon DSP）
    - **不依赖 QNN**：通过逆向工程 QNN 二进制库中的未公开 HMX 指令实现 FP16 矩阵单元编程
    - 基于 llama.cpp（NPU backend ~7K lines C/C++ + inline assembly）
  - 额外 GPU（仅精度评估）：NVIDIA RTX3090

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：修改版 llama.cpp（https://github.com/haozixu/llama.cpp-npu）+ 独立算子库 htp-ops-lib（https://github.com/haozixu/htp-ops-lib）。
  - 修改内容：
    1. **HTP backend（~7K lines）**：为 llama.cpp 添加 Hexagon NPU backend（`ggml-htp`）。包括 NPU 算子库（GEMM、FlashAttention、LayerNorm、RoPE、激活函数等）、电源管理、硬件资源管理、计算线程池。
    2. **HTP-Ops-Lib（C 92.5% + C++ 6.5%）**：独立 Hexagon DSP 共享对象。编译为 Stub（AArch64 CPU 侧，`libhtp_ops.so`）和 Skeleton（Hexagon DSP 侧，`libhtp_ops_skel.so`）。包含：
       - `flash_attn.c`：FP16 FlashAttention（Algorithm 1），含 LUT-based Softmax
       - Dequantization GEMM kernel：支持 Q4_0/IQ4_NL/Q8_0 量化格式的混合精度 GEMM
       - HVX 算子：LayerNorm、RoPE、激活函数等的向量化实现
    3. **权重转换脚本**：`extras/convert_hf_to_gguf_htp.py`，将 HuggingFace 权重转换为 HMX tile 布局的 GGUF 格式。
    4. **量化支持**：`llama-quantize` 添加 `REPACK_FOR_HVX=1` 环境变量，触发 super-group 合并的重量排。
    5. **通信层**：基于 FastRPC + rpcmem shared memory 的 CPU-NPU 通信，NPU 侧轮询接收计算请求。
    6. **精度评估**：使用 lm-eval-harness 评估 MATH500、GSM8K、MMLU、WinoGrande、WikiText-2。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源状态：**已开源**。主仓库 https://github.com/haozixu/llama.cpp-npu（MIT License）+ 算子库 https://github.com/haozixu/htp-ops-lib。EUROSYS '26 论文。
  - 评估原理与全过程（以 OnePlus 12 上评估 dequantization GEMM kernel 延迟为例）：

    ```
    === Step 0：环境准备 ===
    0a. 设备要求：Android 手机（Snapdragon 8 Gen 2+），root 权限推荐
    0b. 软件依赖：Android NDK + Hexagon SDK 6.x (v6.0.0.2) + CMake + Python
    0c. Hexagon SDK 环境：
        source <hexagon_sdk_root>/setup_sdk_env.source

    === Step 1：构建 NPU 算子库（htp-ops-lib） ===
    1. git clone https://github.com/haozixu/htp-ops-lib
    2. cd htp-ops-lib
    3. build_cmake android                    # → libhtp_ops.so (AArch64 stub)
    4. build_cmake hexagon DSP_ARCH=v75       # → libhtp_ops_skel.so (DSP skeleton)

    === Step 2：构建 llama.cpp NPU backend ===
    5. git clone https://github.com/haozixu/llama.cpp-npu
    6. cd llama.cpp-npu
    7. mkdir build && cd build
    8. cmake .. \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 \
         -DGGML_HTP=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF
    9. make -j llama-cli llama-quantize

    === Step 3：模型转换与量化 ===
    10. python extras/convert_hf_to_gguf_htp.py \
          --outfile qwen2.5-1.5b.f16-hmx.gguf --outtype f16 \
          $path_to_hf_model
        # 权重已排列为 HMX tile 布局

    11. REPACK_FOR_HVX=1 ./build/bin/llama-quantize \
          qwen2.5-1.5b.f16-hmx.gguf \
          qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf IQ4_NL+Q8_0
        # IQ4_NL: 4-bit 量化 (attention proj + FFN up/gate)
        # Q8_0: 8-bit 量化 (FFN down, 保留精度)
        # REPACK_FOR_HVX=1: 触发 super-group coalesce

    === Step 4：部署到设备 ===
    12. adb push build/bin/llama-cli /data/local/tmp/llama.cpp/
    13. adb push build/bin/lib*.so /data/local/tmp/llama.cpp/
    14. adb push libhtp_ops.so libhtp_ops_skel.so /data/local/tmp/llama.cpp/
    15. adb push qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf /data/local/tmp/llama.cpp/

    === Step 5：运行推理 Benchmark ===
    16. adb shell
    17. export LD_LIBRARY_PATH=/data/local/tmp/llama.cpp:/vendor/lib64:/system/lib64
    18. export DSP_LIBRARY_PATH=/data/local/tmp/llama.cpp
    19. ./llama-cli -t 4 -fa \
          -m qwen2.5-1.5b.iq4_nl+q8_0-hmx.gguf \
          -p "Solve: If x^2 + 5x + 6 = 0, find x." \
          -n 256 --batch-size 4
        # -fa: 激活 FlashAttention (NPU 侧 FP16 HMX FlashAttention)
        # --batch-size: test-time scaling 的 parallel sampling batch size
        # 通过 -n 控制生成 token 数 → 测量 decode 延迟和吞吐

    === 评估原理（GEMM Dequantization Kernel 内部） ===
    一次 dequantization GEMM（W4A16，shape=[hidden_dim, proj_dim]）的执行流程：

    输入：
      - INT4 weights (Q4_0): 每 32 个 4-bit 值 + 1 个 FP16 scale
      - FP16 activations: [batch_size, hidden_dim]
      - weights 已预先排列为 HMX tile 布局 + super-group coalesce

    执行阶段（在 Hexagon NPU 上）：

    1. DMA 预取（CPU 侧 rpcmem shared memory → NPU TCM）：
       - DMA 引擎从 DDR 加载 INT4 weights + FP16 scales 到 TCM
       - DMA 加载 FP16 activations 到 TCM
       - LUT table (64 KiB, 用于 dequantization) 常驻 TCM

    2. HVX Dequantization（向量单元）：
       - 对每个 super-group (256 个 INT4 值 = 128 bytes，精确填充 1 个 HVX 寄存器)：
         a. vlut16 查表：INT4 → FP16 值（单指令转换，含 qfloat 格式）
         b. vlut16 查表：广播 4 组 FP16 scales 到向量寄存器
         c. HVX FP16 向量乘法：dequantized_weight = int_value * scale
       - 输出：连续 FP16 weights 写入 TCM（HMX 期望的 tile 布局）

    3. HMX GEMM（矩阵单元）：
       - 加载 FP16 activation tile (32×32) + dequantized FP16 weight tile (32×32)
       - mxmem 指令：paired load activation + weight tiles
       - FP16 矩阵乘累加 → internal accumulator (FP32 precision)
       - cvt.hf = acc()：累加器结果转 FP16
       - mxmem store：输出 tile (32×32) 写回 TCM
       - 外层级 column-major tile 迭代，tile 级 inner product
       - 支持 per-channel scale & bias（256-byte bias region）

    4. DMA 写回（NPU TCM → DDR）：
       - DMA 引擎将输出 FP16 activations 从 TCM 写回 DDR shared memory

    5. CPU 侧读取结果：
       - CPU 从 shared memory 读取 NPU 计算完成的输出
       - 执行未在 NPU 上实现的算子（如 lm_head vocab projection）

    关键性能指标：
      - 单次 dequantization GEMM kernel latency (μs)
      - 端到端 decode throughput (tokens/s) = batch_size × tokens / decode_time
      - HMX 利用率 = 有效 tile 计算量 / 峰值 tile 计算量
        （test-time scaling 通过增大 batch_size 填充 tile 行来提升利用率）

    === FlashAttention Kernel 执行流程（NPU 侧 Algorithm 1） ===
    输入：FP16 Q ∈ R^{B_q×d}, K_j ∈ R^{B_kv×d}, V_j ∈ R^{B_kv×d}
    参数：head_dim d, Query tiles T_q, KV tiles T_kv
    输出：FP16 O ∈ R^{B_q×d}

    对每个 KV tile j：
      1. HMX MatMul: S = Q × K_j^T  (FP16 GEMM, HMX FP32 accum)
      2. HVX rowmax: m_new = max(m_old, rowmax(S))  (HVX 向量比较)
      3. HVX LUT_Exp: P = LUT[S - m_new]  (vgather 查 32768-entry 预计算 LUT)
         - S 为 FP16，exp 输入 ≤ 0 (Safe Softmax)
         - 忽略 MSB (sign bit)，左移 1 bit → byte offset → vgather 并行查表
         - 单指令 gather 64 个 FP16 值
      4. HVX rowsum: l_new = exp(m_old - m_new)*l_old + rowsum(P) (FP32 accum)
      5. HVX rescale: O = diag(exp(m_old - m_new)) × O + P × V_j (FP16, HMX GEMM)
      6. 最终：O = diag(l)^{-1} × O
    ```

## Efficient, VRAM-Constrained Cross-Lingual Model Inference on Client Devices

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：Pipelined Sharding 的 Profile 阶段对 CPU 和 GPU kernel 进行 benchmark，覆盖不同量化格式（f16, q4, q2_k）、不同 op shape、不同 CPU 线程数。Plan 阶段通过 roofline model + profile 数据，对每个 token tier 在 GPU-only / Static / Dynamic 三种 kernel 调度策略之间做 cost 比较，选最低 cost plan。Infer 阶段运行时按 schedule table 分派 layer 到 CPU 或 GPU 执行，支持 compute/copy overlap（async copy backend + split scheduling callback）。VLMOpt 中 Tiled FlashAttention 对 vision encoder 的 Q 做 tile 以限制 KQ 张量峰值显存（1440p 下 attention 显存 <2GB）。
  - 实验比较：单用户 LLM 在不同 VRAM budget 和 context length 下的 TPS/TTFT（对比 llma.cpp -ngl 手动调优）；多用户 batched 下的 TPS scaling（bs=1/4/16/64）；CPU thread count 和 PCIe 代际对调度性能的敏感度分析；不同 scheduler plan（GPU-only/Static/Dynamic）在不同配置下的被选择分布。

- 后端平台是什么，配置是什么。
  - GPU 后端：NVIDIA RTX 3500 (Ada Lovelace, 12GB), RTX 5070 Ti (Blackwell, 16GB), RTX 5090 (Blackwell, 32GB)
  - CPU 后端：Intel Ultra 7 (16 cores) / AMD Ryzen 7 (8 cores) / AMD EPYC (16 cores)
  - 异构互联：PCIe Gen4 (cli1, 13 GBps effective) / PCIe Gen5 (cli2/cli3, 50 GBps effective, 64 GBps peak)

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：llama.cpp b6097（开源 GGML 推理框架），论文在此基础上实现了 pshard runtime。
  - 修改内容（kernel 层面）：
    1. Kernel profile database：存储各 kernel 在 CPU/GPU 上的 latency，key 为 (quant_format, op_shape, thread_count)。
    2. Roofline model：结合硬件 peak compute（GPU TFLOPS, CPU cores × frequency）和 PCIe bandwidth，估计每种 plan 的 cost。
    3. Schedule lookup table：每个 token tier → (selected_plan, layer_assignment, stream_schedule)。
    4. Async copy backend：实现 CPU→GPU weight streaming 与 GPU kernel execution 的 overlap。
    5. Split scheduling callback：执行时动态调整 layer 在 CPU/GPU 间的分派。
    6. Tiled FlashAttention kernel（vision encoder）：Q 分 tile 处理，限制 O(N²) KQ 中间张量峰值。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源状态：代码上游 PR https://github.com/ggml-org/llama.cpp/pull/22692（open）。
  - 评估原理与全过程（以 nemo8b f16 在 cli3 上 4G VRAM budget 为例）：
    1. **Install-time profiling**：`pshard profile` 命令在用户机器上运行。对每个 op（attention Q/K/V proj, FFN up/gate/down, output proj），在 CPU（不同线程数：1/2/4/8/16）和 GPU 上分别 warmup+benchmark 多次取中位数。也在 PCIe contention（同时跑 memory copy）下测 CPU kernel。输出 `kernel_profile.db`。
    2. **Planning**：`pshard plan --vram-budget 4G --model nemo8b-f16.gguf`。Planner 读 profile db，遍历 token tiers：对每个 tier，估算 GPU-only plan cost = Σ GPU_kernel_latency + weight_transfer_time（全部 weight 过 PCIe）；Static plan cost = Σ(CPU_kernel_latency || GPU_kernel_latency) per layer + intermediate_tensor_transfer；Dynamic plan cost = Static + overlap_adjustment。选 min cost plan。输出 `schedule_lut.bin`。
    3. **Inference 评估**：`llama-cli -m nemo8b-f16.gguf --pshard --vram-budget 4096` 运行标准 benchmark（如 ShareGPT 或合成 prompt）。每 decode step：统计 new_tokens → 查 LUT → 提取 plan 和 layer assignment → CPU 线程池执行 CPU-resident layers，同时 CUDA stream 执行 GPU-resident layers，async memcpy 做 weight streaming → sync point 收集结果 → 输出 token。记录 wall-clock time 计算 TPS/TTFT/E2EL。
