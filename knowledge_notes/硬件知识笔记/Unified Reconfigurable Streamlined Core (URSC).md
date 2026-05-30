## Unified Reconfigurable Streamlined Core (URSC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

URSC（Unified Reconfigurable Streamlined Core）是 HLX 硬件加速器的核心计算模块，设计目标为统一支持 Hybrid Transformer-Mamba 模型的两种核心计算：FlashAttention-2（通过 PipeFlash 数据流）和 SSD（通过 PipeSSD 数据流）。每个 URSC 由四个专用引擎组成：(i) **DPE#0**（Dot-Product Engine #0）——32 lanes × 8 DPU × 16 FP16 MAC，执行 MatMul 和 conv1D；(ii) **RVPE**（Reconfigurable Vector Processing Engine）——2 RVPU + VMEM，执行 softmax/exp/cumsum/SiLU 等向量和特殊函数操作；(iii) **DPE#1**——与 DPE#0 规格相同；(iv) **UpE**（Update Engine）——2 UpU + OMEM，执行 update O/update states 和 Y_Final 组合。四个引擎通过 NoC 和 mux/demux 互连，支持数据在引擎间直接转发而不经过外部 DRAM。URSC 的可重构性体现在：(a) RVPE 内 local NoC 支持 4 种操作模式（PipeFlash local softmax / PipeSSD pre-processing / Y_Diag element-wise mul / Y_Off/states_N element-wise mul）；(b) mux/demux 支持 Stage 3 的数据方向切换（dC_Off→DPE#0, dBdt^T→DPE#1）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

URSC 在 PipeFlash 模式下的数据流（一个 attention block 的完整流水线）：

```
Cycle 1..N:  Top Controller loads K_block, V_block from DRAM → GS
Cycle N+1..: Pipeline starts

DPE#0 (QK^T):
    Input: Q[2×d_head] (broadcast from GS), K[2×d_head]^T (from GS)
    DPU lanes: 每 lane 共享 16 broadcast activations, 各自接收不同 weights
    Output: score[2, block_size] → forwarded to RVPE via NoC

RVPE (local softmax):
    Input: score[2, block_size] from DPE#0
    Mode: "PipeFlash local softmax"
    Ops: rowmax → exp(score - max) → rowsum → div
         rescale_factor = exp(prev_max - current_max) 
    Output: prob[2, block_size] → DPE#1 via NoC
            rescale_factor → UpE via NoC

DPE#1 (PV):
    Input: prob[2, block_size] (from RVPE), V[2, d_head] (from GS)
    Output: PV[2, d_head] → forwarded to UpE via NoC

UpE (update O):
    Input: PV[2, d_head] (from DPE#1), rescale_factor (from RVPE),
           O_prev[2, d_head] (from OMEM)
    Ops: O_new = O_prev × rescale_factor + PV
    Output: O_new → stored in OMEM
```

数据转发全部在片上完成（DPE#0→RVPE→DPE#1→UpE），不经过 GS 或 DRAM。当所有 Q block 行处理完后，UpE 执行 final rescale（O /= final_rescale_factor），结果通过 NoC → GS → DRAM 写回。

URSC 在 PipeSSD 模式下，同样利用 DPE/RVPE/UpE 的流水线能力，但需要 GS 暂存 Y_Diag 结果（Stage 2 产出 → Stage 3 消费），以及 mux/demux 实现 Stage 3 的双向数据流。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HLX 完整芯片由多个 URSC + 全局 GS + NoC + Top Controller + Transpose Unit + DRAM 接口组成。三个配置：(i) HLX^6：6 URSC, 61.44 TFLOPS, 3.04MB SRAM, 对标 TPUv3；(ii) HLX^30：30 URSC, 307.2 TFLOPS, 15.2MB SRAM, 对标 A100；(iii) HLX^60：60 URSC, 614.4 TFLOPS, 30.4MB SRAM, 对标 H100。单 URSC RTL 实现（SystemVerilog），Synopsys Design Compiler 综合（14nm, 625MHz, 0.8V），面积 7.89mm²，功耗 5.39W。DPE 是最主要的面积和功耗来源（约 62.4% 面积，74.9% 功耗）。统一支持两个模型的硬件 overhead 极小：vs Transformer-only 增加 3.0% 面积 + 2.9% 功耗（conv1D/softplus/cumsum 逻辑），vs Mamba-2-only 增加 4.4% 面积 + 3.5% 功耗（softmax/reciprocal/mux/demux 逻辑）。SOTA 加速器对比：VGA（H3 only）、MARCA（Mamba-1 only）、SOFA（attention only with sparsity），均仅支持单一模型类型。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models
