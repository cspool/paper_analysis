## PipeSSD (Fused Pipelined SSD Dataflow)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PipeSSD 是 HLX 论文提出的首个融合流水线 SSD（State-Space Duality）数据流。它在两个层面创新：(1) **块级融合（fused SSD）**——将 GPU 上 5 个分离的 SSD kernel（chunk cumsum, chunk state, state passing, BMM chunk, chunk scan）融合为单一 kernel 的 6 个操作组（dA pre-processing、$Y_{Diag}$、$Y_{Off}$、$Y_{Final}$、$states_N$、update states），类似 FA-2 的融合思想，消除中间数据的 DRAM 访存；(2) **三阶段细粒度流水线**——将融合后的操作按依赖关系分为三个阶段流水线执行，并利用 Y_Off 与 states_N 计算间的独立性实现并行。关键效果：DRAM 访问减少 $6.8\times$，中间数据量从 642KB 降至 58.5KB（$11\times$），compute utilization 从 GPU 的 26.9%（A100）/38%（H100）提升至 78.4%。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fused SSD 的 6 个操作（单 block 内 for loop），伪代码：

```
# Fused SSD (before pipelining)
for chunk_i in range(c):  # c chunks, linear complexity O(c)
    # 1. Pre-processing (related to dA)
    sdt = softplus(dt + dt_bias)            # [b,n,cl]
    dA_CS = cumsum(sdt × A)                  # cumulative decay
    
    # 2. Y_Diag computation (diagonal block)
    CB_T = C @ B^T                           # [b, cl, cl]
    CB_TLdt = CB_T × L × dt                 # element-wise
    Y_Diag = CB_TLdt @ x                     # [b, h, cl]
    
    # 3. Y_Off computation (off-diagonal)
    dC_Off = C × exp(dA_CS)                  # decayed C
    Y_Off = dC_Off @ states_int              # [b, h, cl]
    
    # 4. Y_Final combination
    Y_Final = Y_Diag + Y_Off
    
    # 5. states_N (new states for current block)
    dBdt_T = (B × dt)^T     # [b, s, cl]
    states_N = dBdt_T @ x   # [b, s, h]
    
    # 6. Update states (row-wise dependency from previous chunk)
    states_int = exp(dA_CS[-1]) × states_int_prev + states_N
```

PipeSSD 的三阶段流水线映射：

```
Stage 1 (RVPE): dA pre-processing (sdt, dA_CS computation)
    ↓
Stage 2 (DPE#0 → RVPE → DPE#1): CB^T → CB_TLdt → Y_Diag
    ↓ (Y_Diag stored in GS)
Stage 3 (RVPE → DPE#0 ∥ DPE#1 → UpE): 
    RVPE: dC_Off and dBdt^T (via mux/demux direction switch)
    DPE#0: Y_Off = dC_Off @ states_(i-1)
    DPE#1: states_N = dBdt^T @ x           (∥ concurrent)
    UpE: Y_Final = Y_Diag + Y_Off      (∥ concurrent)
         update states = states_(i-1) × exp(dA_CS[-1]) + states_N
```

注意：与 PipeFlash 不同，PipeSSD 的 for loop 为 linear（单方向沿 chunk 维迭代），且存在 column-wise dependency（states 从前一 chunk 传递到后一 chunk）和 row-wise dependency（$Y_{Diag}$ 与 $Y_{Off}$ 需累加，$states_{int}$ 与 $states_N$ 需累加）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PipeSSD 仅在 HLX 自研 cycle-level simulator 中实现。Fused SSD 在 GPU 上不可行的原因：即使消除了 DRAM 访问，融合后中间数据 642KB/block 远超 GPU SM 的寄存器+共享内存容量（导致 register spilling），且列向依赖消除了 sequence length 维度的并行性。PipeSSD 的 11× 中间数据压缩（642KB→58.5KB）使之可在专用硬件的片上 SRAM 中完整存放。RVPE 中的 mux/demux 方向切换机制支持 Stage 3 中 dC_Off 和 dBdt^T 的并发计算及数据流向控制。Y_Diag 在 Stage 2 计算后暂存于 GS（Global Scratchpad），Stage 3 中从 GS 读回用于 Y_Final 加和。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models
