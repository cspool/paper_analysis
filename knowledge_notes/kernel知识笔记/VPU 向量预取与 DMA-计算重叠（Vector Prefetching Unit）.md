## VPU 向量预取与 DMA-计算重叠（Vector Prefetching Unit）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VPU 是 BAAP 放在 DMA 引擎与 AP 数据通路之间的微结构：AP 计算当前向量时，非投机地预发射下一向量 load/store 到 DMA 引擎（操作数就绪才发起、命中五级流水线的访存级），用额外一个向量大小的缓冲吸收预取数据。动机：位串行算术每指令数百至数千周期（ap_mul 4n²+4n），若不重叠，DMA 空闲、流水线干等内存——把"位串行长延迟计算"与"bank 级快速 DMA"这两个本来串行的环节并行化。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
周期三分类（论文 §IV-C）：每轮执行周期分为 Overlap（AP 忙 ∧ bank 忙）、Idle（等内存）、Compute-only（AP 忙 ∧ 内存闲）；VPU 把 Idle 转为 Overlap。
```
# VPU 生效的执行流水（decode GEMV，每 token 一个激活向量）
loop:
  VPU.preissue(DMA.load(vec_next))     # 非投机预发射下一向量
  ASU.run(ap_mul(vacc, vw, vec_cur))   # 当前向量位串行乘（与 DMA 重叠）
  vec_cur = VPU.buffer                 # 就绪后切换缓冲
```
消融结果（GPT-2 Large，VL 96–384）：prefill GEMM（计算受限）关 VPU 仅 −5.9~−7.9%；decode GEMV（访存受限）关 VPU −39.0%（VL=96）→ −20.9%（VL=384）——宽向量每指令处理更多数据、每指令访存停顿更少，VPU 可隐藏窗口变小。VPU 把约 1/4 的 decode 执行时间从内存停顿转为有效计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：非投机（不污染状态、操作数就绪检查）、单级向量缓冲、复用 DMA 引擎既有能力（无新增数据通路）。同类参照：PUMICE（DAC 2023）的访存-计算重叠。使用场景：乘法密集 + 流式 DMA 的 kernel（PrIM 的 GEMV/TS/MLP、LLM decode GEMV）；计算受限的 GEMM 收益小，应靠 AP 原始吞吐而非重叠。

涉及论文标题：
- BAAP: Coupling Compute-in-SRAM with DRAM Banks for Near-Memory Processing
