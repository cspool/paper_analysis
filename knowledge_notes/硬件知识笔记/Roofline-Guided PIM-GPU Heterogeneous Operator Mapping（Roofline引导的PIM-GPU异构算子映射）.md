## Roofline-Guided PIM-GPU Heterogeneous Operator Mapping（Roofline引导的PIM-GPU异构算子映射）

术语是什么？通过联网搜索让回答具体和精准。
Roofline-Guided PIM-GPU Heterogeneous Operator Mapping是SADDLE中基于roofline模型指导PIM vs GPU operator分配的方法论。不同于单设备roofline（仅判断单设备上compute-bound vs memory-bound），SADDLE扩展roofline为cross-device比较：同时绘制PIM和GPU的roofline curves，根据operator的当前(Effective Batch Size, Draft Length)参数估计其arithmetic intensity，与两设备的ridge points比较决定optimal execution target。当operator CI低于PIM ridge时→PIM更优（PIM内部带宽TB/s级远超GPU HBM的TB/s级）；当operator CI高于GPU ridge时→GPU更优（GPU peak compute TFLOPS远超PIM PE array）；中间区域为"共优区"（两设备均可，选择利用率较低的设备）。该分析是Scheduler动态remap决策的理论基础。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。
SADDLE的cross-device roofline分析（Fig.6, OPT-66B+OPT-1.3B）：
```
PIM roofline: peak_compute ~X TFLOPS, peak_bw ~6.4 TB/s (per stack)
GPU roofline: peak_compute ~312 TFLOPS (A100 BF16 TC), peak_bw ~1.5 TB/s

Ridge points:
  PIM_ridge ≈ X TFLOPS / 6.4 TB/s → ~1-2 FLOPs/Byte
  GPU_ridge ≈ 312 TFLOPS / 1.5 TB/s → ~208 FLOPs/Byte

Operator mapping decision:
  DLM Attention (1 token/req): CI ≈ d_head/2 → ~32 FLOPs/Byte
    → Below GPU ridge, above PIM ridge → PIM (bandwidth advantage)
  DLM FC (eff_bs=4): CI ≈ 4 → between ridges → PIM preferred
  DLM FC (eff_bs=12): CI ≈ 12 → closer to GPU ridge → GPU preferred
  TLM Attention (1 token/req): CI ≈ 32 → PIM
  TLM Attention (8 tokens/req): CI ≈ 96 → GPU (higher compute utilization)
  TLM FC (pooled tokens): CI ≈ N_tokens → always > GPU ridge → GPU

关键观察: draft_length ↑ → TLM attention CI ↑ → optimal target shifts PIM→GPU
           eff_batch_size ↓ → DLM FC CI ↓ → optimal target shifts GPU→PIM
```
该交叉roofline揭示了传统单设备roofline无法捕捉的维度：operator可能在GPU上memory-bound但在PIM上compute-bound，或反之。SADDLE利用此分析为每speculative iteration做per-operator device selection。

