## Bubble-free pipelining（无气泡流水与 head 映射）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CHIME-PIM 的 attention kernel 流水化设计，目标是让跨芯片数据传输完全被 bank PU 计算覆盖（无气泡）。两个基础观察：(1) 内部内存总线（bank PU↔bank）与外部内存总线（rank PU↔bank PU/shared buffer）可解耦，允许 bank PU 计算与 rank PU 传输同时进行；(2) 借鉴 FlashAttention 的 chunked tile 融合，attention 按 chunk 计算并流水。具体：bank PU 每算出一个 token 的 score 输出 O^s 暂存本地 result buffer，rank PU 立即经空闲外部总线取回，adder 累加后 softmax 单元做 per-chunk softmax；全部 token 处理完后 streaming chunk-wise 做跨 chunk 归一化得到全局正确 S，S 元素写回 DRAM 与后续 context（S×V）计算再流水。无气泡条件 T_comm ≤ T_comp：跨芯片传输时间 T_comm = L_t×N_gqa×N_chips/B_rk（DIMM 多芯片协作使传输放大 N_chips 倍、GQA 进一步放大），bank PU 计算时间 T_comp = L_t×E_h×⌈N_gqa/N_cmr⌉/(B_bk×N_bk×N_hc)，由此推出 head 映射约束 N_hc ≤ E_h×B_rk×⌈N_gqa/N_cmr⌉/(B_bk×N_bk×N_gqa×N_chips)——MHA（N_gqa=1）取 N_hc=8、GQA-8 取 N_hc=1。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
score 阶段流水（一个 head、N_hc 个 chips）：
```
for chunk in chunks:                          # chunk = 单 head 跨多个 bank PU 并行产生的数据
    O_s[chunk] = bank_PUs.MAC(Q, K[chunk])    # bank PU 从 DRAM cell 读 K、从 shared buffer 读 Q
    rank_PU.fetch(O_s[chunk])                 # 经外部总线，与下一 chunk 的 MAC 重叠
    S[chunk] = softmax_unit(adder.accum(O_s[chunk]))   # per-chunk softmax
# 所有 token 完成后：
S = normalize(S_chunks)                       # streaming 跨 chunk 归一化（online softmax 修正）
# S 逐元素写回 DRAM 与 context 计算 S×V 流水
```
无气泡判据推导：T_comm = N_comm/B_comm，N_comm = L_t×N_gqa×N_hc、B_comm = B_rk×N_hc/N_chips → T_comm = L_t×N_gqa×N_chips/B_rk；T_comp = L_t×E_h×⌈N_gqa/N_cmr⌉/(B_bk×N_bk×N_hc)；T_comm ≤ T_comp 即式 (1)，N_hc 是唯一可调变量。效果：MHA 延迟 -27.9%、GQA -74.4%（对 baseline = 跨芯片映射无重叠的 bank-level CHIME-PIM）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：bank PU 与 rank PU 异步执行（rank PU 在 buffer chip 上以逻辑工艺实现，独立于 DRAM 阵列运行）；PIM 命令流 PIM_MAC/PIM_RD_RB 交替驱动；chunk 粒度与中间结果缓冲（rank PU 片上 SRAM）约束 head 足迹。使用方式：MHA 多 head 可映射多 chips（N_hc=8）利于 rank 级负载均衡，GQA-8 大 group 只能 N_hc=1；计算访存比 N_cmr 按 GQA-n 配置保证带宽利用率。类比：与 GPU 上 FlashAttention 的 tiling 目标一致（减少中间物化），但重叠对象是跨芯片数据传输而非 HBM 访存。

涉及论文标题：
- CHIME: A Case for Efficient Long-Context Attention-FC Disaggregated Inference with DIMM-PIM
