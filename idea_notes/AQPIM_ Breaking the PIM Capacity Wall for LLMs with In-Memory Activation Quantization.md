## AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

- baseline方法是什么？
  Baseline是SOTA PIM-based LLM inference accelerator AttAcc! [56] 配合标准KV cache处理方法。(1) AttAcc!假设KV cache完全fit在PIM的有限on-chip memory中，不做压缩；(2) 当KV cache溢出HBM-PIM memory时，Baseline被迫通过offloading (KV cache→CPU memory via PCIe) 处理，产生巨大communication overhead（GPU-CPU通信占decoding latency的90~98.5%）；(3) 如果使用现有quantization方法(KVQuant/SKVQ等uniform/non-uniform quantization)，需在面积受限的BankPE中添加INT32 MAC units等额外ALU用于scaling/dequant，面积开销从FP16-only的50%增加到FP16+INT32的126%，严重损害memory density；(4) 如果使用sparse attention (SnapKV/StreamingLLM) 配合offloading，scattered memory access pattern与PIM对data locality的刚需冲突。

  全栈执行例子（以Mistral-7B-Instruct-v0.2, LongBench, S_in=4096, S_out=128/batch_size=16为例）：
  - 算法层：标准attention GEMV计算qK^T (q: [1,d] × K: [N,d]^T)，KV cache无压缩，长context下N增长导致KV cache size线性膨胀。当HBM-PIM capacity (4×16GB=64GB) 无法容纳全部KV cache时，触发KV offloading到CPU memory。
  - 系统框架层：GPU+HBM-PIM异构系统，GPU负责projection/FFN，PIM负责attention GEMV。无KV cache compression，decode阶段GPU需等待PIM完成attention后处理后续层。
  - 编译框架层：论文未明确说明。
  - kernel调度层：AttAcc!的attention kernel直接使用BankPE FP16 MAC units做标准GEMV (q×K^T)，每decode step需读取完整K matrix所有行。KV cache expand后offloading via PCIe → GPU-CPU communication latency 11.385ms/decode step (batch_size=32, gpu+cpu场景，图13)，远超matmul本身的0.181ms。
  - 硬件架构层：AttAcc!的HBM-PIM架构：BankPE在DRAM die bank旁（面积受限），BufferPE在buffer die（含accumulators和softmax units）。不支持compression、不支持intra-row indirection、无PQ相关commands。BankPE仅含FP16 MAC units，无INT/quantization专用ALU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出AQPIM框架，通过algorithm-hardware co-design用PQ在线聚类量化解决PIM capacity wall。核心设计及其与Baseline缺陷的对应关系：

  1. **Capacity Wall (Baseline: KV cache溢出PIM capacity) → PQ-based online KV cache quantization**：利用PIM高内部带宽在线执行k-means clustering生成codebook（m=32 subvectors, K=512 centroids），在prefill期间与GPU computation并行完成（4次迭代收敛），compression ratio最高80%+，将KV cache footprint从~120GB降至~0GB水平（图13, gpu+pq: 0.181ms, aqpim: 0.047ms）。

  2. **Area Overhead (Baseline: 添加quantization ALU需126% area) → PQ eliminates dequantization, reuses existing FP16 MAC**：AQPIM将GEMV qK^T转换为query splits×codebook的lookup+summation（图5: query→m subvectors→multiply with codebook→inner product matrix→lookup indices→sum→qK^T approximation），无需dequantization step，完全使用现有BankPE FP16 MAC units (ADD/MUL/SUM)。仅添加intra-row indirection硬件0.0565mm² (0.43% of BankPE area)。

  3. **Random Access Penalty (Baseline: PQ lookup产生大量随机DRAM row activation) → Page-aware windowed clustering + Intra-row indirection**：算法端保证每个window内不超过512个centroid={512 inner product values}，完全fit在1KB HBM row buffer中；架构端添加GRF→MUX→column decoder的indirection datapath，将随机logical access转为单一row-buffer hit。

  4. **Accuracy Loss (Baseline: naive PQ equal treatment causes accuracy drop) → Importance-weighted k-means + Channel pre-sorting**：Weighted clustering使高attention score token获更小quantization error（µ_k = Σ w_n x_n / Σ w_n, w=last t tokens' attention scores）；Channel pre-sorting offline生成sorting matrices absorb到projection weights，将高cosine similarity channels聚合同一subvector减少信息损失。Ablation: Standard PQ avg 44.29 → AQPIM avg 50.00 (+5.71), compression scenario K=128 centroids。

  全栈执行例子（以Mistral-7B-Instruct-v0.2, LongBench, S_len=32768, batch_size=16):
  - 算法层：PQ-based attention → lookup+summation替代GEMV。Importance-weighted k-means (t=32, m=32 subvectors, K=512 centroids) → codebook+indices替代full KV cache。Channel pre-sorting (Wikitext-2-v1 offline) absorb到projection。Sink tokens (前8)+sliding window (最近32)保留full precision。
  - 系统框架层：GPU+HBM-PIM pipeline: GPU生成qkv→offload到PIM→PIM append PQ indices+PQ-attention→output回传GPU→GPU projection/FFN。Sequence-by-sequence pipelining隐藏GPU-PIM sequential gap。Head-wise HBM mapping + subvector-wise bank mapping。
  - 编译框架层：论文未明确说明。
  - kernel调度层：BankPE执行DC/ATNK/ATNV (FP16 ADD/MUL/SUM)，BufferPE执行CA/SFM (MIN/DIV/EXP)。4-iteration codebook generation在prefill期间与GPU并行→query×codebook inner product→intra-row indirection lookup from row buffer→summation for qK^T→softmax→attention reconstruction。256个centroid = 512 FP16 values = 1KB = 1 row buffer → 每window仅1次row activation。
  - 硬件架构层：HBM3-PIM with BankPE+BufferPE dual architecture, intra-row indirection (GRF→MUX→column decoder), new PIM commands (PIM_SET_CONFIG/PIM_MAC_AB/PIM_SFM/PIM_RET等), page-aware memory allocation。Area overhead仅0.43%。Decoding per-step latency 0.12× vs GPU baseline (8.33× speedup), energy 0.07× vs GPU baseline (图14, S_len=32768)。3.4× speedup over SOTA PIM (AttAcc!)。
