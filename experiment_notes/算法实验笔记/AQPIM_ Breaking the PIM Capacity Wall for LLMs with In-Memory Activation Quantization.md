## AQPIM: Breaking the PIM Capacity Wall for LLMs with In-Memory Activation Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  提出基于Product Quantization (PQ)的PIM-aware KV cache在线量化压缩框架AQPIM。核心算法创新：(1) PQ-based KV cache量化：将高维KV vector分解为m=32个子向量，用k-means clustering生成K=512个centroid per subvector的codebook，在线适应activation分布，compression ratio最高达80%+；(2) Importance-weighted k-means clustering：计算每个token的attention score权重w=sum(S[-t:, :], axis=0)，在k-means迭代中使用weighted centroid update（µ_k = Σ w_n x_n / Σ w_n, n∈C_k），使高attention score的token获得更小的量化误差；(3) Channel pre-sorting optimization：基于cosine similarity将高相关channel分组到同一subvector，sorting matrices P_k, P_v离线生成(calibration dataset: Wikitext-2-v1)并absorb到projection matrices W_q'=W_q·P_k, W_k'=W_k·P_k, W_v'=W_v·P_v, W_o=W_o·P_v^T，无运行时overhead；(4) Page-aware windowed clustering：限制每个window内512个centroid，保证indirect lookup完全在单一DRAM row buffer内完成，消除随机访问penalty；(5) PQ-based direct attention computation：将GEMV qK^T转换为query splits×codebook的inner product matrix lookup + summation，无需dequantization直接在压缩数据上计算。实验比较accuracy vs memory reduction ratio trade-off，对比SnapKV(sparse attention)、PQCache(PQ-based offloading)、SKVQ(channel-reorder quantization)，在LongBench 6个task上的accuracy；ablation study验证weighted clustering和channel pre-sorting对accuracy的提升贡献。

- 硬件平台是什么，配置是什么。
  GPU+HBM-PIM异构系统：1×NVIDIA H100 GPU + 5×16GB HBM（GPU+HBMs baseline）；PIM系统中4×16GB HBM替换为4×16GB HBM-PIM（参考AttAcc!架构）。CPU：Intel Xeon Platinum 8480+ Processor。bfloat16精度用于模型推理，FP16用于架构模拟（与prior work AttAcc!保持一致）。

- 模型是什么。数据集和bench分别是什么。
  模型：Mistral-7B-Instruct-v0.2 (32K context window), Llama-3.2-3B-Instruct (128K context window)。Benchmark：LongBench（含NarrativeQA/HotpotQA/GovReport/TREC/PassageRetrieval-en/LCC六个代表性task，覆盖single-/multi-document QA、summarization、few-shot learning、synthetic tasks、code completion）。Calibration dataset: Wikitext-2-v1（用于channel pre-sorting矩阵离线生成）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文代码未直接开源。构建于AttAcc! simulator ([55] https://github.com/scale-attacc-kr/attacc-sim) 和Ramulator2 ([20] https://github.com/CMU-SAFARI/ramulator2) 之上。补充材料见Zenodo (https://zenodo.org/records/17378113)。算法pipeline：
  1. Prefill阶段：GPU生成QKV matrices→KV offload到HBM-PIM→GPU执行attention和projection/FFN→PIM并行生成Key/Value codebook：在BankPE执行distance calculation (DC, FP16 MAC units)→BufferPE执行cluster assignment (CA, MIN unit)→BankPE+BufferPE协同centroid calculation (CC, MUL+SUM+DIV)→迭代4轮收敛。
  2. Decode阶段：GPU生成qkv→offload到PIM→PIM append新token的PQ indices→PIM执行PQ-based attention: query分成m=32 subvectors→BankPE执行query×codebook ATNK (MAC)得inner product matrix→BufferPE lookup indices+softmax (SFM)→BankPE ATNV (attention×value codebook reconstruction)→output回传GPU。
  3. Hyperparameters: m=32 subvectors, K=512 centroids, sink tokens保留前8个full precision, sliding window保留最近32个tokens full precision。Importance weight t=32（window内最近t tokens的attention scores）。
