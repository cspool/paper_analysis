## 双内存墙（Dual Memory Wall：权重内存墙与 KV Cache 内存墙）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 双内存墙是 SingularBit 论文提出的 LLM 推理内存带宽瓶颈概念：自回归解码同时面临两个随 inference-time scaling（长上下文+长生成+推理扩展）加剧的访存瓶颈（Fig.3）。①权重内存墙：每步解码重复从片外取模型权重，每 token 产生 $O(L\cdot D^2\cdot b^W)$ 权重流量（L 层数、D 隐藏维、b^W 权重位宽），整个解码会话总流量 $O(N)$，在生成早期（KV 未长大时）主导延迟（<1K context 时占 >70% 内存延迟）；②KV cache 内存墙：每步访问所有前序 token 的 K/V 向量，缓存以 $O(N\cdot L\cdot D\cdot b^{KV})$ 增长，解码会话总流量 $O(N^2)$，在长上下文/推理场景（N 达数万）后期主导。两个瓶颈随请求顺序出现（weight-bound 阶段 → KV-bound 阶段），交叉点随每请求的 prompt/生成长度动态变化，单一 regime 优化的系统无法同时覆盖。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在硬件中的体现与应对（论文的加速器视角）：A100 类 GPU 以固定 2TB/s HBM 带宽服务两种流量——短序列时权重读（如 Llama2-7B 每步 ~14GB 权重流量）占满带宽、长序列时 KV 读（每步读全部历史 KV）占满带宽；权重压缩（OmniQuant/LUT-TensorCore）只减前者、KV 压缩（KIVI/Oaken）只减后者，各自在另一 regime 失效。SingularBit 加速器对双墙的统一应对：SingularBit-W 用 rank-aware 混合精度把权重位宽降到平均 2-bit（减 $b^W$）、SingularBit-KV 用 token×rank 混合精度把 KV 位宽降到 ~2-bit（减 $b^{KV}$），且 KV 走 K'/V' 中间表示（省掉 K/V 投影的 U 矩阵访问），SingularBit-WKV 把总 DRAM 流量最多降 8.5×——权重与 KV 两条访存路径同时变窄。论文数据：单独权重型方法在长序列吞吐下降（KV 拖累）、单独 KV 型方法在短序列受限（权重流量），WKV 全序列范围维持高吞吐（batch 16–64、7B/13B 稳定）；对 decode-heavy reasoning 达 5.27× speedup、4.64× 能耗节省。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：作为分析与设计原则使用——先以流量公式（$O(LD^2b^W)$ vs $O(NLD b^{KV})$）识别瓶颈区间，再选择对应压缩维度；对加速器设计意味着必须同时具备"低比特权重执行"与"在线 KV 压缩"两类硬件（SingularBit 的 Tensor Core + Compression Engine 双部件即为此）。评估上用它构造两类压力场景：prefill-heavy 长上下文（LongBench，KV 主导）与 decode-heavy 推理（GSM8K/CoQA，权重+KV 交替主导）。与相关概念对比：传统"内存墙"指计算快于访存导致带宽受限；双内存墙进一步区分两个随不同变量（模型规模 vs 序列长度）增长的访存源，并指出只有联合压缩才能同时缓解。论文数据：LongBench 上联合的 SingularBit-WKV 43.7% 远优于最佳 prior 组合 GuidedQuant+ReCalKV 29.6%（后者组合反而比单独更差）。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
