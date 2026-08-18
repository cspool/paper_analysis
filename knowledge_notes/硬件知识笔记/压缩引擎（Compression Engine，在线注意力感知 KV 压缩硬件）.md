## 压缩引擎（Compression Engine，在线注意力感知 KV 压缩硬件）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SingularBit Compression Engine 是加速器中执行在线 KV cache 压缩的专用硬件单元，与 SingularBit Tensor Core 并行工作，把"token 重要性打分 → token/rank 混合精度量化 → 位打包 → 路由"做成专用逻辑（Fig.10）。组成：global GeMV 单元（算 QK^T）与 global SIMD 单元（softmax）负责注意力计算；precision allocator 内含 maximum-tracking 逻辑（head-wise max pooling、归一化、recent-k=128 窗口、每 token 重要性=近 k 步最大注意力），按容量线性递增的阈值把重要性映射到 token 级最大位宽（base precision b 可调，是压缩-精度权衡旋钮）；FP16-to-INTx 量化器按 token 最大位宽与 rank 元数据沿 rank 边界逐级降精度（K'/V' 输入）；mixed-precision compressor 做无 padding 的紧凑位打包；quantized KV router 在 KV buffer 内按 token 位宽做位级打包并把逻辑 token 索引翻译成物理存储地址；precision information table 记录每 token 位宽供反量化。开销极小：compressor 21.5 mW/1.23 mm²（占压缩引擎 23.3%/27.9%），一个压缩引擎共享 ≥8 个 tensor core，芯片级面积/功耗开销 <4%。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（每 decode 步，Fig.10）：①global GeMV 算 $QK^T$、global SIMD 做 softmax，得当前注意力图 $A_t\in\mathbb{R}^{H\times N_t}$；②precision allocator 的 maximum-tracking 逻辑逐 head max pool → 归一化 → 写入 recent-k 窗口，算每 token 重要性并映射为 token 级最大位宽；③KV quantizer 接收 K'=xU_{W_K}、V'=xU_{W_V}，从每 token 最大位宽出发沿 rank 边界逐级降精度（FP16→INTx）；④compressor 把量化值按各自位宽无 padding 紧凑打包；⑤router 按 token 位宽决定物理地址写入 KV buffer，precision information table 记录位宽表。压缩与 post-attention 计算并行进行，且在下一 attention 层开始 KV 压缩前完成——不在关键路径、不增加端到端延迟。attention 前 V^T 重构在 tensor core 执行（+5%@ctx64、+2%@ctx2048 延迟，换来 KV DRAM 流量约 5× 下降）。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：ASIC 综合实现（28nm @1GHz，论文未指明 EDA 工具）；片上还含 attention 计算所需资源（global GeMV/SIMD 66.2 mW/3.15 mm² 占压缩引擎 72.0%/71.4%），即压缩引擎同时承担 attention score 计算。使用方式：与 tensor core 流水——tensor core 做权重侧 GEMM 与 V^T 重构，压缩引擎并行算 attention 并在线压缩新 token 的 K'/V'；无需 CPU/软件干预。论文数据：GPU 无法原生支持细粒度 KV 压缩（需格式转换与数据移动开销），而压缩引擎使 KV 压缩能耗收益（算法侧 30.8–32.8%）被完全兑现到系统级（SingularBit-WKV 79% 系统能耗下降、DRAM 流量最多降 8.5×）。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
