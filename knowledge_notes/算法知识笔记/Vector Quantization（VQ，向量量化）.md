## Vector Quantization（VQ，向量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VQ 是非解析量化（non-analytic quantization）的一种：把权重矩阵 W∈R^{K×N} 沿 K 维每 d 个连续元素分为一组 d 维向量，用 k-means 等聚类把所有权重向量映射到共享码本（weight codebook, WC）B∈R^{d×2^n} 的 2^n 个 centroid（n 为索引位宽），权重矩阵被替换为低精度索引矩阵（weight index, WI）I∈[0,2^n)^{K/d×N}。与解析量化（AWQ/GPTQ 的线性缩放+取整，可闭式表达）不同，VQ 的量化函数没有算术闭式，重建靠 1-to-1 查表。EVA 采用 d=8、n=8（码本 2^8=256 条目），单码本平均每元素 n/d=1 bit；用 C 个码本叠加（加法 VQ）达到有效精度 q=C·n/d bits（C=2/3/4 → 2/3/4-bit）。VQ 在 2-bit 级仍保持高保真（解析法在此崩坏），但 decode 时 1-to-many 查表导致不规则访存与 bank 冲突，且比 FP16 推理更慢（VQ-LLM 观察）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 离线量化（k-means / AQLM）：W ∈ R^{K×N} → 每 d 元素一组 → 聚类得 B ∈ R^{d×2^n}、I ∈ [0,2^n)^{K/d×N}
# 在线 decode（常规 VQ）：y = x @ W_hat，W_hat[i,j] = B[:, I[i,j]]（逐元素查表重建后 GEMV）
# 压缩比：FP16 权重 K×N×2B → 索引 (K/d)×N×(n/8)B + 码本 d×2^n×2B
```
EVA 例子：LLaMA-2-7B FC 层 W∈R^{4096×4096}，d=8、n=8 → WI∈[0,256)^{512×4096}（1 字节/索引）+ WC∈R^{8×256}（FP16），单码本压缩 16×；AQLM-2×8（2 码本）平均 2-bit。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：AQLM（EVA 采用的算法，https://github.com/Vahe1994/AQLM）、GPTVQ、QuiP# 等 PyTorch 后训练量化框架，离线 k-means/残差量化学习码本；EVA 仓库（https://github.com/dbw6/Eva.git，MIT）的 algorithm/ 提供 eval_ppl.py/lmeval.py 复现精度（依赖 aqlm[gpu,cpu]>=1.1.6）。使用方式：作为 weight-only 压缩手段部署到内存受限场景；GPU 上常规实现因查表不规则常比 FP16 慢（VQ-LLM 用 hot/cold profiling 缓解）；EVA 用"码本×输入 GEMM + 输出码本查找"（VQ-GEMM）把查表变成规则访存，专门适配硬件加速器。

涉及论文标题：
- EVA: Accelerating LLM Decoding via an Efficient Vector Quantization Architecture
