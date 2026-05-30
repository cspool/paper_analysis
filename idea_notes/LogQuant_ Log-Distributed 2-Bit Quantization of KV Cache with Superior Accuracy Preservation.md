## LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

- baseline方法是什么？
  KiVi (Liu et al., 2024c)：一种 training-free 的非对称 2-bit KV Cache 量化方法。KiVi 仅保留最近 R 个 token 为 BF16 全精度，将所有更早的 token 量化为 2-bit。核心假设是"最近的 token 总是最重要的"，因此采用简单的均匀窗选择策略。Per-channel 量化，group size 64。

  Baseline 全栈执行例子（以 Llama3.1-8B-Instruct 解码阶段为例）：
  - **算法层**：KiVi 的非对称 INT2 量化。KV Cache 中第 1 到第 (L-R) 个 token 的 K/V 被量化为 INT2（per-channel, group=64），最近 R=128 个 token 保持 BF16。解码时对量化 token 做 dequantize→BF16，与全精度 token 拼接后计算标准 Scaled Dot-Product Attention。
  - **系统框架层**：HuggingFace transformers pipeline。Cache 类统一管理 KV 存储与量化/反量化。每次生成新 token 时触发量化逻辑——当 reserved tokens 超过 R 时将最早的全精度 token 量化为 INT2。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 PyTorch GPU kernels（matmul + softmax）。反量化操作（INT2→BF16）插入在 attention 计算前，每个 decoding step 的 K/V 加载后执行，无 custom fusion。
  - **硬件架构层**：NVIDIA GPU（H100 用于效率测试）。KV Cache 存储在 GPU HBM 中，每个 decoding step 从 HBM 加载全部 K/V 到计算单元。量化 KV 通过反量化恢复为 BF16 后参与 attention 计算。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  LogQuant 将 KiVi 的"均匀最近窗"替换为"对数分布 token 选择"，核心洞察是：attention spikes（高注意力分数的位置）遵循对数分布——距离当前位置越远的 token，其 attention spikes 的密度越稀疏（见图 1）。基于此，LogQuant 以递减密度保留 token：最新 W 个 token 密度 p，次新 W 个 token 密度 p/2，再次新 W 个 token 密度 p/4……以此类推，形成自然对数稀疏性。

  三个具体设计解决 KiVi 的三大缺陷：

  **缺陷 1：KiVi 的均匀窗硬截断会丢失远处的关键 token**。KiVi 保留最近 128 个 token 但丢弃了所有更早的 token，然而许多问题（如"法国的首都是什么？"）的答案 token 可能位于远处。LogQuant 的对数选择在远处仍有稀疏保留——每 2^k 个位置保留 1 个 token——捕获了这些"远处但重要"的 token。实验证明对数选择方案的 token coverage（公式 1：所选 token 的平均 attention score）在所有模型上均优于 KiVi/StreamingLLM/H2O（图 4）。

  **缺陷 2：KiVi 未区分量化与 eviction 的信息损失差异**。论文证明在相同的对数选择方案下，量化（降低精度）比 eviction（移除 token）保留更多信息。这是因为 softmax 归一化使得 eviction 移除 token 后剩余 token 的 attention 权重被重新分配，造成更大的 attention 分布偏差。量化仅降低单个 token 的数值精度，不改变 attention 的归一化结构。实验验证：L1 attention 误差——LogQuant (2-bit) 432.50 vs LogQuant (Eviction) 1076.70，证明量化策略优于 eviction（表 2）。

  **缺陷 3：KiVi 未利用 attention 计算的置换不变性优化内存布局**。LogQuant 证明 A·V = A_P·V_P（P 为任意置换），即 K/V Cache 中 token 的排列顺序不影响 attention 输出。这使得 LogQuant 可以将全精度 token 和量化 token 分别连续存储（而非按原始位置交错存储），改善内存局部性，减少碎片化，无需额外计算开销。

  论文方法全栈执行例子（以 Llama3.1-8B-Instruct 解码阶段为例）：
  - **算法层**：LogQuant 的 log₂-分布式 2-bit 量化。Algorithm 1 的 APPENDTOKEN 过程：KV Cache 起始为空，依次追加 token。当 cache 长度 < 3W 时直接追加（全精度）。当长度达到 3W 时，将前 2W 个 token 做步长=2 子采样（保留一半），与新 token 拼接，总长回到 2W。反复执行后，cache 中 token 的保留密度自然呈现：Window_0 密度 p，Window_1 密度 p/2，Window_2 密度 p/4……非保留 token 量化为 INT2。W = ⌊KiVi_R/3⌋ = ⌊128/3⌋ = 42，实际最多 126 个全精度 token。压缩率 ≈ 16L / (2(L-126) + 16×126)。
  - **系统框架层**：继承 HuggingFace transformers Cache 类的 derived class LogQuantCache。量化后端使用 Quanto（Key-per-channel 策略，也可切换 HQQ）。position-agnostic 重组：将全精度 K/V 与量化 K/V 分别连续存储。与 HuggingFace 推理 pipeline 无缝兼容。batch size 比 BF16 baseline 增加 60%。
  - **编译框架层**：论文未明确说明。未来工作提及 operator fusion——将 dequantization 与 attention 计算融合为单一 kernel，直接在量化数据上计算 attention，消除反量化开销。
  - **kernel调度层**：标准 PyTorch attention kernels。每个 decoding step 中，量化 K/V 需先反量化至 BF16 再参与 attention 计算——反量化操作是当前 throughput 瓶颈之一。25% 吞吐量提升主要来自更大 batch size（内存节省释放了 batch 扩展空间）而非单步计算加速。论文明确指出可进一步通过 fused kernel 优化。
  - **硬件架构层**：NVIDIA H100 48G MIG。KV Cache 存储于 GPU HBM——2-bit 量化将非保留 token 的内存从 16-bit/entry 压缩至 2-bit/entry（~8× reduction）。全精度保留 126 token（BF16），其余 INT2。48GB 内存限制下，batch size 从 baseline 的 X 增至 1.6X。Dequantization 在 HBM→SRAM 加载后、Tensor Core 计算前执行。

  对比 baseline 的关键差异：
  - **KiVi 均匀窗 (uniform recent window)** → **LogQuant 对数窗 (log-sparse window)**：KiVi 保留最近 128 token 但完全丢弃更早 token；LogQuant 以 log₂ 递减密度在更远位置保留 token，捕获跨距离的关键信息。
  - **KiVi 仅量化无选择性** → **LogQuant 量化+对数选择联合设计**：KiVi 对所有非保留 token 一视同仁地量化；LogQuant 先通过对数分布选择保留更重要的 token（全精度），再对剩余 token 量化——重要性判断基于 attention spike 的位置分布规律而非单一时间衰减假设。
  - **未利用置换不变性** → **利用 A·V = A_P·V_P 重排**：KiVi 按原始位置存储 K/V；LogQuant 将全精度和量化 token 分别连续存储，改善内存局部性。
