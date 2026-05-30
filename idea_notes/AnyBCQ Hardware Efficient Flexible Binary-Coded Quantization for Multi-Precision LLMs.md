## AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs

- baseline方法是什么？
  Baseline 是 **Any-Precision LLM**（Park et al., 2024），当时多精度 LLM 量化的 SOTA 方法。其核心设计：(1) 采用聚类-based 非均匀量化，将权重矩阵按 K-means 聚类为若干 centroid，每个权重存储 centroid index；(2) 通过 Incremental Upscaling 实现多精度：从低精度开始，逐步分裂聚类中心（如 4 个 centroid → 8 个），使单个模型覆盖多种精度；(3) 推理时将权重以比特平面形式存储，按需加载 p 个比特平面，经比特转置（bit-transpose）重组为索引，再通过 centroid table lookup 获取反量化值后执行 GEMM。

  Baseline 全栈执行例子（Any-Precision LLM, 3-bit Llama-3.1-8B 推理）：
  - 算法pipeline：FP16 权重 → K-means 聚类为 2^3=8 个 centroid → 存储 centroid table + 3 个比特平面 → 3-bit 推理时加载 3 个比特平面 → bit-transpose 重组为 8 值索引 → table lookup 获取 FP16 centroid → GEMM。与 FP16 baseline 相比，任何精度下推理均可通过加载更少比特实现内存带宽节省。
  - 系统框架：PyTorch + 自定义 CUDA kernel，GPU 推理。
  - 编译框架：论文未明确说明。
  - kernel调度：自定义 CUDA kernel 实现 bit-transpose + centroid table lookup + GEMM。主要开销：bit-transpose（占 kernel 延迟 35-58%）和 centroid table lookup（占 9-17%）。
  - 硬件架构：论文未明确说明（标准 GPU 执行）。

  **Any-Precision LLM 的核心缺陷：**
  (a) **硬件不友好**：依赖 centroid table lookup，无法直接对二进制比特平面操作（bitwise operations 不适用于非均匀量化的 index 语义）。bit-transpose 和 table lookup 引入额外开销，限制了实际加速比。
  (b) **极低比特退化严重**：2-bit 时准确率急剧下降（MMLU=24.66 vs FP16=65.02，Wiki PPL=1680.77 vs FP16=6.24），有效可用范围仅限于 3-4 bit。
  (c) **非均匀量化的表达能力在低比特下无法发挥**：2-bit 仅 4 个 centroid，K-means 聚类无法充分捕获权重分布。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **AnyBCQ**，基于 Binary-Coded Quantization (BCQ) 的多精度量化框架，通过三个层次的设计解决 baseline 缺陷：

  **(1) BCQ 替代非均匀量化：实现硬件友好的比特平面直接操作（解决缺陷 a）**
  BCQ 将权重表示为 Ŵ = Σ α_i B_i（B_i ∈ {-1,+1}），而非 centroid index。每个比特平面 B_i 天然是二值操作数，可以直接与激活值进行加减运算（+α_i 或 -α_i），无需 centroid table lookup 和 bit-transpose。推理时仅加载所需 p 个比特平面，每个平面独立计算后按 α_i 缩放累加。这使 kernel 可以直接执行：`output = Σ_{i=1}^p α_i · (B_i ⊗ activation)`，其中 ⊗ 表示基于 LUT 的二值-浮点 GEMM。

  **(2) 渐进式精度扩展（Progressive Precision Expansion）：实现单调精度改善（解决缺陷 b）**
  从基础精度 p_L（如 2-bit）开始，逐比特扩展至 p_H（如 4-bit）。每次扩展时：(i) 冻结之前所有比特平面 B_1...B_{p-1}（共享二值表示），(ii) 从残差 R = W - Ŵ^{(p-1)} 中提取新比特平面 B_p = sign(R)，(iii) 通过最小二乘重新优化所有 α_i（而非重新优化 B_i）。这保证了 p-bit 模型的精度 ≥ (p-1)-bit 模型的精度（单调改善），因为新增比特平面总是捕获残差信息。配合 block-wise MRE（最小化重建误差）校准，2-bit 时 MMLU=35.32（baseline=24.66），Wiki PPL=19.01（baseline=1680.77）。

  **(3) 共享二值表示 + 独立 scale：内存高效的多精度存储（解决缺陷 c 的工程影响）**
  所有精度共享同一组比特平面 B_1...B_{p_H}，仅维护精度特定的缩放因子 {α_i^{(p)}}。因为比特平面占总存储的绝大部分（如 4-bit BCQ: 3.89GB binary vs 0.49GB scale），共享二值表示使多精度模型仅需 4.99GB（vs Multi-model 独立存储三个模型的 9.85GB，↓49%）。

  论文方法全栈执行例子（AnyBCQ, 3-bit Llama-3.1-8B 推理）：
  - 算法pipeline：FP16 权重 W → 基础精度 p_L=2：GREEDY(W) → T=20 交替优化（LS + BS）→ 精度扩展 p=3：冻结 B_1,B_2，初始化 B_3=0, α_3=0 → 从残差 R=W-(α_1B_1+α_2B_2) 提取 B_3=sign(R) → T=20 LS 优化 α → Block-wise MRE 校准 10 epochs → 最终模型：1 组比特平面 {B_1,B_2,B_3} + 3 套 scale {α_i^{(2)}}, {α_i^{(3)}}（2-bit 和 3-bit 各一套）。3-bit 推理：Ŵ = α_1^{(3)}B_1 + α_2^{(3)}B_2 + α_3^{(3)}B_3。
  - 系统框架：PyTorch + HuggingFace Transformers → lm-eval-harness v0.4.5 评估。校准集：C4 512 sequences。
  - 编译框架：论文未明确说明。
  - kernel调度：自定义 CUDA kernel：① 加载 p 个比特平面（packed binary）→ ② LUT-based GEMM（每个比特平面独立计算加减结果）→ ③ 乘 α_i 累加 → ④ 输出。消除 bit-transpose 和 centroid lookup 开销。GEMV 延迟：M=4096, K=14336, 2-bit: 315µs（×2.78 vs cuBLAS, Any-Precision LLM=356µs）。端到端吞吐：2-bit=245 tok/s vs Any-Precision LLM=228 tok/s vs FP16=105 tok/s。
  - 硬件架构：CUDA 12.6 + NVIDIA A100 80GB。论文讨论中提到 AnyBCQ 可部署到 BCQ 原生加速器（iFPU、FIGLUT）获得更大加速。

  关键设计动机映射：
  - Any-Precision LLM 非均匀量化无法直接操作比特平面（需 centroid lookup + bit-transpose）→ AnyBCQ 用 BCQ 二值表示，使计算简化为 {-1,+1} 的加减操作
  - 2-bit 精度退化严重（K-means 仅 4 centroid 不足以表达权重分布）→ MRE-based BCQ 在 2-bit 时已有 2 个比特平面（4 种组合值），且逐比特贪心优化更有效利用有限表达空间
  - 多精度模型存储冗余（多套独立模型）→ 共享比特平面 + 独立 scale 减少 49% 内存
  - Bit-transpose 和 LUT lookup 占 kernel 延迟 44-75% → AnyBCQ kernel 直接比特平面操作消除两项开销，换算为 1.07-1.17× 端到端吞吐提升
