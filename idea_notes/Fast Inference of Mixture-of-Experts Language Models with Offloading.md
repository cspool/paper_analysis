## Fast Inference of Mixture-of-Experts Language Models with Offloading

- baseline方法是什么？
  - **Naive offloading（HuggingFace accelerate device_map="auto"）**：标准的 per-layer offloading 方案。每个 Transformer 层（含所有 expert）作为一个整体，在需要时从 host RAM 完整加载到 GPU，用完后卸载。对于 MoE 模型，这意味着每次计算一个 MoE 层时，需要将全部 8 个 expert（每个约 5.6B 参数在 FP16 下）加载到 GPU，但实际只使用 top-2。
  - 全栈执行例子（以 Mixtral-8x7B-Instruct 在 T4 16GB 上 batch=1 为例）：
    - **模型推理算法层**：Mixtral-8x7B，32 层，每层 attention + MoE（8 experts, top-2 routing）。Per-token autoregressive decode，batch=1。
    - **系统框架层**：HuggingFace accelerate 的默认 offloading（`device_map="auto"`）。执行顺序：① 加载 layer 0 全部参数到 GPU（含 attention weights + 8 expert weights）→ ② 运行 attention → ③ 运行 MoE gate → ④ 运行 top-2 expert FFN → ⑤ 卸载 layer 0 全部参数 → ⑥ 加载 layer 1 全部参数...。每层加载全部 8 个 expert，但仅使用其中 2 个，浪费 75% 的 PCIe 带宽。
    - **编译框架层**：论文未明确说明。PyTorch + HuggingFace accelerate，无编译框架修改。
    - **kernel调度层**：论文未明确说明。标准 PyTorch CUDA kernel 执行，per-layer load → compute → unload 循环。
    - **硬件架构层**：NVIDIA T4 (16GB VRAM)，host RAM ~13-16GB，PCIe Gen.3 8-16GB/s。
  - **Baseline 痛点**：
    1. **无效参数传输**（核心痛点）：Naive offloading 以层为单位加载，但 MoE 每层只需 2/8 expert。每次加载浪费 75% 的 host-to-device 带宽传输不需要的 expert。
    2. **无缓存复用**：相邻 token 常复用相同 expert（图 1 显示 expert 局部性），但 naive offloading 每层都从 host RAM 重新加载，无状态记忆。
    3. **无法预取专家**：MoE gate 在当前 layer 输出后才选择 expert，无法像 dense 模型那样预先加载下一层参数。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：三项针对 MoE 特性的 offloading 优化：
    1. **Expert LRU Cache**（解决痛点 1 和 2）：以 expert 为单位（而非层）进行 offloading，保持每层 k 个最近使用的 expert 在 GPU 上。仅加载 cache miss 的 expert，大幅减少 host-to-device 传输量。通过观测到 MoE 模型在相邻 token 间存在 expert 复用局部性（图 1 蓝色方块），LRU 策略简单但有效。
    2. **Speculative Expert Loading**（解决痛点 3）：利用 Transformer 残差连接的归纳偏置——当前层 hidden states 可作为下一层 hidden states 的合理近似。在对当前 token 计算的同时，将下一层 MoE gate 应用于当前 hidden states 预测下一层最可能使用的 1-2 个 expert，在独立的 CUDA stream 上后台异步预取。预测正确时可消除下一层的加载延迟；错误时仅浪费带宽但正确性不受影响。
    3. **混合量化（Mixed MoE Quantization）**：attention 层保持 4-bit（高质量），expert 层压缩到 2-3 bit。利用 expert 占总参数 96.6% 这一特性，显著缩小 offloading 传输量，同时保持 perplexity 可接受（2-bit experts 下 WikiText2 从 3.59→4.52）。

  - 全栈执行例子（与 baseline 同配置，Full algorithm + 2-bit experts）：
    - **模型推理算法层**：与 baseline 相同（Mixtral-8x7B-Instruct，top-2 routing），不改变模型架构、gate 逻辑或生成质量。
    - **系统框架层**：自建 offloading 系统替代 accelerate。执行顺序：① 加载 layer l 的 attention（常驻 GPU）→ ② MoE gate → ③ **检查 expert cache**，仅加载 cache miss 的 expert → ④ **异步启动投机预取**（对 layer l+1 预测所需 expert，后台 host-to-device copy）→ ⑤ 运行 top-2 expert FFN → ⑥ 进入 layer l+1 时投机预取结果可能已就绪。与 baseline 的关键区别：仅加载 0-2 个 expert/token/layer（而非 8 个），且下一层加载与当前层计算重叠。
    - **编译框架层**：论文未明确说明。纯 PyTorch 实现，使用 pin_memory + CUDA stream 异步拷贝。
    - **kernel调度层**：Contiguous pinned memory buffer 实现单次 host-to-device copy。b=4 个共享 device buffer 实现异步 expert 交换。CUDA stream 层面：计算 stream 执行 attention + expert FFN，拷贝 stream 执行投机预取。使用 `tensor.pin_memory()` + `cudaMemcpyAsync` 模式。
    - **硬件架构层**：与 baseline 相同。关键约束仍为 PCIe 带宽（8-16GB/s），瓶颈从"加载 8 个 expert"变为"加载 0-2 个 expert + 后台预取"。结果：T4 上从 0.66 tok/s（naive）提升到 2.09 tok/s（full algo），3.2× 加速。

    核心设计洞察：该工作是一种"特性感知 offloading"——不改变模型，而是利用 MoE 的两大固有特性（expert 局部性和残差连接的 gate 预测能力）来优化 offloading 调度。相比通用的 per-layer offloading，这种 MoE-aware 调度在 consumer hardware 上实现了可交互的推理速度（2-3 tok/s），使 47B MoE 模型能在免费 Colab 上运行。
