## VideoLLaMB: Long-context Video Understanding with Recurrent Memory Bridges

- baseline方法是什么？
  Baseline 是 **PLLaVA**（基于 Adaptive Pooling 的视频压缩方法）以及现有视频压缩策略（sampling、aggregation、semantic consolidation、resampling、video segmentation）。

  全栈执行例子（以 PLLaVA + Vicuna-7B 为例）：
  - 模型推理算法层：视频 V={v_1,...,v_n} → ViT-L/14 逐帧提取特征 → Adaptive Pooling 将 n 帧特征池化到固定 M 帧（如 16 帧）→ Linear Projector 投影 → Vicuna-7B 生成答案。核心机制是通过 pooling 丢弃部分帧信息来压缩视频长度。训练数据为 Video-LLaVA 视频数据 + LLaVA-1.5 图像数据，16 帧训练，推理可使用 32 帧（但需复用 16 帧训练权重）。EgoSchema 上 PLLaVA 16→16 帧仅 45.6%，32→16 帧降至 43.8%。
  - 系统框架层：基于 LLaVA-1.5 初始化，Video-LLaVA 的 image/video encoder，无 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，无自定义 kernel。
  - 硬件架构层：4× NVIDIA A800 GPU（训练），单 A100/A800（推理），无硬件定制。

  核心缺陷：
  (1) **Pooling 导致关键视觉信息丢失**：Adaptive pooling 将 n 帧强制压缩到固定数量，无论原始视频长度多少，必然丢弃大量视觉细节。在 NIAVH benchmark 中，pooling 策略在 haystack:needle 比例较高时几乎无法定位目标帧（Figure 3a），因为 needle 帧的视觉信息被 pooling 平均化淹没了。
  (2) **位置外推能力差**：基于 position extrapolation + sampling 的方法（如 LLaVA-NeXT-Video-DPO）在训练时仅见 32 帧，推理时对长于训练的序列预测能力急剧下降（Figure 3b），无法有效利用更长视频提供的信息。
  (3) **视频分割破坏语义流**：将视频均匀或按时间切分为短片段的方法（如 VideoStreaming、Video-XL）切断了跨片段的语义连贯性，使模型难以理解跨越场景边界的因果和时序关系。
  (4) **Resampler 压缩容量有限**：MA-LMM 等方法使用 resampler + memory consolidation，但 resampler 的压缩比固定，超长视频的信息编码最终受限于 resampler 的容量瓶颈（Figure 3c）。
  (5) **GPU 显存随视频长度线性增长**：无压缩方法（如 LongVA）直接将所有帧的视觉 token 送入 LLM，显存开销与帧数成正比，限制了可处理的视频长度。PLLaVA 通过 pooling 缓解但信息损失严重。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **VideoLLaMB** 通过三个协同设计的模块解决上述缺陷：

  **(1) SceneTiling 语义分割 → 解决缺陷 (3)**

  Baseline → 均匀切分破坏跨场景语义连贯性。
  VideoLLaMB → 基于 ViT [CLS] token 帧间余弦相似度计算 depth score $d_i = (cl_i + cr_i - 2c_i)/2$，按 μ+α·σ 阈值自适应检测语义边界，将视频划分为 K 个语义独立的 segment。每个 segment 内部帧高度相关，跨 segment 边界处为语义转换点，此处的信息损失不破坏语义完整性。消融实验（Table 8）显示替换为 uniform segment 后性能下降 1.8 点（EgoSchema: 53.8→52.0）。

  **(2) Recurrent Memory Bridge Layers → 解决缺陷 (1)(4)**

  Baseline → Adaptive pooling 丢弃信息；Resampler 有压缩容量上限。
  VideoLLaMB → 单层 Transformer Bridge Layer，每个语义段前 prepend 32 个 memory tokens，通过 self-attention 将当前段信息压缩入 memory tokens：$[m_{i+1}; o_i] = \text{BridgeLayer}([m_i; s_i])$。memory tokens 递归遍历所有语义段，逐步累积全视频信息，**不主动丢弃任何帧**。最终 LLM 仅接收 memory-token-augmented 的当前段视觉表示，输入 token 数固定（约 32+M 个 token per step），GPU 显存线性缩放（Figure 4），支持 320 帧处理（仅训练于 16 帧）。消融实验（Table 8）：移除 recurrent 机制（mean pooling）→ 51.61%（-2.19），移除（adaptive pooling）→ 49.4%（-4.4）。

  **(3) Memory Cache with Retrieval → 解决缺陷 (2)(4)**

  Baseline → 位置外推能力差；长视频中早期信息被遗忘。
  VideoLLaMB → 维护 MemoryCache = [m_1, ..., m_i]，以当前 m_i 为 query、历史 memory cache 为 key/value 进行 cross-attention 检索：$m_{i+1} = \text{Softmax}(W_i^Q m_i (W_i^K M_i)^\top / \sqrt{d_k}) W_i^V M_i$。实现 BPTT 绕过：梯度通过检索路径传播（仅到当前几步），避免 RNN 式的长时间反向传播梯度消失。在 NIAVH 320s 测试中（Figure 3d），VideoLLaMB 在 depth=12 处（needle 在视频 30% 位置）仍然保持最高分数 5.73（vs LLaVA-NeXT-Video-DPO 1.72, PLLaVA 1.82）。消融实验（Table 8）：移除 retrieval → 52.2%（-1.6）。

  全栈执行例子（VideoLLaMB + Vicuna-7B）：
  - 模型推理算法层：
    训练阶段 → 16 帧视频 → ViT-L/14 编码 → SceneTiling 分为 4 段 → 初始化 32 个 memory tokens → Bridge Layer（单层 Transformer, 8 heads, hidden=1024）逐段递归处理 → Memory Retrieval（单层, 8 heads, hidden=1024）更新 memory → Projector → Vicuna-7B 生成。仅训练 Bridge Layer + LLM（LoRA），冻结 ViT。Learning rate=2e-4, batch=8, epoch=1, warmup=0.03，4×A800 GPU 训练。推理阶段 → 支持动态段数（SceneTiling 自适应）或静态 4 段 → 处理 320 帧时 GPU 显存线性增长，4.21s 推理 300s 视频（Table 6），2.3s 用于特征处理 + 1.91s 用于生成。3000s 视频需 31.5s（23.4s 特征 + 8.1s 生成，Table 7）。
  - 系统框架层：基于 LLaVA-1.5 初始化，Video-LLaVA image/video encoder。开源代码包含 CLI、streaming、Gradio demo。streaming 模式下 SceneTiling 仅用左侧相似度 $d_i = (cl_i - c_i)/2$ 实时检测边界，无需预知全视频。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：标准 PyTorch + HuggingFace Transformers，无自定义 kernel。Memory Cache 随视频长度线性增长但 token 数极少（每段仅 32 个 memory token），额外显存可忽略。
  - 硬件架构层：训练 4× NVIDIA A800 GPU，推理单 A100 (80GB)/A800 GPU。处理 320 帧仅需单卡 A100，显存线性缩放。

  对比 baseline 的解决效果：
  | 缺陷 | Baseline 表现 | VideoLLaMB 解决方式 |
  |------|-------------|-------------------|
  | Pooling 信息丢失 | PLLaVA EgoSchema 45.6%，NIAVH 1.82 | VideoLLaMB EgoSchema 53.8%（+8.2），NIAVH 5.73（+3.91） |
  | 位置外推差 | LLaVA-NeXT 32→长视频准确率骤降 | 16帧训练→320帧推理性能保持（Figure 2 dynamic segments） |
  | 语义流破坏 | uniform segmentation | SceneTiling 语义分割，+1.8 点 vs uniform |
  | 压缩容量瓶颈 | Resampler (MA-LMM) NIAVH 3.39 | Recurrent memory 无损累积，NIAVH 5.73 |
  | 显存线性增长 | LongVA 全 token 输入显存随帧数增长 | Memory Bridge 固定输入长度，显存线性缩放但仍可控 |
  | 推理时间 | MA-LMM 14.5s, MovieChat 143.7s (300s video) | VideoLLaMB 4.21s（压缩视觉输入使 LLM 处理时间最短） |
