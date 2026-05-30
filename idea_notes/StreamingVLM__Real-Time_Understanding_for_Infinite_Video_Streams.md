## StreamingVLM__Real-Time_Understanding_for_Infinite_Video_Streams

- baseline方法是什么？
  Baseline 是现有 VLM 处理长视频的三种方式：
  
  **(a) Full Attention**：对视频全部帧计算 full causal attention。O(T²) 计算复杂度，内存无界，视频超出训练长度后性能退化。执行例子：Qwen2.5-VL-7B-Instruct 对完整视频做 full attention → 一次自回归生成 caption，所有 vision tokens 在所有层 attend 到所有历史 tokens。2-5 分钟后超出 training context length，latency 急剧上升直至 OOM。
  
  **(b) Sliding Window Attention (w/o Overlap)**：将长视频切分为固定长度 chunk，每个 chunk 内独立处理，上下文在 chunk 边界重置。执行例子：LiveCC-7B-Instruct 在 chunk 模式下，每 100s 视频重置 KV cache，只读入当前 chunk 的 frames + 前序 text 作为 prompt 生成解说。短 chunk 破坏跨块连贯性（coherence），长 chunk 则延迟高且仍会超出 training length（Figure 6）。
  
  **(c) Sliding Window Attention (w/ Overlap)**：相邻 chunk 有重叠，window 滑动时 recompute 重叠部分的 attention。执行例子：维护固定长度窗口（如 100s），每次新帧到来重新计算窗口内全部 tokens 的 attention，KV cache 不跨窗口复用。维持了 coherence 但 computation redundancy 严重，latency 高且不稳定（Figure 7 中 SI. w/ Overlap 曲线）。
  
  训练无关的 KV cache 驱逐方法（如 ReKV）：训练时使用 full attention，推理时强行驱逐 tokens 会破坏模型期望的 attention pattern，对 fine-tuned 模型常常导致无输出（Table 2）。
  
  全栈执行例子（以 Sliding Window w/o Overlap 为例）：
  - 模型推理算法层：VLM（Qwen2.5-VL / LiveCC）将视频按 100s chunk 切分，每个 chunk 独立做 full attention，chunk 间通过 previous text 传递上下文 → 自回归生成解说。没有 KV cache 跨 chunk 复用，跨块长程依赖丢失。
  - 系统框架层：论文未明确说明。标准 VLM 推理 pipeline，逐 chunk 调用模型。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。GPU 上标准 FlashAttention，无自定义 kernel。
  - 硬件架构层：论文未明确说明。在 H100 GPU 上运行。
  
  核心缺陷：(1) **缺乏训练-推理一致性**：训练时用 full attention，推理时用 sliding window，模型遭遇 distribution shift；(2) **计算冗余**：overlap 模式反复 recompute attention，w/o overlap 模式丢失上下文连贯性；(3) **位置编码漂移**：native RoPE 索引随视频增长超出训练范围，导致 long-horizon 性能退化；(4) **不支持真正的无限流式输入**：现有方法处理有限长度 video clip，无法应对 continuous/infinite 视觉流。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **StreamingVLM** 通过统一训练-推理框架 + KV cache 复用 + contiguous RoPE 解决上述四个缺陷：

  **(1) 训练-推理一致性：Overlapped-Chunk Full-Attention Training**
  训练时将长视频切为 W=24s 的 overlapped chunk（O=12s），每个 chunk 内做 full attention，vision/text tokens 以 1s 间隔交错排列（而非传统 VLM 的 vision-then-text 布局）。这种 overlapped full-attention 的 effective attention pattern 与推理时的 "sink tokens + 近期 text 长窗口 + 近期 vision 短窗口" 高度近似（Figure 4 右侧），使模型学到 recency bias 而非跨 chunk 的突然重置。只在 text position 计算 loss，并在无解说词的秒插入占位符 "..."，训练模型同时学会"何时说话、何时沉默"的流式行为。

  **(2) 计算效率：Streaming-Aware KV Cache + Asymmetric Retention**
  推理时维护紧凑 KV cache，复用历史 KV 而非 recompute。非对称保留策略：attention sink tokens（Tsink=512，system prompt + 早期 text）保证 attention 稳定性；近期 text 长窗口（Twindow=512）保留长期语言记忆；近期 vision 短窗口（Vwindow=16s）跟踪连续动作。旧 vision tokens 优先驱逐（视觉冗余高），旧 text 仅在超 budget 时驱逐（语义信息密度高）。此设计消除了 Sliding Window w/ Overlap 的重计算，使 latency 保持低且稳定（Figure 7），单 H100 达到 8 FPS 实时性能。

  **(3) 位置编码稳定性：Contiguous RoPE**
  当旧 tokens 被驱逐后，后续及新 tokens 的 RoPE 位置索引左移，保持与最后保留 token 的数值连续性。当视频长度超出总窗口尺寸后，effective RoPE 索引停止增长，保持有界。这使位置编码始终在训练分布内，防止 long-horizon 性能退化。Ablation（Table 4）显示 native RoPE 在 infinite stream 上急速退化（win rate 25.09 vs. GPT-4o），而 contiguous RoPE 维持 66.18%。对 Qwen-VL 的 3D RoPE（time, height, width），同样应用 contiguous 左移规则。

  **(4) 无限流式输入：数据 + 推理协同设计**
  构建 Inf-Streams 数据集（>4000 小时体育解说 SFT + 14K 高质量 annealing 样本）和 Inf-Streams-Eval benchmark（20 场完整比赛，平均 2.12h，per-second 帧-文本对齐），验证真正无限流式理解能力。推理时按 1s 步进接收新视觉帧，KV cache 增量更新，自回归生成解说或等待下一个信息 token，实现 closed-loop streaming behavior。

  全栈执行例子（对比 baseline）：
  - 模型推理算法层：Baseline → VLM full attention 或 sliding window 逐 chunk 处理。StreamingVLM → SFT 后的 Qwen2.5-VL 接收每 1s 新帧 → vision encoder 编码为 V_new tokens → interleave text tokens → contiguous RoPE 计算 bounded 位置索引 → attention 计算时 Q 仅 attend 到 KV cache 中保留的 sink+text_window+vision_window tokens（复用历史，不 recompute）→ KV cache 按 asymmetric policy 驱逐 → 自回归生成解说（无解说时输出 silence placeholder）→ 循环。训练时 overlapped chunk（W=24s, O=12s）内 full attention 的 attention pattern 与推理时 sink+sliding window 的 effective 注意力模式一致。
  - 系统框架层：论文未明确说明。基于 Qwen2.5-VL 的推理管线，修改 token interleaving layout + KV cache management。
  - 编译框架层：论文未明确说明。
  - kernel 调度层：论文未明确说明。使用标准 FlashAttention，GPU 上 bfloat16 推理。
  - 硬件架构层：论文未明确说明。训练 128 H100-days，推理单卡 H100。

- baseline方法是什么？
  Baseline包括两类方法：
  (a) **DIRECT范式**：给定采样的视频帧，通过单轮推理（single sequence prediction）直接输出最终答案。代表：Qwen3-VL系列（直接将128帧concat后一次性输入MLLM自回归生成答案）、Video-R1（GRPO训练但仅用option-matching和ROUGE作为reward）、VideoRFT（semantic-consistency reward）、LongVILA-R1（用序列并行支持数千帧的RL训练，但同样用string-matching reward）。这些方法的核心局限：对所有video duration一视同仁——短视频和2小时长视频都用相同计算量处理，且依赖string-matching reward导致在open-ended问题上无效。
  (b) **AGENT范式（Over-reliance on Temporal Grounding）**：多轮agent系统（VideoAgent, VideoMind, VideoExplorer, LVAgent等），核心依赖temporal grounder在整个视频范围内迭代定位事件。局限：(i) 缺乏鲁棒的长视频时序定位模型，(ii) 系统设计中缺乏知识驱动推理能力，(iii) 过度工程化于MCQ问题导致open-ended性能差。

  Baseline全栈执行例子（以Qwen3-VL-8B-Instruct DIRECT模式，128 frames输入为例）：
  - 算法层：视频→均匀采样128帧(2 FPS)→每帧ViT编码→temporal pooling(factor=2)→约128×min_tokens=16384 visual tokens→LLM单轮自回归生成答案。不区分短视频和长视频的处理方式，对2小时视频只采样128帧导致无法回答需要时间定位的问题。
  - 系统框架层：PyTorch模型 + vLLM serving。无专用调度。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention标准kernel。无自定义kernel。
  - 硬件架构层：NVIDIA H100 GPU集群。DIRECT RL baselines（Video-R1等）也需要多GPU训练。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SAGE通过三方面设计解决Baseline的缺陷：

  **(a) 缺陷1：DIRECT方法对所有时长一视同仁，缺乏自适应推理能力** → Any-Horizon Agent System
  SAGE将统一单轮推理改为自适应Agent系统。SAGE-MM（orchestrator VLM）根据任务难度和视频时长自主决定：简单问题单轮直接回答，复杂/长视频问题通过多轮tool calling逐步推理。Nmax=11限制步数。关键设计：
  - Stage-1（Context VLM）：一次性产出video_context + query_intent + 首步action
  - Stage-2（Iterative Reasoner）：迭代判断answerable，每步产出tool call或final answer
  - 5 tools + analyze tool提供多种信息获取渠道，不依赖单一temporal grounding

  **(b) 缺陷2：AGENT系统过度依赖temporal grounding，缺乏知识驱动推理** → Knowledge-Driven Multi-Tool System
  现有AGENT系统几乎完全依赖temporal grounder在整个视频中定位事件。SAGE引入web-search和speech transcription工具，使系统能利用外部知识和语音信息智能缩小搜索空间。案例：知道F1 2024赛季排名后，看2025 livery reveal视频时可以推理出目标片段的大致时间区间。实验验证（Table 10）：去除web-search/parse-website导致overall降2.5%，去除transcribe-speech降5.5%（verbal问题降36.5%）。

  **(c) 缺陷3：现有RL recipe依赖string-matching reward，对open-ended问题无效** → Multi-Reward GRPO + LLM-as-Judge
  Video-R1/VideoRFT/LongVILA-R1等用option-matching和ROUGE作为reward，只能处理MCQ问题。SAGE使用：
  - GPT-4o作为LLM-Judge判断答案语义正确性（binary correctness verdict）
  - 多层reward设计：format(+0.05/-0.10) + reasonable-tool(+0.10/-0.10) + args-repeat(-0.05×√rep) + args-valid(-0.1/0) + accuracy(-2.0~+1.25)
  - 正确+使用visual tools额外+0.25奖励（鼓励视觉信息利用）
  - 前100步Nmax=6稳定训练，防止RL初期因长trajectory方差过大导致不收敛
  结果：RL后SFT模型改善4.1%（Qwen3-VL-4B，Table 7），open-ended从51.1%→57.4%。

  对比Baseline的全栈执行例子（SAGE, Qwen3-VL-8B-Instruct SFT+RL SAGE-MM）：
  - 算法层：视频→128帧采样→SAGE-MM接收T|F|Q|M→Stage-1输出video_context + tool_call → 执行tool（如transcribe 2分钟segment, web search）→ Stage-2迭代（平均1.74-3.54 turns根据视频时长）→ final answer。对于短视频（<60s），单轮直接回答率更高（Tab 9）。对于长视频（>600s），平均2.49 turns的multi-turn推理。
  - 系统框架层：vLLM serving所有模型。Tool执行链：Serper Google Search API（web-search）+ Whisper-large-v3（transcribe-speech）+ Qwen3-VL-30B-A3B-Instruct（ground-event & analyze）。推理耗时8.6s/sample，远快于VideoMind 24.7s和VideoAgent 1445s。
  - 编译框架层：论文未明确说明。
  - kernel调度层：标准FlashAttention，无自定义kernel。直接复用vLLM PagedAttention。
  - 硬件架构层：NVIDIA H100 GPU集群（训练16×H100）。冻结visual encoder和projector，仅训练LLM部分。

- baseline方法是什么？
  （第二组baseline视角——合成数据生成方法）现有的长视频QnA数据生成方法（如LongVILA、Eagle 2.5）采用bottom-up pipeline：将视频切割为10-30秒的subclip，分别用模型处理生成caption或QnA pairs，再聚合。对1小时视频需处理120个subclip，单个subclip 10秒即需20分钟。

  Paper方法：一次性利用Gemini-2.5-Flash的长上下文能力（支持2小时视频），通过carefully designed prompt直接生成10-20个覆盖全视频时间跨度的QnA pairs。通过percent_video_parsed字段强制模型按时间顺序生成并覆盖至少90%视频。成本约为人工标注的1/100，时间约为subclip pipeline的1/10。人工验证1700+样本仅5%错误率。

- baseline方法是什么？
  Baseline 是传统的**固定分辨率全量编码**（Vanilla）以及两类主流后编码压缩方法：
  (a) **Model-side compression**：在视觉编码后对 token 进行剪枝或合并（ToMe token merging, VisionZip attention-guided pruning, FlashVid spatiotemporal tree-based merging）。这些方法"接受编码器的全分辨率输入作为固定成本"，先付出全部计算再试图压缩，一旦细粒度证据被丢弃就无法恢复。不规则 token 布局还会破坏 FlashAttention 和 vLLM 等优化内核的兼容性。
  (b) **Output-side agentic reasoning**：通过迭代检索或缩放步骤多次调用 backbone（VideoAuto-R1 等）。虽可恢复覆盖范围，但每步检索需要独立 backbone 调用，首轮粗视图常欠采样目标证据。

  Baseline（Vanilla Qwen2.5-VL-7B，32 frames 全分辨率）全栈执行例子：
  - 算法层：视频 T=32 frames → 每帧固定 res 448×448 → ViT pathify (P=14) → 32×32×32=32768 visual tokens → LLM backbone 自回归生成答案。计算代价与像素量二次方关系，但回答复杂查询所需的证据在时间上高度稀疏。
  - 系统框架层：PyTorch + vLLM/SGLang for serving。无专用 Serving 框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：FlashAttention 标准 kernel，无自定义 kernel。但 model-side 压缩方法产生的不规则 token 布局会破坏这些优化内核。
  - 硬件架构层：NVIDIA H100 GPU 集群（训练 32×H100），推理在 4-GPU vLLM engine。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ResAdapt 通过**输入侧自适应（Input-side Adaptation）**原则将干预点从"编码后"移到"编码前"，用 RL 训练的 Allocator 预测 query-aware 的每帧分辨率预算，解决了 Baseline 的三大缺陷：

  **(a) 缺陷1：编码后的压缩无法恢复已丢失的细粒度证据** → 输入侧自适应分配
  Model-side 方法（ToMe/VisionZip）在 32 帧全分辨率编码后才剪枝 token，此时高分辨率编码的计算成本已全部付出，且被剪枝的细粒度证据无法恢复。ResAdapt 的 Allocator 在编码前接收粗粒度特征 + query，预测每帧缩放因子 st ∈ [0.2, 1.8]，只让 backbone 处理 resize 后的像素。在 ~10% retention 时，ResAdapt 在 VideoMMMU 上达 45.7，显著优于 ToMe 的 39.2 和 VisionZip 的 39.1。关键设计：
  - Beta 分布参数化保证连续动作空间：s_t 从 Beta(α_t, β_t) 采样后线性映射到 [s_min, s_max]
  - 分配策略完全兼容 FlashAttention、vLLM、SGLang，无需定制 kernel

  **(b) 缺陷2：Naive accuracy-cost penalty 导致策略崩溃** → CAPO 非对称 reward shaping
  若直接使用 Lagrangian R = Q(x,y) − λC(s)，策略会无条件向最小预算崩溃——任何成本降低都获得等量奖励，无论答案质量。CAPO 通过三项机制克服：
  - Dynamic cost pivot：τ_dyn = κ_mix·c̄_group + (1−κ_mix)·τ_fix，同时提供局部比较基线和全局压缩锚点
  - Asymmetric shaping：正确+低成本 → 中等奖励 λ_+；错误+高成本 → 强惩罚 λ_−（λ_− > λ_+ > 0）。这种不对称性是防止崩溃的核心——降低成本的激励只在保持正确性时才有效
  - 对正确 rollout 施加正下限 ε_+ > 0，确保正确低成本的 rollout 始终获得正向学习信号

  **(c) 缺陷3：相邻冗余帧上的均匀分配浪费预算** → Temporal Similarity Regularizer
  没有 L_sim 时，Allocator 对视觉相似的相邻帧分配近乎相同的 scale。L_sim 通过余弦相似度门控权重 w_t = σ((cos(f_t, f_{t+1}) − τ_sim)/γ_sim) 激活，惩罚冗余联合高预算分配，迫使策略区分视觉相似的相邻帧。消融实验（Figure 7）显示：去除 L_sim 后 scale trace 坍缩为接近 FixedScale 的常数分布；恢复 L_sim 后恢复锐利的帧级分化。

  对比 baseline 的全栈执行例子（ResAdapt, Qwen2.5-VL-7B, T=32 frames, ρ≈11%）：
  - 算法层：视频 T=32 frames → SmolVLM 轻量编码器提取粗粒度特征 [T, 1024] → Transformer decoder（时序self-attn + gated cross-attn to query）→ Beta 头预测 (α_t, β_t) → 采样 s_t ∈ [0.2, 1.8] → bilinear resize 每帧 → backbone 接收约 3604 visual tokens（vs 32768）→ 单次自回归生成。CAPO 训练中，GRPO 循环：M=16 allocations × N=1 rollout → CAPO advantages → 交替更新 Allocator（PPO clip + L_sim + L_con）和 Backbone（token-level PPO）。
  - 系统框架层：VeRL + DeepSpeed ZeRO + vLLM 分布式训练（32×H100）。推理时单 GPU Allocator + 4-GPU vLLM engine。在 128 frames, R≈28% 时，E2E 延迟从 4877ms 降至 1977ms（−59.5%）。
  - 编译框架层：论文未明确说明。ResAdapt 保持 backbone 的 native token 接口，无需编译框架修改。
  - kernel调度层：完全兼容 FlashAttention 和 vLLM PagedAttention，无自定义 kernel。与需要处理不规则 token 布局的 model-side 方法形成对比。
  - 硬件架构层：NVIDIA H100 GPU（训练 32×H100）。推理延迟节省在长序列下最显著——128 frames + ~28% retention 时 attention FLOPs 降低为 ρ² ≈ 0.012（~83×），叠加 Allocator 固定开销后仍有净加速。空间预算节省还可 reinvest 为时间覆盖：在同计算量下可处理 16× 更多帧，长视频推理相对增益 >15%。

- baseline方法是什么？
  Baseline是传统frame-centric密集视觉编码器（以SigLIP2为代表），其核心假设是：视频中的所有空间patch和时间帧同等重要，因此需要均匀密集处理。

  Baseline（以SigLIP2 ViT-L/16 dense帧采样为例）全栈执行例子：
  - 算法层：视频→均匀8帧采样（约sparse temporal sampling）→每帧dense patchify（16×16 patches, 256 patches/frame）→ViT编码全部256×8=2048个patches→attentive pooling聚合→class embeddings。所有空间区域（前景/背景、运动/静止）以相同计算量处理。问题：大量计算（>75%）浪费在静态背景和不变区域，且8帧稀疏采样可能完全错过关键瞬时动作（如短暂倾倒、击球瞬间）。
  - 系统框架层：PyTorch + HuggingFace Transformers + Flash Attention。无专用Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Flash Attention 2标准实现。
  - 硬件架构层：NVIDIA A800/H100 GPU集群。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  OV-Encoder通过**Codec Patchification**将编解码器的信息论分解原理引入ViT设计，根本性地将"密集均匀计算"改为"只编码信息熵高的patch"，解决了baseline的三大缺陷：

  **(a) 缺陷1：密集计算浪费** → Codec Patchification
  传统方法对所有空间patch平均分配计算，背景静态区域和运动前景区域同等处理。OV-Encoder利用HEVC的运动矢量（motion vectors）和预测残差（residuals）作为patch级信息熵的代理信号，仅在64帧密集输入中选择3.1%-25%的显著patch（motion rich + high residual），剩余87.5%-96.9%的patch完全跳过。关键设计：
  - I/P-frame分解：每GOP（32帧）保留1个I-frame全量编码（建立完整空间上下文）+ 31个P-frame稀疏编码（仅运动显著区域）
  - Clip-level全局Top-K选择：不按帧独立选patch，而是在整个64帧clip中全局排序显著性，确保token budget最优分配到真正需要的时间位置
  - 可视patch indices机制：被跳过的patches通过visible_indices标记其时空位置用于3D-RoPE，保留temporal coverage

  **(b) 缺陷2：均匀采样导致关键帧丢失** → 密集时间覆盖
  传统方法均匀采样8帧（限制于token budget）意味着采样间隔内发生的快速/短暂动作可能完全丢失。OV-Encoder保留全部64帧，但P-frames仅取显著区域，实现"时间密集、空间稀疏"——例如Diving场景中，64帧覆盖连续的pose transitions，而8帧均匀采样可能跳过来回翻滚的过渡姿态。Case study 1和2论证了连续运动场景和离散关键帧场景下codec采样的优势。

  **(c) 缺陷3：缺乏语义结构化** → 百万级聚类判别
  传统contrastive learning（CLIP/SigLIP）用instance-level discrimination + text supervision，无法建模intra-class consistency和fine-grained inter-class relationship。OV-Encoder用frozen metaCLIP提取嵌入，k-means聚类为2M图像类中心和400K视频类中心，用multi-label sigmoid BCE监督，同时建模物体级（object-level）和动作级（motion-level）语义，无需外部语言监督。

  对比baseline的全栈执行例子（OV-Encoder Codec, 64 frame input, budget=2048）：
  - 算法层：原始视频→HEVC解码提取motion vectors + residuals→按patch聚合为saliency score→全局Top-K选择2048个显著patches（512来自2个I-frame所有patches + 1536来自62个P-frame最显著patches）→3D-RoPE编码时空位置→24层ViT编码（Flash Attention 2）→attentive pooling聚合→image branch对比2M object centroids / video branch对比400K motion centroids→sigmoid BCE loss。保留密集64帧时间覆盖的同时token减少87.5%。
  - 系统框架层：PyTorch + Flash Attention 2。128×A800 GPUs（16 nodes × 8）分布式训练。无Serving框架修改。
  - 编译框架层：论文未明确说明。
  - kernel调度层：Flash Attention 2标准实现，无自定义kernel。Codec处理（motion vector extract + residual decode）在CPU上进行。
  - 硬件架构层：NVIDIA A800 128 GPU预训练（Stage 1: 13B samples; Stage 2: 4B samples）。Attention probing on 8×A800。
