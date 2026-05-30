## CurveStream__Boosting_Streaming_Video_Understanding_in_MLLMs_via_Curvature-Aware_Hierarchical_Visual_Memory_Management

- baseline方法是什么？
  Baseline 方法是现有的流视频记忆管理方法（多应用于无限长 streaming video 场景，统一受限于 token economy，固定 GPU 内存预算下必须持续管理视觉记忆队列），主要包括两类：(1) 视觉信息保持策略（Visual Retention）—— Uniform Sampling（均匀采样 1fps 或 64fps 下行采样）、Cosine Similarity-based selection（帧间余弦相似度做低层次筛选）、Optical Flow / Pyramid Optical Flow（密集光流计算物理运动强度）；(2) 记忆管理机制（Memory Management）—— HERMES（规则驱动的 KV cache 逐出，使用滑窗和时空冗余度量被动丢弃旧 token）、FreshMem（frequency-space hybrid memory，频率域混合记忆）、ReKV（外部存储 + post-hoc query-driven retrieval，延迟查询驱动的特征检索）。
  
  Baseline（Qwen2.5-VL-7B + 1fps uniform sampling, StreamingBench 场景）全栈执行例子：
  - 算法层：无限长流视频持续输入 → 每 1 秒均匀采样 1 帧 → 固定窗口内保留最近 64 帧 → Vision Encoder 编码为 visual tokens → 与 text query 拼接送入 LLM decoder → 当窗口满时 FIFO 丢弃最旧帧 → LLM 生成答案。整个过程对所有帧一视同仁，不区分帧的语义信息价值。对于长时间段中稀缺但关键的事件（如一个短促的动作变化），uniform sampling 可能在大量静态背景帧（如长时间观察静止物体）上浪费 token 配额，导致关键帧在 FIFO 驱逐中被挤压出去（catastrophic forgetting）。
  - 系统框架层：论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单张推理 GPU，论文未明确 GPU 型号和硬件细节

  Baseline 的缺陷：
  1. **语义碎片化（Semantic Fragmentation）**：Uniform sampling 或 cosine similarity 等低层次物理度量缺乏内在语义感知——均匀选择或基于像素相似度过滤的帧难以保持上下文连贯性。例如长时间固定场景中，uniform 可能在语义不变时持续积累冗余帧，而真正的语义突变帧（如新物体突然出现）可能因队列刚满被逐出。
  2. **信息模糊化（Information Blurring）**：无差别的 Feature Compression（如 HERMES 对不活跃 token 的全局压缩）不可逆地模糊了短暂但关键的语义转换点——这些转换点是因果推理链的核心锚点。例如在动作识别中，一个 0.5s 的"闪避"动作可能被压缩为模糊背景区域的一部分。
  3. **延迟感知（Delayed Perception）**：ReKV 和类似 retrieval 机制的框架依赖 post-hoc 查询在外部存储中检索相关帧——这在无限流场景中本质上是被动且滞后的，限制了对未知事件的实时主动感知能力。
  4. **对局部噪声过度敏感**：Cosine similarity 和 optical flow 等物理度量在动态场景中易受局部运动噪声影响——平滑的相机平推（背景全局大位移）和真正的语义突变（新物体进入/动作边界）在物理度量下表现相似，导致选择性模糊或过度保留无关帧。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法：CurveStream 通过特征流形几何曲率驱动的层级化视觉记忆管理解决上述问题：
  (1) **Curvature-Aware Scorer (CAS)**：使用冻结的 DINOv2-small 编码器提取每帧特征 F_t，计算一阶 Motion Variation M_t 和二阶 Geometric Curvature C_t = 1 - cos(d1, d2)（特征位移向量的角度偏差）。C_t 在微分几何视角下等价于 1/2 ||T2 - T1||²（单位切向量变化的平方），即流形曲率的离散近似——当特征演化方向突变时 C_t 急剧增大（语义转换尖峰），恒速运动/平滑相机平移时 C_t ≈ 0（几何惩罚自然抑制物理噪声）。最终 CS_t = M_t + λ·C_t。
  (2) **Hierarchical Visual Memory Management (HVMM)**：使用 EMA 在线更新曲率分数的分布参数 (μ_t, σ_t²)，构建 K-Sigma 动态双阈值 g1/g2。每帧根据 CS_t 动态路由为 Clear Memory（CS_t ≥ g2，保留原始高分辨率）、Blurred Memory（g1 ≤ CS_t < g2，降采样 224×224 作为平滑过渡态）或 Discard（CS_t < g1，丢弃冗余帧），队列以 FIFO 严格控制 |M_t| ≤ N_max = 20。
  (3) **Semantic-decoupled geometric prior**：曲率度量的理论优势在于：C_t 在恒速物理运动中近似为零（免疫平移/旋转噪声），在语义突变时产生显著尖峰（对方向导数而非模长敏感），实现了从低层次物理运动到高层次语义转换的数学解耦。

  对比 baseline 的全栈执行例子（CurveStream + Qwen2.5-VL-7B, N_max=20, StreamingBench）：
  - 算法层：无限长流视频输入 → 每帧经 DINOv2-small 提取 F_t → CAS 计算 CS_t（融合一阶运动和二阶曲率）→ HVMM 在线更新 (μ_t, σ_t²) 并生成自适应双阈值 → 动态路由：高曲率帧（如物体突现/动作翻转）→ Clear Memory（原始高分辨率），中等曲率帧（场景平移中的过渡状态）→ Blurred Memory（224×224），低曲率帧（长时间静态观察）→ Discard → 当 |M_t| > 20 时 FIFO 驱逐最旧 token → 查询时刻 t_q 的帧强制执行 Clear Memory → visual tokens + text query 送入 MLLM decoder → 生成答案。Clear Memory 占比自适应维持在 ~50%（图 3b），既保持关键语义锚点的高保真，又通过 Blurred Memory 的低分辨率过渡保持动作连贯性和因果链完整。最终 StreamingBench accuracy 84.00%（+10.69% over Qwen2.5-VL-7B uniform baseline），OVOBench 73.48%（+13.58%）。
  - 系统框架层：论文未明确说明 Serving 框架
  - 编译框架层：论文未明确说明
  - kernel调度层：论文未明确说明
  - 硬件架构层：单张推理 GPU，论文未明确型号

  解决对应关系：
  | Baseline 缺陷 | CurveStream 解决方案 |
  |---|---|
  | 语义碎片化（低层次物理度量缺乏语义感知） | CAS 二阶曲率度量：C_t 对方向导数敏感而非模长，恒速运动 ≈ 0、语义突变 → 尖峰。Table III: Uniform=69.04%, Cosine Similarity=73.28%, Optical Flow=46.54%, Curvature=77.31%（训练无关方法中最优） |
  | 信息模糊化（无差别压缩不可逆模糊关键语义转换点） | Clear/Blurred/Discard 三层级路由：高曲率帧保留原始高分辨率（Clear Memory），过渡帧低分辨率维持因果连贯（Blurred Memory），冗余帧直接丢弃。图 3b: 自适应 ~50% Clear Memory ratio 在 accuracy 和 token 成本间达到最优 |
  | 延迟感知（post-hoc retrieval 被动滞后） | CAS+HVMM 构成在线主动感知闭环：曲率分数和 K-Sigma 阈值均实时计算和更新，不依赖用户查询触发——帧路由决策在帧到达瞬间完成，实现真正的实时主动语义感知 |
  | 局部噪声敏感（Cosine similarity/optical flow 混淆物理运动和语义突变） | C_t = 1/2 \|\|T2-T1\|\|² 几何等价性（Appendix C 证明）：免疫恒速运动（T1≈T2 → C_t≈0），仅对特征演化方向突变敏感。Table III: Optical Flow 仅 46.54% accuracy（被像素噪声严重干扰），Curvature 达 77.31% |

  组件协同效应（Ablation Table IX/X）：
  - CAS only: +9.12% (StreamingBench), +8.39% (OVOBench) —— 曲率度量提供了精确的语义感知
  - HVMM only: +9.76% (StreamingBench), +4.69% (OVOBench) —— 层级记忆架构在无感知评分时退化为均匀交替分配，但二值 Clear+Blurred 结构本身即提供比 FIFO 更宽的上下文覆盖
  - CurveStream (CAS+HVMM): +12.00% (StreamingBench), +10.66% (OVOBench) —— 组合增益超过各自贡献之和（非线性协同放大），证明感知和调度之间深度互补：CAS 标记高曲率转换点，HVMM 将其锚定为 Clear Memory，同时将低曲率段平滑压缩为 Blurred Memory，共同构建紧凑的因果拓扑链
