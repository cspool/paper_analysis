论文标题：LEGO: Supporting LLM-enhanced Games with One Gaming GPU

    本地条目说明：
        - 本地编号：paper_2026 第 58 篇
        - 本地 PDF：paper_2026/58-LEGO Supporting LLM-enhanced Games with One Gaming GPU.pdf
        - 本地文本抽取：paper_2026/58-LEGO Supporting LLM-enhanced Games with One Gaming GPU.txt
        - 发表信息：HPCA 2026，DOI：https://doi.org/10.1109/HPCA68181.2026.11408477

    原文与开源仓库确认：
        - 原文状态：已找到本地 PDF 全文，正文包含 Abstract、Introduction、Background and Motivation、LEGO Design、Layer-skipping Adaptor、Headroom-maximizing Scheduler、Implementation、Evaluation、Related Work、Conclusion 等完整章节。
        - 开源状态：未找到明确官方开源仓库。
        - 链接：N/A
        - 说明：论文没有单独的 code availability / artifact availability 段落，也没有在正文中给出 LEGO 官方仓库链接。文中实现基于 llama.cpp、Unreal Engine 4 和 DirectX 12，并引用了开源 Street Fighter III LLM 评测项目作为真实游戏实验，但这不是 LEGO 系统本身的官方开源仓库。基于论文引用和精确网页检索，当前只能确认论文与依赖项目存在，不能确认 LEGO 官方代码已经公开。

    1、论文工作：
        - 论文要解决的核心问题：
          LEGO 解决的是“LLM 增强游戏”在客户端单张游戏 GPU 上部署时的共置执行问题。现代游戏希望用 LLM 控制 NPC、生成动作或对话，但用户本地通常只有一张 GPU；如果把游戏渲染和 LLM 推理分开部署，会增加硬件成本；如果使用云端 LLM 服务，则会引入 20ms 到 110ms 甚至跨洲更高的网络延迟，并且在线游戏场景中 200 APM 和 300 APM 分别对应约 300ms 和 200ms 的动作生成 SLO，云端延迟很容易不可接受。论文因此尝试把游戏渲染和 LLM 推理放在同一张 Nvidia RTX 4090 上运行，同时保证游戏 FPS 与 LLM APM。
        - 瓶颈来源：
          瓶颈主要来自动态且碎片化的 GPU compute headroom，而不是单个算子本身的实现效率。游戏在 60 FPS 下每帧约有 16.6ms deadline，但真实渲染任务并不总是占满整段时间。例如论文对 Black Myth: Wukong、Final Fantasy XVI、Red Dead Redemption 2 的 30 分钟 trace 分析显示，三者长期需要的 GPU time slice 分别约为 60.8%、54.8%、47.6%，说明存在可利用空隙；但这些空隙分散在帧与帧之间，也分散在单帧内部的 rendering subtask 与 auxiliary subtask 之间。同时，Llama3-8B、Mistral-7B 这类 LLM 在 100/200/300 APM 下的资源需求可能使 14/18 个 Game-LLM 场景超过总 GPU compute limit。
        - 论文的主要贡献：
          论文提出 LEGO，一个 algorithm-system co-design。算法侧提出 resource-oriented layer-skipping adaptor，在必须根据资源约束跳过若干 Transformer 层时，用一个小型 FFN adaptor 近似被跳过层的知识变换，降低精度损失。系统侧提出 headroom-maximizing LLM scheduler，用线性回归预测下一个 LLM execution window 内的总渲染 headroom，再选择跳层策略，并把 LLM 推理拆成不同粒度的 subtask 以填充 inter-rendering headroom 和 intra-rendering headroom。实验显示，在 RTX 4090 上 LEGO 在所有测试场景中同时满足 FPS 与 APM 目标，rendering headroom usage 最多提升 28.6%，相对现有 layer-skipping 方法最多减少 86.3% accuracy loss。
        - 论文所处背景：
          背景是游戏引擎和本地 LLM 推理的实时共置。游戏端优先级最高的是视觉体验，因此 LLM 推理必须服从渲染任务 deadline；但 LLM 推理又要维持动作频率，论文使用 100、200、300 APM 表示普通、进阶、专业级游戏动作频率。输入 prompt 通常包含场景描述、NPC/玩家状态、历史动作效果，代表性输入长度取 512 tokens，输出长度取 16 tokens。论文关注的不是云端批量推理，也不是纯 LLM serving，而是消费级游戏 GPU 上“渲染实时任务 + LLM 动作生成实时任务”的细粒度共置。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：
          第一类 baseline 是 SmallModel，即用 Llama3-3B / Mistral-4B 替换 Llama3-8B / Mistral-7B，以降低推理开销。这能缓解 100 APM 或部分 200 APM 场景，但精度下降明显，论文在 MMLU、ARC-C、SQuAD-2.0 上观察到平均 20.41% accuracy drop；在 300 APM 或 variable-length prompt 下仍可能同时丢 FPS 和 APM。
          第二类 baseline 是 LayerSkip，论文主要用 LITE，对比中也报告 CALM。LITE 按 token confidence threshold 做早退或跳层，目标是减少平均计算量，而不是严格控制每个推理请求在游戏共置场景中的资源预算。论文测得 LITE 在 SQuAD-2.0 上即使平均计算时间对齐 latency target，仍有 47.1% 推理任务超过 deadline；若强行加入 SLO 约束形成 LITE-S，需要提前跳过更多层，造成约 27.2% accuracy drop。
          第三类相关系统是 PilotFish 这类 time-division GPU sharing。它可在渲染任务结束后利用帧间 idle time，但对 LEGO 场景不够，因为 LLM 推理还需要利用单个 rendering task 内部更细的 idle gap，否则高 APM 下必须跳过过多 Transformer 层。
        - 论文的设计方法：
          LEGO 的设计分成两条主线。第一条是 resource-oriented layer-skipping adaptor：先基于层间输出 tensor 的 cosine similarity heatmap，寻找在跳过 N 层时最适合替换的一段连续 Transformer 层；再训练一个 FFN adaptor，用 MSE loss 让 adaptor 输出近似被跳过层原本产生的表示。第二条是 headroom-maximizing scheduler：用过去三个 LLM execution windows 的总渲染 headroom 预测下一个 window 的总 headroom，据此确定需要跳过多少层；运行时把 LLM inference 拆成 subtask，decode 阶段可按 Transformer layer 粒度调度，prefill 阶段可按 self-attention / FFN sublayer 粒度调度，以分别适配 inter-rendering 与 intra-rendering headroom。
        - 方法如何对冲 Baseline 缺陷：
          对 SmallModel，LEGO 不是直接换小模型，而是保留较大的 LLM 主体，并只在资源不足时按需要跳过部分层，因此在可用 headroom 较多时能维持接近原始模型的能力。对 LITE/CALM，LEGO 不依赖 token confidence 来决定是否早退，而是先由资源预算决定跳层数量，再用 adaptor 对被跳过连续层做知识蒸馏补偿，因此更适合严格 SLO 的游戏场景。对 PilotFish 类调度，LEGO 不只利用帧间空隙，还把单帧内部 GPU idle time 纳入总 headroom 预测和 runtime dispatch；论文测得 intra-rendering headroom 平均为 0.24ms，90% 小于 0.73ms，总 intra-rendering headroom 每帧平均 1.39ms、最大 3.1ms，只有把 LLM subtask 粒度降到 layer / sublayer 才能有效利用。
        - 关键 trade-off：
          LEGO 接受了离线 adaptor 训练成本、额外内存占用和运行时调度复杂度，以换取更高的实时共置可行性。以 BlackMyth 为例，最多需要 14 个 LLM adaptors，总训练时间约 36 小时；每个 FFN adaptor 占 268.8MB，12 个 adaptor 合计约 3.23GB。Adaptor 对少量跳层很有效，但跳层数量过大时仍会明显损失精度，例如 Llama3-8B 跳 13/14 层时 MMLU、ARC-C、SQuAD 指标明显下降。Scheduler 的线性回归预测精度依赖游戏 trace 的稳定性；论文报告 sudden spike 只占 1.2% frames，且 execution window 跨 12 到 36 帧，因此单帧波动影响较小，但对完全不同游戏、不同画质设置或不同 GPU，仍需要重新 profiling 和训练/拟合。LEGO 与静态量化、静态 sparsity 兼容，但不适合与动态加速方法叠加，因为动态方法会带来执行时间不确定性和额外开销。

    3、论文实现：
        - Baseline 如何实现：
          SmallModel baseline 用同系列小模型替换大模型，例如 Llama3-3B 替换 Llama3-8B，Mistral-4B 替换 Mistral-7B；运行时按游戏平均 rendering headroom 把 LLM inference 切成等大小 subtasks，渲染任务完成后提交一个推理 subtask。LayerSkip baseline 采用 LITE 和 CALM，其中 LITE 因精度更好作为主要对比；二者根据预定义 threshold 在运行时做跳层，跳层策略确定后使用与 SmallModel 相同的调度方式。所有 baselines 还增强了 PilotFish 式动态 time-slice 机制，使 LLM 任务能在渲染释放 GPU 后立即使用资源，而不是等待静态时间片。
        - 新设计如何实现：
          LEGO 使用 llama.cpp 作为 LLM inference framework，使用 Unreal Engine 4 作为 game engine，图形库为 DirectX 12。实现上只把 llama.cpp 的 front-end 集成到 UE4 中，其他功能通过 dynamic library 调用。因为 llama.cpp 把 computation graph creation 与 traversal 分离，作者修改 traversal function 来加入调度逻辑：游戏引擎监控 rendering task 的状态变量，在渲染完成后启动 inference subtask；decode 阶段可提交 Transformer layers，prefill 阶段可提交 self-attention 和 FFN sublayers。如果没有新的渲染任务到达，scheduler 继续提交后续推理 subtask。为此系统在 dynamic library 中注册新的 schedulable traversal function，以保持推理执行正确。
        - 实验 / 实现平台：
          实验平台为 Windows 11、CUDA driver 566.36、CUDA SDK 12.1、DirectX 12.1、llama.cpp fc83a9e；硬件为 Intel i9-13900KF @ 3.00GHz 和 Nvidia RTX 4090。模型包括 Llama3.2-8B-Instruct 与 Mistral-7B-Instruct-v0.3，并扩展评估 DeepSeek-V2-Lite、Mixtral-8x7B 等 MoE 模型。游戏包括 Black Myth: Wukong、Final Fantasy XVI、Red Dead Redemption 2，均设置为 4K、高画质、60 FPS。APM 场景为 100、200、300；默认代表性输入长度为 512，输出长度为 16，variable-length prompt 实验中输入长度在 [256, 1024] 均匀采样。
        - 关键实验设置与指标：
          主要实时指标是 99th-percentile FPS 与 99th-percentile APM，用来检查游戏渲染和 LLM 动作生成是否都满足目标。LLM 质量指标包括 MMLU、ARC-C accuracy 与 SQuAD-2.0 F1，也包括 Street Fighter III 真实游戏对战中的 win rate。Headroom prediction 中，朴素时间序列模型按 rendering task 粒度预测误差可超过 3%，最大 5.49%，且 ARIMA/SVM 开销过高；LEGO 改为以 LLM execution window 为单位，用 LR 模型基于前三个 window 的总 headroom 预测下一个 window，总体最大误差 1.3%，平均误差 0.6%，三输入窗口推理开销约 1.3ms，运行时拟合约 0.9ms。
        - 关键实验结论：
          在所有 Game-LLM co-location 场景中，LEGO 同时维持目标 FPS 和目标 APM。SmallModel 在 300 APM 下出现 26.2% FPS drop 和 20.5% APM drop；LayerSkip 在 200/300 APM 下维持游戏 FPS 但引入 28.6% APM drop。精度方面，Llama3-8B 跳 12 层时，LEGO 在 MMLU/ARC-C/SQuAD 上显著优于 LITE 和 CALM；相对 LITE，最多减少 86.3% accuracy loss。真实 Street Fighter III 实验中，在固定 200 APM 下，LEGO-4 优于 LEGO-8、LEGO-12、Llama3-3B 与 LITE-4，LEGO-12 与 Llama3-3B 接近。Variable-length prompt 下，LEGO 仍能维持 FPS 与 APM，而 SmallModel 和 LayerSkip 在高 APM 下出现明显下降。Headroom usage 方面，相比 SmallModel，LEGO 在 100/200/300 APM 下分别提升 25.2%、28.6%、18.8%；相比 LayerSkip，在 200/300 APM 下分别提升 14.0%、16.2%。

    4、pipeline/kernel 解析：
        - 新 pipeline/kernel 是什么：
          论文没有提出一个单独的新 GPU kernel；它的新机制更接近一条面向 Game-LLM co-location 的 runtime pipeline，核心包括 resource-oriented layer-skipping adaptor pipeline 和 headroom-maximizing scheduling pipeline。Adaptor pipeline 在离线阶段分析 Transformer 层间相似度、选择要跳过的连续层段、训练 FFN adaptor；在线阶段根据 scheduler 给出的跳层数量，用 adaptor 替代对应 Transformer 层段。Scheduling pipeline 在在线阶段持续监控游戏渲染状态、预测下一个 execution window 的总 headroom、选择跳层策略，并把 LLM 推理拆成适合当前 GPU 空隙的 subtask。
        - 离线 adaptor 准备流程：
          对一个游戏和模型组合，系统先 profile 代表性 gameplay 中的 rendering task 执行时间，得到 Cmin/Cmax 和不同 APM 下可用于 LLM 的 Hmin/Hmax。然后测量一个 LLM action 的总计算时间 Toverall，以及 prefill 阶段 Tpl 和 decode 阶段每个 Transformer layer 的 Tdl。根据这些 profiling 结果，计算在最小 headroom 下必须跳过的最小层数 M，以及在最大 headroom 下可能跳过的最大层数 N，因此需要准备 N-M+1 种跳层配置。接着用训练数据生成层间 similarity heatmap；当需要跳过 N 层时，选择 heatmap 对应 diagonal 上相似度最高的连续层区间，例如 Llama3-8B 跳 4 层时选择 L25 到 L29，跳 8 层时选择 L23 到 L31。最后训练 FFN adaptor，使其输入为第 k 层输出 fk，输出近似第 k+n 层原始输出 fk+n，只更新 adaptor 权重。
        - 在线调度流程：
          一个 LLM action 到达后，scheduler 首先读取当前 APM 对应的 execution window，例如 100 APM 为 600ms、200 APM 为 300ms、300 APM 为 200ms。它使用过去三个 execution windows 的总渲染 headroom，通过 LR 模型预测下一个 window 内可用于 LLM 推理的总 headroom。然后根据预测 headroom 选择跳层数量和对应 adaptor，使 LLM 推理计算量落入资源预算。由于该预测已经包含 intra-rendering headroom，scheduler 后续要在运行时同时利用两类空隙：帧间较大的 inter-rendering headroom 与单帧内部很短的 intra-rendering headroom。
        - 一个具体请求如何流过系统：
          以 200 APM 的 BlackMyth + Llama3-8B 为例，游戏每约 300ms 需要一次 LLM action。游戏引擎从场景状态、角色状态和历史动作构造 prompt，LLM 开始生成动作 token。调度器预测接下来 300ms window 中的总 GPU headroom，如果不足以运行完整 Llama3-8B，就选择一个跳层配置，例如跳 4、8 或 12 层，并启用对应 adaptor。Prefill 阶段中，系统把 self-attention 和 FFN sublayer 作为较细的调度单位；decode 阶段中，把 Transformer layer 作为调度单位。当渲染 subtask 结束并出现短 idle gap 时，scheduler 提交一个细粒度 LLM subtask；如果下一个 rendering subtask 已经开始，scheduler 等待，避免抢占渲染；如果整帧渲染完成并进入帧间空隙，则提交更粗粒度、包含多个 Transformer layers 的 LLM subtask。所有 token 生成完成后，LLM 输出动作描述，游戏引擎执行对应技能或移动。
        - runtime dispatch 的安全条件：
          为保证渲染任务不被 LLM 推理破坏 deadline，LEGO 对每个被提交的 LLM subtask 施加 Tsubtasks <= Tminimal 的条件，其中 Tsubtasks 是该推理 subtask 的执行时间，Tminimal 是游戏所有 rendering tasks 中最小的 inter-rendering headroom。这样即使利用 intra-rendering headroom，推理 subtask 也不会长到阻塞下一个渲染计算阶段。对于 sudden spike，scheduler 在每个 token 生成后用最新 workload 数据更新预测；若发现 QoS violation 风险，就动态调整后续 token 的 layer-skipping strategy。
        - 与传统 pipeline/kernel 的区别：
          传统本地游戏和 LLM 推理要么分设备运行，要么简单时间片共置；前者成本高或云端延迟大，后者难以同时满足 FPS 与 APM。传统 layer-skipping 主要面向平均推理加速，按 token confidence 动态早退，不能保证每个游戏动作请求适配严格资源窗口。LEGO 的关键变化是把“资源预算”提升为跳层决策的主导因素，再用 adaptor 对被资源约束牺牲的层做表示补偿；同时把 LLM execution 从完整请求拆成 layer/sub-layer 级 subtask，使推理能填充游戏渲染留下的碎片化 GPU headroom。它的创新点不在于替换 LLM 模型结构或写一个新 attention kernel，而在于把模型跳层、知识蒸馏、headroom prediction 和 game-engine-aware scheduling 连接成一条面向单 GPU 游戏客户端的实时共置执行路径。
