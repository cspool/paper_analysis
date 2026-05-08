论文标题：ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models

    本地条目说明：
        - 本地编号：paper_2026 第 29 篇
        - 本地文件名：29-Difflow A Data-Characteristic-Aware Serving System for Diffusion Models.pdf
        - 标题差异：PDF 正文、PPoPP 2026 页面和 DOI 记录均使用 ChituDiffusion；本报告以论文正文题名为准。

    原文与开源仓库确认：
        - 原文状态：已找到本地 PDF 全文；论文发表于 PPoPP 2026，DOI 为 https://doi.org/10.1145/3774934.3786424
        - 官方页面：https://ppopp26.sigplan.org/details/PPoPP-2026-papers/14/ChituDiffusion-A-Data-Characteristic-Aware-Serving-System-for-Diffusion-Models
        - 开源状态：已找到官方仓库 / 官方分支
        - 仓库链接：https://github.com/thu-pacman/chitu/tree/Diffusion
        - 说明：论文摘要与 PPoPP 2026 页面均声明 ChituDiffusion 代码和 production traces 已开源在 thu-pacman/chitu 的 Diffusion 分支；GitHub 页面显示该分支包含 chitudiffusion 目录，并采用 Apache-2.0 license。本文未进一步运行仓库复现实验，因此只确认公开可访问状态，不评价 artifact 完整性。

    1、论文工作：
        - 论文要解决的核心问题：
          Diffusion model serving 中，真实请求并不是“同 prompt、同 shape、同 pipeline 参数”的规整批次，而是同时存在多种数据属性：prompt / image / ControlNet / LoRA 等输入可能在部分请求间重复，生成分辨率和视频帧形状可能不同，denoising loop 中又会出现跨迭代不变张量。现有通用 DNN 编译器和 diffusion 框架通常只能对固定 pipeline 或均匀 batch 做优化，难以在 heterogeneous requests 中同时利用“部分重复”和“动态 shape”。

        - 瓶颈来源：
          主要瓶颈来自三个层面。第一是计算与显存访问冗余：多个请求共享 prompt、LoRA、ControlNet 或中间结果时，baseline 仍可能重复执行 CLIP / U-Net / attention 等计算，或重复读取相同 K/V、权重和 conditioning。第二是调度和 batching 低效：按完全相同 shape 才 batching 会让许多请求无法合批；直接使用 ragged batch 又会为均匀 shape 请求引入额外索引和变换开销。第三是编译期组合爆炸：一个 diffusion pipeline 可有十几个甚至二十多个输入，如果为所有输入属性组合生成整图特化版本，dEngine 数量会指数增长。

        - 论文的主要贡献：
          论文提出 ChituDiffusion，将 diffusion pipeline 重组成多个具有相同数据属性传播条件的 dGraph，并为每个 dGraph 编译多个针对特定属性的 dEngine；运行时再把用户请求拆成 dTask，用动态规划调度器把属性兼容的 dTask 合批并派发给匹配的 dEngine。它把数据属性局部性显式化，使编译期优化和运行时 scheduling 能协同利用冗余、ragged shape 和 loop invariant。

        - 论文所处背景：
          目标场景是在线 diffusion model serving，覆盖 text-to-image、image-to-image、video generation、ControlNet、LoRA、SDXL refiner、SVD、FLUX、Hunyuan 等 pipeline。论文特别强调用户侧的 grid search、prompt 变体、不同分辨率生成和重复输入，这些真实行为使 request-level data properties 具有强局部性，但又不是传统 batch 编译器容易处理的完全规则输入。

    2、相对 Baseline 解决的问题与设计方法：
        - Baseline 的具体问题：
          PyTorch、PyTorch-Inductor、TensorRT 这类通用执行/编译框架关注 operator 或静态 graph 优化，缺乏 diffusion request 数据属性感知，通常不能识别“多个请求共享一部分输入但 shape 不同”的机会。Diffusers / Stable Fast 等 diffusion-specific framework 可以为同 prompt、同 shape 等少数规则场景手工优化 pipeline，但无法覆盖多个输入属性组合；如果枚举所有组合，编译版本会爆炸。Katz 这类专用 diffusion serving 系统能优化 ControlNet-as-a-service，但论文指出它的服务模型和硬件假设更重，且对 LoRA 等场景不完全等价。

        - 论文的设计方法：
          ChituDiffusion 的核心方法是“pipeline recompose + request recompose”。编译期先把整个 diffusion pipeline 作为 DFG，通过符号数据属性传播规则分析 redundancy、raggedness、invariant 等属性如何沿 tensor operator 传播；然后按相同输出属性表达式把连续 operator 分组成 dGraph。每个 dGraph 再根据可能的输入属性生成多个 dEngine，并剪枝冲突条件和收益低于阈值的无关条件。运行期把请求按 dGraph 粒度拆成 dTask，调度器在一个 scheduling window 内寻找 dTask 与 dEngine 的合批方案。

        - 方法如何对冲 Baseline 缺陷：
          对编译组合爆炸，ChituDiffusion 不为整条 pipeline 枚举所有属性组合，而是把优化机会局部化到 dGraph；论文在 SDXL U-Net + ControlNet 案例中显示，monolithic 需要 2^14 个 engine、估计 11 天编译，而 dGraph 重组后只需 4 个 dGraph、7 个 engine、约 7 分钟。对运行时 heterogeneous requests，系统不强行选择“只用 uniform batch”或“只用 ragged batch”，而是让调度器在不同 dEngine 之间选择，例如同 prompt 的请求走 redundancy dEngine，同 shape 的请求走 uniform dEngine，混合 shape 但可共享权重的请求走 ragged dEngine。对重复计算和内存访问，系统用维度级 redundancy elimination 压缩冗余维度，再通过 broadcast 恢复等价输出。

        - 关键 trade-off：
          该方法接受更复杂的编译器分析和 runtime scheduler，以换取更细粒度的属性感知优化。dGraph/dEngine 机制降低了全组合编译成本，但仍需要为新 operator 提供属性传播规则和优化 kernel。ragged regularization 能合批不同 shape 请求，但会引入 transpose、reshape、im2col、索引等额外开销，所以在 raggedness 很低时未必优于 uniform dEngine。调度器依赖轻量性能模型估算 dEngine 执行时间，模型足够实用但仍是经验型预测。

    3、论文实现：
        - Baseline 如何实现：
          论文将 ChituDiffusion 与 PyTorch v2.1、PyTorch-Inductor v2.1、TensorRT v8.6、Stable Fast v1.0 对比，并把这些 baseline 调到尽量饱和 GPU throughput。对于 Katz，论文只在 edit 应用中比较 ControlNet 部分，因为 Katz 的 LoRA serving 与 ChituDiffusion 的数学语义不完全等价；Katz 至少需要 4 张 H100 服务单个 ControlNet，因此论文用 per-GPU throughput 做归一化比较。

        - 新设计如何实现：
          ChituDiffusion 用 C++ 和 Python 实现，复用 Diffusers、Triton、Stable Fast 和 FlashAttention 的部分组件。系统提供 diffusion pipeline 定制能力，支持用户把应用特征输入编译器，例如 prompt 可能重复、image shape 可能 ragged。对 ragged batching，论文实现了 4 个基于 Triton 和 CUDA 的 ragged data-independent operation kernels。核心优化包括：
          1. 符号 dGraph 识别：把 input data property 表示为布尔或符号表达式，沿 tensor operator 传播，按相同输出属性表达式划分 dGraph。
          2. dEngine 特化编译：为每个 dGraph 生成适配 redundancy、raggedness、uniform shape 等属性的多个 dEngine，并剪枝不可满足或收益很小的版本。
          3. redundancy elimination：按 operator 规则消除冗余计算和冗余显存访问，例如 attention 中 K/V 在 batch 维重复时压缩 K/V 并拼接不同请求的 Q。
          4. ragged operation regularization：把 ragged Matmul、elementwise、convolution 转换成标准 operator 加少量 ragged data-independent 操作，而不是为每个 operator 手写 ragged kernel。
          5. invariant tensor elimination：用 constant、loop-invariant、loop-variant、unknown 四态传播算法识别 compile-time constant 和 denoising loop invariant；constant 预计算，loop invariant 提到循环外。

        - 实验 / 实现平台：
          论文在两类单卡服务器上评测：NVIDIA A100 40GB PCIe 和 NVIDIA H100 80GB PCIe。UNet 结构模型使用 CUDA 12.1；DiT 系列模型使用 CUDA 12.8。应用覆盖 SD1.5、SDXL、SDXL refiner、SVD、ControlNet、LoRA，以及 Hunyuan / FLUX 等 DiT 结构场景。请求分布和 ragged shape 参考 DiffusionDB 与 Civitai；denoising steps 使用各应用默认设置，random seeds 均匀采样。

        - 关键实验设置与指标：
          主指标是 throughput，单位为 requests/s。论文在 5 个 UNet-based diffusion 应用上报告，相比最优 baseline，ChituDiffusion 在 A100 上最高提升 2.13 倍、平均 1.58 倍，在 H100 上最高提升 2.19 倍、平均 1.51 倍。对 refine、edit、video 这类 correlative request 场景，收益主要来自共享输入和冗余计算消除；对 venti / grande 这类标准 text-to-image 服务，冗余较少，收益更多来自 ragged batching 与 dEngine 选择。DiT 额外实验显示，ChituDiffusion 在 refiner-mix、refiner-DiT、edit-DiT 上也能获得 1.40 倍到 2.97 倍 throughput 提升，其中 torch.compile 后端和数据感知调度都参与了收益。

        - Ablation 与局限证据：
          edit 应用中的消融实验显示，在 ChituDiffusion-base 上逐步开启 dTask scheduling、property-aware multi-version dEngine compilation、invariant redundancy elimination 后，throughput 分别提升到 1.29 倍、1.56 倍和 1.71 倍。调度开销方面，论文报告动态规划 scheduling cost 小于 dEngine runtime 的 10%，在较大 scheduling window 下可通过 batching 与异步 overlap 进一步隐藏，GPU idle ratio 可低于 5%。局限上，论文的优化收益依赖请求是否具有可利用的数据属性；当请求完全独立且 shape 很规则时，收益会收窄，而 ragged dEngine 的额外开销也可能让 uniform dEngine 更合适。

    4、pipeline/kernel 解析：
        - 新 pipeline/kernel 是什么：
          论文引入的不是单一 named GPU kernel，而是一条数据属性感知 serving pipeline，核心对象是 dGraph、dEngine 和 dTask。dGraph 是由 pipeline DFG 中共享优化条件的一段连续 operator 组成的子图；dEngine 是某个 dGraph 在特定数据属性条件下编译得到的执行引擎；dTask 是运行时从用户请求拆出来、对应某个 dGraph 的细粒度任务。closest concrete kernel-level mechanism 是 ragged operation regularization：把 ragged Matmul / elementwise / convolution 变换成现有 regular kernel 可执行的形式，并用少量 Triton/CUDA kernel 处理 ragged data-independent 操作。

        - 编译期 pipeline 执行流程：
          1. 输入一个 diffusion application pipeline，例如 SDXL + refiner 或 SDXL + ControlNet + LoRA。
          2. 开发者或系统给定应用特征，例如 prompt 可能相同、image shape 可能不同、某些 LoRA 权重可能固定。
          3. ChituDiffusion 将 pipeline 转为 DFG，并把输入属性初始化为符号变量或固定属性。
          4. 属性传播规则沿 operator 推导输出属性表达式；对 denoising loop，系统先展开初始迭代直到 loop 输入稳定，避免循环内属性表达式冲突。
          5. 连续 operator 如果具有相同输出属性表达式，就被划成同一个 dGraph；很小的 dGraph 会在后处理阶段并入后续 dGraph，降低 runtime scheduling 负担。
          6. 对每个 dGraph，编译器枚举输入属性条件，生成 redundancy、ragged、uniform 等多版本 dEngine，并剪枝不可满足条件和收益小于 5% 的条件。

        - 运行时 pipeline 执行流程例子：
          假设一个 scheduling window 中有 4 个 image generation 请求：R1、R2、R3 使用同一个 prompt P1，但生成 shape 分别是 512x768、512x512、768x768；R4 使用 prompt P2，shape 为 512x512。系统先把每个请求按 dGraph pipeline 拆成 dTask，放进 dTask pool。对于某个 dGraph，调度器会比较多个方案：把 R1-R4 全部放到通用 ragged dEngine 执行；把 R1-R3 放到 redundant-prompt dEngine 以消除共享 prompt 相关计算；或把 R2 与 R4 放到 uniform-shape dEngine 以避免 ragged 开销。动态规划搜索会枚举 dEngine 要求，找到满足条件的最大 batch，利用性能模型估算执行时间，再递归处理剩余 dTask，最后生成一个总时间估计最小的 execution plan。

        - 一个 tensor / request 如何流过系统：
          对 prompt embedding 这类可能重复的数据，系统先用 fingerprint 判断不同 dTask 的输入是否等价；如果重复，则多个 dTask 可以共用 dGraph 输出，或在 attention 中压缩冗余 K/V，只保留非冗余维度执行计算，最后 broadcast 恢复到每个请求需要的形状。对 ragged image latent，系统把 ragged dimension 作为符号形状传播到 dGraph 输出；执行 ragged Matmul 时，通过 transpose/reshape 把 batch 维和 ragged 维合并成 regular dimension，使已有 Matmul kernel 能处理拼接后的数据；对 convolution，则使用类似 im2col 的变换把 ragged data-sharing operation regularize 成标准矩阵乘路径。执行完成后，输出属性由符号表达式结合真实输入属性异步推断，下一阶段 dTask 可以在上一阶段 executor 运行时提前被调度。

        - 与传统 kernel/pipeline 的区别：
          传统 diffusion serving pipeline 往往以完整请求为单位 batching，只有当请求 shape、prompt 或 pipeline 参数完全对齐时才有手工优化机会。ChituDiffusion 把完整请求拆成 dGraph-level dTask，使“局部相同 prompt”“局部相同 shape”“局部 invariant tensor”都能在对应子图内触发优化。它的关键不是把某个算子写得更快，而是让 runtime 根据数据属性选择正确的 dEngine，从而在 uniform batch、ragged batch、redundancy elimination 之间做细粒度切换。
