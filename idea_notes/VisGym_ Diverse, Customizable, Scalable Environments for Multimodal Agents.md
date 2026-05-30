## VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

- baseline方法是什么？
  Baseline是已有的VLM视觉交互评估框架和VLM的零样本视觉决策能力。具体而言：(1) 已有benchmark（OSWorld、LIBERO、VideoGameBench、LMGame-Bench、VLABench、VLM-Gym、KORGym、Visual Agent Bench、VAGEN等）存在三大缺陷——任务数量少（4-6个非robot任务）、缺乏跨领域覆盖（domain-specific而非domain-agnostic）、诊断能力弱（仅观察性"what fails"而非受控系统性"why fails"）；(2) 前沿VLM在多步视觉决策中表现出四种系统性失败模式——受限动作空间和动作循环（action looping）、状态管理失败（state mismanagement）、提前终止（early termination）、视觉/空间信息利用失败（failure to use visual or spatial information）。

  全栈执行例子（以GPT-5在VisGym评估前，使用现有benchmark的典型零样本VLM评估流程）：
  - 算法层：VLM zero-shot推理——模型接收单张图像或简单prompt，输出文本回复或单步动作，无多步交互历史管理。
  - 系统框架层：API-based evaluation（如通过OpenRouter调用GPT-5 API），每次请求独立，无状态追踪。评价指标为单步准确率或简单成功率。
  - 编译框架层：论文未明确说明（云端推理，无编译层）。
  - Kernel调度层：论文未明确说明（云端GPU推理）。
  - 硬件架构层：论文未明确说明（云端GPU）。

  Baseline核心缺陷：
  1. **缺乏受控诊断能力**：已有benchmark只能报告"模型在某任务上失败"，无法系统性地隔离导致失败的具体因素（如context length、representation modality、feedback design、goal visibility）。
  2. **缺乏训练支持**：大多数benchmark仅支持evaluation，不提供training pipeline（demonstration生成、SFT支持、RL接口），无法将评估洞察转化为模型改进。
  3. **跨领域覆盖不足**：已有benchmark各自聚焦单一领域（robotics、computer use、games），缺乏跨symbolic puzzles→real-image understanding→navigation→manipulation的统一框架。
  4. **VLM的多步视觉决策能力未被充分理解**：前端模型在static VQA benchmarks上表现优异（MMMU、MMBench），但在interactive multi-step视觉任务中的行为特征（如context length的倒U曲线效应、文本vs视觉表示的性能翻转、目标观察的逆火效应）未被系统化揭示。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出VisGym——一个包含17个环境的gymnasium框架，通过四个协同设计解决baseline缺陷：

  **(1) 17个跨领域环境 + 细粒度可控参数**——解决"跨领域覆盖不足"和"缺乏受控诊断能力"：
  VisGym覆盖四大领域：symbolic puzzles（Matchstick Equation/Rotation, Sliding Block, Maze 2D, Patch Reassembly）、real-image understanding（Colorization, Counting, Jigsaw, Mental Rotation 2D, Referring Dot-Pointing, Video Unshuffle, Zoom-In Puzzle）、navigation（Maze 3D）、manipulation（MuJoCo Fetch Pick-and-Place/Reach）。每个环境提供2-5个可控难度参数（如maze size、puzzle pieces count、angular tolerance），支持easy/hard两档。关键诊断开关：history length（1/2/4/∞ turns）、observation modality（image vs ASCII text）、textual feedback（on/off）、goal observation（with/without）——允许对同一任务进行受控对照实验，系统性隔离各因素对性能的影响。

  **(2) 多步求解器 + Demonstration生成**——解决"缺乏训练支持"：
  为每个任务实现启发式solver（BFS/DFS/图搜索/状态机oracle），支持多策略（如Jigsaw的reorder vs swap策略）和可选随机性（通过插入可逆padding动作），生成多样化demonstration轨迹。Demonstration预处理过滤失败轨迹和data leakage。Solver使VisGym从纯evaluation框架升级为evaluation+training框架。

  **(3) Function-Conditioned Action Space + 统一Step接口**——解决"多任务统一评估"：
  动作表示为函数调用格式（如('swap', ((0,0),(0,1))), ('rotate', (30.5,20.4,15.1))），充分发挥VLM的function-calling能力。统一Step函数处理解析、验证、执行、反馈（Algorithm 1），使所有17个任务共享相同的评估协议——多轮对话、完整历史（Eq.1）、统一步数限制。

  **(4) 系统性诊断 + 失败模式自动发现**——解决"VLM行为未被充分理解"：
  通过受控诊断实验发现五个关键insight：(i) 历史长度的倒U曲线——模型受益于有限历史（~4 turns），但无限历史反降性能；(ii) 文本vs视觉表示——GPT-5在文本表示下成功率高3-4×，瓶颈在视觉grounding而非推理；(iii) 文本反馈依赖——所有模型移除文本feedback后性能一致下降，无法从纯视觉变化推断action结果；(iv) 目标观察逆火效应——提供目标图提升大多数任务性能，但当VLM视觉感知不足时（如Zoom-In Puzzle/Matchstick Equation误判概率80%/57%），反而更差；(v) 信息揭示型demonstration——在partial observability/unknown dynamics下，暴露隐藏状态的demonstration（如Matchstick Rotation先做unit-scale探索步骤再对齐）比标准demonstration显著更有效（32.9%→70.0%）。

  全栈执行对比baseline（以Qwen2.5-VL-7B在VisGym上完成Maze 3D + SFT训练为例）：
  - 算法层：VisGym Maze 3D环境——部分可观察3D迷宫，agent通过move(0)/turn(d)动作导航到target。Solver基于图搜索生成最优路径demonstration。Qwen2.5-VL-7B全参数SFT微调1500步（单任务）或5000步（多任务），仅用easy难度demonstration训练。
  - 系统框架层：VisGym基于Gymnasium框架（与MuJoCo、Atari同一底层库），扩展了function-conditioned action space和function instructions。LlamaFactory处理训练数据预处理和训练编排。OpenRouter API统一proprietary模型评估接口。
  - 编译框架层：论文未明确说明（标准PyTorch eager execution）。
  - Kernel调度层：论文未明确说明（标准GPU kernel执行）。
  - 硬件架构层：论文未明确说明具体GPU硬件。训练使用bf16精度，full-parameter fine-tuning。

  效果：SFT后Qwen2.5-VL-7B在大多数任务上达到SOTA，验证环境的可学习性和solver demonstration的有效性。Qwen3-VL-8B比Qwen2.5-VL-7B在easy→hard泛化上表现更好（近2×成功率），说明更强的基座模型提供更好的分布外泛化。LLM backbone微调贡献大于vision encoder微调（尤其在partial observability任务上），说明时序推理和历史整合是当前VLM的主要瓶颈。

  设计思路核心：VisGym的本质是将VLM评估从"domain-specific observational benchmarking"转变为"domain-agnostic controlled diagnosis + training"。三个关键设计支撑这一转变：(1) 环境多样性+参数可控性使cross-domain controlled experiments成为可能（而非per-domain case study）；(2) solver-generated demonstrations使评估洞察可以直接转化为训练数据（闭环：发现failure→生成针对性demonstration→SFT改进）；(3) 信息揭示型demonstration的发现揭示了VLM在partial observability/unknown dynamics下需要的是structured exploration（暴露隐藏状态和dynamics），而非更多的标准demonstration——这对future VLM training data curation有重要指导意义。
