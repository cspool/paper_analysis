## VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

- 属于算法pipeline的实现是什么？实验比较什么？
  属于算法pipeline的实现是：(1) 一个包含17个视觉交互环境的gymnasium框架，用于评估和训练VLM在多步视觉决策中的表现；(2) 多步求解器（multi-step solvers），通过BFS/DFS/图搜索/状态机等策略为每个任务生成结构化demonstration轨迹，用于监督微调（SFT）；(3) 训练方法：基于solver生成的demonstration进行全参数SFT，使用LlamaFactory框架。

  实验比较：(1) 12个前沿VLM模型在17个环境的easy/hard设置下的零样本成功率对比（Sec 3）；(2) 诊断实验——历史长度（full/4/2/1 turns）、文本vs图像观察表示、有无文本反馈、有无目标观察的对照实验（Sec 4）；(3) 训练实验——单任务SFT vs 多任务SFT的效果、easy→hard泛化能力对比（Qwen2.5-VL-7B vs Qwen3-VL-8B）、vision encoder vs LLM的模块贡献分析（Sec 5.3）、信息揭示型demonstration vs 标准demonstration的数据筛选策略对比（Sec 5.4）。

- 硬件平台是什么，配置是什么。
  论文未明确说明评估和训练的硬件平台。评估中proprietary模型通过OpenRouter API调用（云端推理），open-weight模型和训练使用的GPU未显式列出。训练配置：Qwen2.5-VL-7B-Instruct全参数微调，global batch size=64，learning rate=1×10^{-5}，bf16精度，单任务训练1500步，多任务训练5000步，使用LlamaFactory进行数据处理和训练编排。

- 模型是什么。数据集和bench分别是什么。
  评估模型（12个）：proprietary — Gemini 3 Pro、Gemini 2.5 Pro、GPT-5、Claude Sonnet 4、Grok 4 Fast、Qwen-VL-Max；open-weight — Qwen3-VL-235B-Instruct、GLM-4.5V、Llama-4-Maverick、Qwen-2.5-VL-72B-Instruct、Gemma 3-27B-Instruct；专用模型 — UI-Tars-1.5-7B。训练基座模型：Qwen2.5-VL-7B-Instruct。
  
  17个环境覆盖：Colorization（LLaVA数据集）、Counting（LVIS数据集）、Jigsaw（LLaVA）、Matchstick Equation（合成）、Matchstick Rotation（合成）、Maze 2D（合成/Maze-World）、Maze 3D（合成/Maze-World）、Mental Rotation 2D（LLaVA）、Mental Rotation 3D Cube（合成）、Mental Rotation 3D Objaverse（Objaverse数据集）、MuJoCo Fetch Pick-and-Place（MuJoCo）、MuJoCo Fetch Reach（MuJoCo）、Patch Reassembly（合成）、Referring Dot-Pointing（RefCOCO数据集）、Sliding Block（合成）、Video Unshuffle（SS2 Something-Something数据集）、Zoom-In Puzzle（LLaVA数据集）。每个环境70 episodes/task/setting，easy设定最大20步，hard设定最大30步。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：GitHub [visgym/VisGym](https://github.com/visgym/visgym)，Apache-2.0 license；HuggingFace模型 [VisGym/visgym_model](https://huggingface.co/VisGym/visgym_model)；项目页面 [visgym.github.io](https://visgym.github.io/)。

  算法pipeline核心流程：
  1. **环境交互循环**：每步t，模型接收完整历史 H_t = (I, {(o_τ, a_τ, f_τ)}_{τ<t})，其中I为任务指令，o_τ为观察图像（H×W×C），a_τ为已执行动作，f_τ为环境文本反馈。模型输出动作a_t。
  2. **Function-Conditioned Action Space**：动作表示为函数调用格式，如 ('swap', ((0,0),(0,1))), ('rotate', (30.5, 20.4, 15.1))，而非离散/连续动作向量。
  3. **Step函数**（伪代码）：
     ```
     function Step(a):
         ρ ← 0; (τ, υ) ← (false, false)
         Parse a → (α, π)
         if invalid format: return (obs(), 0, τ, υ, "invalid format")
         if α ∈ A and π ∈ A[α]: (φ, τ, υ) ← Apply(α, π)
         else: return (obs(), 0, τ, υ, "invalid action")
         if τ = true: ρ ← ComputeReward()
         return (obs(), ρ, τ, υ, φ)
     ```
  4. **Solver生成Demonstration**：每个任务配备启发式solver（Maze用图搜索找最优路径，Sliding Block用BFS，Matchstick Equation用BFS/DFS，Jigsaw用贪心swap或直接reorder排列，Fetch用状态机oracle）。Solver支持多策略和可选随机性，生成多样化轨迹。轨迹预处理过滤掉失败轨迹和与测试集重叠的初始状态。
  5. **SFT训练**：Qwen2.5-VL-7B-Instruct全参数微调，LlamaFactory编排，solver demonstrations仅来自easy难度——hard表现衡量难度泛化。

  **近似层次匹配说明**：本论文是VLM多步视觉决策评估与训练框架，不完全匹配传统算法pipeline（稀疏/量化/蒸馏加速推理），但因其包含训练方法（SFT with solver demonstrations），按最接近层次分类到算法pipeline。
