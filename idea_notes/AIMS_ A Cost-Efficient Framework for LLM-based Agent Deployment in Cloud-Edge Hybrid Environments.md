## AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

- baseline方法是什么？
  Baseline有两个层次：(1) HybridLLM [8]：使用classifier对每个subtask独立判定走SLM或LLM，subtask之间无依赖感知、无位置感知。每次路由决策仅考虑当前subtask的内容特征，忽略subtask在agent reasoning workflow中的阶段（early/mid/late）和subtask间的因果链（前一个subtask的SLM输出可能导致后续subtask偏离LLM路径）；(2) Minions [31]：confidence-based routing，SLM先尝试执行每个subtask，用average log-probability衡量uncertainty，低置信度时escalate到云LLM。同样逐subtask独立决策，不建模subtask依赖和位置效应。两个baseline的共同缺陷：将AI agent的subtask序列视为彼此独立的单次请求集合，忽略agent workflow中subtask间的强依赖关系和cascading effect——一个早期subtask的错误路由可能改变后续整个subtask链。

  全栈执行例子（以HotpotQA "maternal grandfather of Titanic director"请求 + HybridLLM + Qwen3-4B/GPT-5 + RTX 5090为例）：
  - 算法层/Serving层：AutoGen agent生成subtask ST1="Identify Titanic director"→HybridLLM classifier判定ST1简单→路由到Qwen3-4B SLM→SLM输出"James Cameron"（正确）。但ST1的SLM执行导致agent state与LLM路径产生微小差异→ST2="Find James Cameron's mother"→SLM可能输出"Shirley Lowe"但遗漏full name细节→ST3="Find Shirley Lowe's father"因缺少中间名而搜索到错误人物。HybridLLM的classifier在每个subtask独立评估时可能都判定"简单"，但累积状态偏移（early divergence accumulation）最终导致最终答案错误。HybridLLM在HotpotQA上accuracy仅76.35%，SLM usage 68.40%，对后期subtask（late-stage）的LLM→SLM切换导致精度损失高达9.53%（vs early-stage仅5.25%），但classifier因无位置感知而对所有stage一视同仁。
  - 编译框架层：论文未明确说明（llama.cpp默认编译路径）。
  - kernel调度层：论文未明确说明（使用llama.cpp默认CUDA kernel）。
  - 硬件架构层：NVIDIA RTX 5090（本地SLM执行），云端LLM API（GPT-5/Claude Sonnet 4），无定制硬件。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **方法概述**：AIMS提出Adaptive Iteration-level Model Selector，将AI agent的subtask调度从"独立per-subtask routing"升级为"position-aware、dependency-aware的workflow-level routing"。五个核心组件：User Request Classifier（全请求级过滤）、Subtask Similarity Evaluator（subtask级fast-path）、S-L Similarity Evaluator（SLM-LLM距离预测回退路径）、Convergence Detector（未来收敛点搜索）、Subtask Decomposer（复杂subtask分解为SLM友好粒度）。所有estimator基于offline profiling数据用ModernBERT/Qwen3-0.6B + LoRA fine-tune（2小时/A100），在线推理仅需2GB VRAM。

  **缺陷→方法映射**：

  **缺陷1：HybridLLM/Minions将subtask视为独立单元，忽略subtask间依赖→早期错误路由产生cascading divergence，最终输出错误**
  → 方法（SSE + SP_SLM/SP_LLM预测比较）：AIMS不在当前subtask输出上做决策，而是用subtask predictor预测下一个subtask的输出，比较SLM和LLM对"next subtask"的预测相似度。这捕捉了路由决策对未来subtask链的影响——如果SLM生成的next subtask与LLM路径分歧，即使当前subtask输出相似也不会走SLM。Trace分析：HybridLLM failure中early divergence accumulation占53.18%，说明忽略依赖是主要失败模式。AIMS通过预测式比较提前阻断cascading error。

  **缺陷2：HybridLLM/Minions对所有subtask使用相同路由标准，不考虑stage-specific sensitivity→late-stage subtask精度敏感却与early-stage同等对待**
  → 方法（position-aware adaptive threshold κ(i)）：AIMS根据subtask序列位置动态调整相似度阈值。κ = threshold_base + min(ID, 5) · 0.02（base=0.6）。早期subtask（ID=1,2）κ≈0.62-0.64（宽松），允许更多SLM offload；后期subtask（ID≥5）κ=0.70（严格），更倾向LLM保证精度。实验验证：LLM→SLM切换在Late stage精度损失9.53% vs Early 5.25%——阈值自适应地反映了这一风险梯度。Trace分析：Minions failure中late-stage sensitivity占45.86%，因其无位置感知在后期仍aggressively offload。

  **缺陷3：HybridLLM/Minions缺乏"SLM虽当前subtask输出不同但未来可能收敛"的机制→过早放弃SLM机会导致cloud cost高**
  → 方法（SLE + CD）：AIMS引入S-L distance概念——LLM subtask的SLM"追赶距离"。通过Distance Predictor预测当前LLM subtask需多少额外SLM subtask才能匹配，再通过SP_SLM预测匹配点的输出与LLM比较。若高S-L similarity则SLM沿路径执行。CD进一步迭代搜索未来收敛点（选取最后一个以最大化SLM使用）。实验：SLE+CD联合贡献accuracy +3.72%、SLM usage +8.06%（消融实验中移除CD后）。Trace分析：HybridLLM/Minions因lack of convergence handling分别损失17.18%/24.07%的case。

  **缺陷4：复杂subtask超出SLM能力→直接走LLM损失offloading机会**
  → 方法（SD subtask decomposition）：AIMS用fine-tune的Subtask Decomposer将复杂subtask拆为更细粒度的子subtask序列（如"Verify Shirley Cameron's father, including corroborating biographical details"→拆为Search/Extract/Bio details/Confirm 4步），仅在所有子subtask都适合SLM时才整组offload。这避免了逐个子subtask单独路由可能导致的额外LLM调用。实验：SD单独贡献accuracy +1.58%、SLM usage +5.54%（消融实验中移除SD后）。SLM total usage 83.58% vs HybridLLM 52.20%。

  **缺陷5：HybridLLM以"request"为粒度做路由，忽略agent workflow是多subtask迭代过程→无法处理整请求级别的简单case**
  → 方法（URC request-level pre-filter）：AIMS在subtask routing之前先用URC判断整请求能否直接走SLM。若全请求输出similarity>0.7则跳过所有subtask routing。实验：w/o URC的SLM usage下降13.40%（from 83.58% to 70.18%），accuracy仅微降0.80%，说明URC在不牺牲精度前提下批量捕获简单请求的SLM机会。

  **论文方法全栈执行例子（HotpotQA "maternal grandfather of Titanic director" + AIMS + Qwen3-4B/GPT-5 + RTX 5090）**：
  - 算法层/Serving层（AIMS routing pipeline）：URC预测request similarity<0.7→进入subtask routing。ST1="Identify Titanic director"→SSE预测next subtask similarity→κ(1)=0.62（宽松）→SLM执行ST1，输出"James Cameron"。ST2="Find director's mother"→SSE similarity<κ(2)=0.64→SLE predict d=1（SLM多需1步可达LLM对应）→SP_SLM predicted ST3 vs SP_LLM predicted ST2 similarity>κ→SLM执行ST2→SLM自动生成ST2.5="Search James Cameron biography for mother's name"（S-L distance的额外subtask）。ST3="Confirm maternal grandfather"→SSE/SLE均失败→CD迭代搜索：SP_SLM/SP_LLM forward predict 3步→第3对similarity>0.7→收敛点在第3个future subtask→SLM执行ST3及后续2步。ST6（新生成的final confirmation）→SSE/SLE/CD全失败→SD分解为"Search Shirley Lowe's father"+"Extract father's full name"+"Find birth/death dates"+"Confirm maternal grandfather"→4个子subtask全部通过SSE→SLM整组执行。最终accuracy 90.75%、SLM usage 81.85%。
  - 编译框架层：论文未明确说明（llama.cpp默认编译，estimator PyTorch + LoRA fine-tune）。
  - kernel调度层：论文未明确说明（llama.cpp默认CUDA kernel，无定制kernel）。
  - 硬件架构层：NVIDIA RTX 5090 GPU本地运行Qwen3-4B（4-6GB VRAM），云端GPT-5 API。Estimator推理约2GB VRAM additional。调度决策overhead占总时间3-7%。网络hop latency平均0.58s。Cloud cost 0.17× vs All-LLM（83% savings）。

  **关键trade-off**：offline profiling需对每个SLM-LLM pair重新收集subtask binary tree数据并fine-tune estimator（约2小时/A100），换model pair时需重新profiling。Decomposition仅在全部分解子subtask适合SLM时才offload整组——保守策略确保了accuracy但可能错失部分子subtask可SLM而另一部分需LLM的混合case。Adaptive threshold依赖论文原实验数据拟合的参数（base=0.6, step=0.02, max=5），换模型对/dataset可能需重新调参。当前不支持显式online latency/货币budget约束——SLA-aware扩展留作future work。仅验证text-based agent（非多模态agent）。
