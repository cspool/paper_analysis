## AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现AIMS（Adaptive Iteration-level Model Selector），一个面向云边混合环境的AI agent subtask调度框架。核心Serving调度实现包含五个离线训练的estimator和四阶段在线决策流程：(1) User Request Classifier (URC)：用ModernBERT fine-tune预测整个请求在SLM和LLM上的输出相似度，若相似则全请求走SLM，否则进入subtask-level分配；(2) Subtask Similarity Evaluator (SSE)：用SP_SLM和SP_LLM两个subtask predictor（Qwen3-0.6B + LoRA fine-tune）预测当前subtask的下一个subtask，比较两者输出的SBERT cosine similarity，超过自适应阈值κ则分配SLM。阈值κ = threshold_base + min(ID, 5) · 0.02（threshold_base=0.6），随subtask位置后移而收紧；(3) S-L Similarity Evaluator (SLE)：用Distance Predictor（ModernBERT + LoRA）预测当前LLM subtask的S-L distance d，再用SP_SLM预测第(i+d)个subtask输出与LLM第i个输出比较——若相似则SLM处理当前及后续d个subtask；(4) Convergence Detector (CD)：从当前subtask起迭代预测SLM和LLM的未来subtask序列，寻找S-L similarity超过κ的收敛点，选取最后一个收敛点以最大化SLM使用；(5) Subtask Decomposer (SD)：将复杂subtask用Qwen3-0.6B fine-tune的模型分解为更简单的子subtask序列，仅在全部子subtask都适合SLM处理时才将整组分配给SLM，否则交LLM。在线决策遵循fast-path/slow-path模式：URC→SSE（fast-path）快速判定可走SLM的请求/subtask，SLE/CD/SD（slow-path recovery）在fast-path失败时寻找更多SLM机会。整个estimator栈约需2GB VRAM。实验比较：(a) accuracy和SLM usage：9个benchmark上vs HybridLLM（classifier-based independent subtask routing）、Minions（confidence-based routing）、All-SLM、All-LLM、Oracle、Random；(b) end-to-end latency breakdown（SLM latency/LLM latency/method&network overhead）；(c) 归一化remote cost（All-LLM=1.0）；(d) 泛化性：跨模型对（Qwen3-4B+GPT-5 / Gemma3-4B+Claude Sonnet 4）、跨硬件（RTX 5090 / iPhone 15）；(e) 消融实验：逐一移除SD/CD/SLE/SSE/URC组件；(f) 训练数据量消融（0%-100% traces）；(g) estimator性能（准确率+latency ratio）；(h) 参数敏感性（τ_req 0.50-0.78、τ_sse 0.60-0.80）；(i) trace-level failure analysis（early divergence accumulation / late-stage sensitivity / lack of convergence and decomposition）。

- 硬件平台是什么，配置是什么。
  本地设备：NVIDIA RTX 5090 GPU，SLM用llama.cpp执行占用4-6 GB VRAM。云端LLM通过public API访问。离线fine-tuning：cloud-based Nvidia A100（全部estimator fine-tune约2小时）。移动端泛化测试：iPhone 15。Estimator栈约需2GB VRAM，可在8-16GB显存游戏本上运行。

- 开源Serving框架是什么。修改了什么。
  基于AutoGen构建agent stack。论文在AutoGen之上实现了AIMS scheduler层：(1) 新增request-level classifier（URC），在agent收到请求后先判定是否整请求走SLM；(2) 新增subtask-level routing逻辑（SSE/SLE/CD/SD），替代AutoGen默认的固定模型选择，对agent workflow中每个subtask的生成和执行位置做动态决策；(3) 新增offline profiling pipeline：生成subtask binary tree（每个节点用SLM和LLM分别处理、递归分支至深度15或完成），收集SLM/LLM的subtask输出、S-L distance、subtask decomposition数据用于训练estimator。agent的tool execution部分（如Wikipedia search/WebShop navigation/code interpreter）未修改，AIMS专注于"语言模型调用"这一环节的SLM/LLM选择。论文未明确说明是否开源。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未提供明确开源仓库URL（EuroSys '26 DOI: https://doi.org/10.1145/3767295.3803622，未在正文/artifact appendix中找到GitHub链接）。以AIMS在RTX 5090上serving HotpotQA请求为例（基于论文描述和AutoGen文档重建）：
  1. 离线profiling：对WorFBench和GSM8K的1000条subtask traces→生成SLM/LLM binary tree→收集输出similarity/S-L distance数据→fine-tune URC (ModernBERT)、DP (ModernBERT)、SP_SLM/SP_LLM (Qwen3-0.6B)、SD (Qwen3-0.6B)，全部用LoRA fine-tune，约2小时/A100。
  2. 线上请求："Who was the maternal grandfather of the director of Titanic (1997)?"→URC预测整请求similarity score若>0.7则SLM处理；否则进入subtask routing。
  3. AutoGen agent生成第一个subtask ST1="Identify the director of Titanic (1997)"→SSE用SP_SLM/SP_LLM预测ST1的next subtask→比较similarity→若>κ（early stage κ≈0.62）则SLM执行ST1。
  4. ST1 SLM执行完生成ST2="Find the director's mother"→SSE判定similarity<κ→SLE预测d=1（SLM多需1个subtask可达LLM对应输出）→SP_SLM预测第3个subtask vs SP_LLM预测第2个subtask→similarity>κ→ST2和ST3走SLM。
  5. ST4="Verify and confirm maternal grandfather"→SSE/SLE均失败→CD迭代预测无收敛点→SD将ST4分解为4个子subtask（Search/Extract/Find bio details/Confirm name）→全部子subtask通过SSE→整组走SLM。
  6. 效果：AIMS accuracy 90.75% vs HybridLLM 76.35% (+14.40%)、Minions 84.20% (+6.55%)，SLM usage 81.85% vs HybridLLM 68.40%、Minions 74.10%。端到端延迟13.33s（All-LLM 15.82s、HybridLLM 12.98s）。Scheduler决策overhead占总时间3-7%，网络hop latency平均0.58s可忽略。83% cloud cost savings vs All-LLM。

