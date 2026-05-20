## LEGO: Supporting LLM-enhanced Games with One Gaming GPU

- 属于算法pipeline的实现是什么？实验比较什么？
  提出resource-oriented layer-skipping adaptor，在必须根据GPU资源预算跳过若干Transformer层时，用小型FFN adaptor近似被跳过层的知识表示变换，降低精度损失。核心算法设计：(1) Layer Similarity Heatmap：用cosine similarity量化所有Transformer层输出tensor间的相似度，发现后层高相似度但最后层与倒数第二层低相似（最后层编码与output layer对接的关键知识），对角线反映各跳层配置的候选层段；(2) Contiguous Layer Selection：当需跳过N层时，沿heatmap对角线选择相似度最高的连续层区间（如Llama3-8B跳4层选L25-L29，跳8层选L23-L31），避免离散跳层造成的inter-layer coherence disruption；(3) Adaptor Training：训练FFN adaptor（单层feed-forward network），输入第k层输出f_k，输出逼近第k+n层原始输出f_{k+n}，MSE loss L_mse = ||f_{k+n} - FFN^{k+n}_k(f_k)||²，仅更新adaptor权重（268.8MB/adaptor）；(4) Resource-driven skip decision：离线profile游戏rendering headroom范围→计算必须跳过的最小层数M和最多层数N→准备N-M+1个adaptor→运行时由scheduler根据预测headroom选择跳层数。实验比较：LEGO vs LITE（confidence-based early exit layer-skipping）和CALM（classifier-based layer-skipping），在MMLU/ARC-C accuracy和SQuAD-2.0 F1上对比不同跳层数（0/4/8/12/13/14）的精度。LEGO跳12层时相比LITE减少86.3% accuracy loss。还评估SmallModel baseline（Llama3-3B替换8B、Mistral-4B替换7B）。

- 硬件平台是什么，配置是什么。
  Windows 11, CUDA driver 566.36, CUDA SDK 12.1, DirectX 12.1。Intel i9-13900KF @ 3.00 GHz, Nvidia RTX 4090 (24GB)。所有游戏配置4K分辨率、高画质、60 FPS。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama3.2-8B-Instruct (FP16), Mistral-7B-Instruct-v0.3 (FP16)；扩展评估DeepSeek-V2-Lite (28层MoE), Mixtral-8x7B (MoE)。数据集：WebInstruct（adaptor训练集），MMLU/ARC-C (accuracy)、SQuAD-2.0 (F1) 作为下游评估benchmark。LLM推理代表性输入长度512 tokens，输出长度16 tokens。运行100/200/300 APM三种场景。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  开源：论文未提供官方开源仓库。基于llama.cpp (github.com/ggml-org/llama.cpp, commit fc83a9e)。算法pipeline（以Llama3-8B + BlackMyth + 200 APM为例）：
  1. 离线阶段：用WebInstruct数据集对fine-tuned Llama3-8B做推理→每层输出tensor提取→计算所有层对的cosine similarity→构建similarity heatmap→profile游戏rendering trace得H_min/H_max→计算需准备的跳层配置数N-M+1（BlackMyth最多14个adaptor）→对每种跳层N，沿heatmap对角线选择最高相似度的连续层区间→训练FFN adaptor以MSE loss优化→总训练约36小时
  2. 在线推理：Scheduler根据预测headroom选择跳过N层→将相应层段的Transformer层替换为已训练的adaptor→LLM推理时，输入f_k经adaptor直接映射到f_{k+n}→跳过N个transformer层→剩余层正常执行→输出token
  3. 跳层粒度：decode阶段每Transformer layer约0.4ms→调度以layer为粒度；prefill阶段以self-attention (0.5ms)和FFN sublayer (1.0ms)为粒度
  4. 精度保证：跳≤12层时LEGO在MMLU上保持≥40.9，优于Llama3-3B baseline (58.2)；100/200 APM下90% case仅需跳≤5层
