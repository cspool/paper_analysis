## TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

- 属于Serving调度的实现是什么？实验比较什么？
  论文实现TailorLLM，一个基于LoRA的task-level端云协同LLM推理系统。核心Serving调度实现包含三个在线模块：(1) Task Classifier：使用Contriever semantic encoder提取高维语义特征→UMAP降维→HDBSCAN层次密度聚类实现无监督动态任务分类，支持开放类别识别（无需重训），在15分类benchmark中准确率>95%；(2) Task Allocator：根据分类结果查表判断SLM是否满足任务精度要求+本地cache是否有对应LoRA，两者都满足则端侧推理，否则卸载到云端LLM；(3) AdapterMgr：基于imitation learning的LoRA library动态管理算法，使用Mamba (SSM)提取用户历史访问序列的时序特征，融合当前端侧LoRA cache状态（双模态embedding+projection融合），以Belady最优替换策略为学习目标，通过BCE loss训练，在端侧存储空间有限（capacity w=5）条件下动态决策LoRA的加载/淘汰。滑动窗口H=100，embedding维度d=128。实验比较：(a) end-to-end性能：cloud computing cost ($/1k queries)、multitasking accuracy、end-to-end latency (s/query)、total cost including transmission；(b) RTT影响：20/50/100/200ms下TTFT和TPOT；(c) microbenchmarks：AdapterMgr hit rate vs Belady/LRU/Parrot on MovieLens和构造数据集（cycle 30/200）；(d) ablation：TailorLLM-LoRA (标准LoRA替代RFLoRA)、TailorLLM-LRU (LRU替代AdapterMgr)。对比baseline：cloud-only (Llama3-70B)、end-only (Llama3-1B)、HSL (token-level speculative decoding，每5 token验证)、Petals (模型拆分5:65)、TailorLLM-LoRA、TailorLLM-LRU。

- 硬件平台是什么，配置是什么。
  Cloud-side: 4×NVIDIA RTX 3090 GPU (24GB GDDR6X)，Ubuntu 20.04 LTS。End-side: NVIDIA Tesla T4 GPU (16GB，通过visible resource constraint限制为10GB模拟资源受限edge设备)，Ubuntu 20.04 LTS。Tesla T4算力约为RTX 3090的1/6。端云通过无线网络连接，标准RTT 47ms。

- 开源Serving框架是什么。修改了什么。
  论文未基于现有开源Serving框架（如vLLM/SGLang），而是自行构建端侧和云侧prototype系统。系统组成：(1) 端侧：部署Llama3-1B + LoRA adapter加载/切换 + Task Classifier (Contriever+UMAP+HDBSCAN) + AdapterMgr (单层Mamba Block) + Allocator逻辑；(2) 云侧：部署Llama3-70B + LoRA library存储。端云间通过无线网络传输LoRA adapter parameters (每个adapter经RFLoRA压缩后约11.56MB)。论文未明确说明是否开源。

- 开源情况。Serving框架如何使用？作用是什么？基于开源文档和论文，使用例子解释。
  论文未明确说明开源。以TailorLLM serving流程为例：
  1. 离线阶段：对9个下游任务数据集(80% fine-tune/20% test)用RFLoRA训练task-specific LoRA adapters→存入云端LoRA library。RFLoRA冻结共享A矩阵，仅存储任务特定的B矩阵+magnitude参数m（相比标准LoRA减少约50%传输参数）。
  2. 在线推理：用户query到达→Task Classifier提取Contriever semantic embedding→UMAP降维到3维→HDBSCAN密度聚类判定任务类别（已知类别/新类别/uncertain标记为-1）。
  3. Allocator查表：若SLM在该任务accuracy达标且本地cache有对应LoRA→加载LoRA到SLM→端侧完成推理（Llama3-1B 22.6ms/token）。否则卸载到云侧Llama3-70B推理（5.3ms/token）。
  4. AdapterMgr后台运行：维护滑动窗口H=100的历史访问序列→Mamba提取时序特征→融合当前cache state→MLP+Softmax输出每个cache slot的替换概率→决定是否从云端下载新LoRA替换某个slot。以Belady最优策略为训练目标。
  5. 效果：约70%请求端侧处理，cloud cost减少69.8%，end-to-end latency减少62%（vs cloud-only）。TTFT在不同RTT下保持低水平，TPOT在RTT 20→200ms时仅1%性能退化（vs HSL 22%、Petals 46%）。

