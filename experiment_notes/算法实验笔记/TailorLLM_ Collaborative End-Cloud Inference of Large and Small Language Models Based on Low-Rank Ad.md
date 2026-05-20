## TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

- 属于算法pipeline的实现是什么？实验比较什么？
  RFLoRA (Resource-Friendly Low-Rank Adaptation)：一种参数高效的LoRA变体，将预训练权重W解耦为direction和magnitude两个分量（W = m · W/||W||_c），仅对direction分量应用LoRA低秩分解（BA），magnitude标量m和B矩阵可训练，A矩阵冻结并在所有任务间共享。核心发现：(1) 跨任务微调时A矩阵趋于收敛（capture domain-invariant features），B矩阵呈现任务特异性（adapt to domain-specific variations）；(2) 权重可分解为方向和幅度分量，分别优化加速收敛。RFLoRA使端侧只需预存一份共享A矩阵，传输时仅需发送任务特定的B矩阵+m（相比标准LoRA减少~50%传输/存储开销）。实验比较：在Llama3-1B上用8个NLP任务（MRPC/COLA/QNLI/RTE/SST-2/MNLI/QQP/BoolQ）对比Llama3-1B (无微调)、Llama3-70B、LoRA、DoRA、AdaLoRA、HydraLoRA。GSM8K数学任务因1B模型无法通过LoRA达到可接受精度而自动卸载到云端。

- 硬件平台是什么，配置是什么。
  Cloud-side: 4×NVIDIA RTX 3090 GPU (24GB GDDR6X)，Ubuntu 20.04 LTS。End-side: NVIDIA Tesla T4 GPU (16GB limited to 10GB)，Ubuntu 20.04 LTS。微调训练在云端RTX 3090上执行。

- 模型是什么。数据集和bench分别是什么。
  SLM: Llama3-1B。LLM: Llama3-70B。数据集：GSM8K（数学推理，自动卸载到云）、MRPC（语义等价）、COLA（语法可接受性）、QNLI（自然语言推理）、RTE（文本蕴涵）、SST-2（情感分析）、MNLI（多体裁推理）、QQP（查询等价）、BoolQ（是否问答）。80% fine-tune / 20% test split。所有PEFT方法共享相同训练schedule、数据split和超参数（r=16，HydraLoRA因r=16下梯度爆炸使用r=32）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline。
  论文未明确说明开源。RFLoRA算法pipeline：
  1. 预训练权重分解：对LLM每层weight matrix W∈R^{d×k}，计算column-wise norm m=||W||_c∈R^d，direction矩阵 V=W/m（即W/||W||_c column-wise normalization）。初始W = m · V/||V||_c。
  2. LoRA低秩注入direction：仅对direction分量V施加ΔV = B·A。A∈R^{r×k} Kaiming初始化并全局冻结（跨任务共享），B∈R^{d×r} zero初始化可训练。magnitude m可训练。
  3. 更新公式：W' = m · (V + ΔV) / ||V + ΔV||_c = m · (W_0 + B·A) / ||W_0 + B·A||_c。反向传播仅更新m和B。
  4. 传输：云端训练完成后仅传输B矩阵+magnitude参数m给端侧（~11.56MB per adapter，原始LoRA ~22MB）。端侧预存一份共享A矩阵。
  5. 端侧推理：加载对应任务B+m→与预存A构成完整LoRA→注入SLM→推理。
  6. 结果：RFLoRA 81.6% avg accuracy (3.4M trainable params, 0.273% of full model)，vs LoRA 81.2% (0.454%)、DoRA 82.1% (0.484%)、AdaLoRA 81.0% (0.680%)、HydraLoRA 81.2% (1.277%)。在trainable params远少于DoRA的情况下精度接近。将Llama-1B与Llama-70B的精度差距从28.2缩小到3.5个百分点。
