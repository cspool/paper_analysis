## Mordal: Automated Pretrained Model Selection for Vision Language Models

- baseline方法是什么？
  Baseline是grid search（穷举搜索）——对给定的pretrained vision encoder和LLM模型库，穷举所有VE×LLM组合，对每个VLM候选用完整alignment数据训练feature projector并评估下游task性能，选择性能最优的组合。以论文中的49个候选（7 VE × 7 LLM）为例，grid search需要5439 GPU hours（每个候选~111 GPU hours，含feature projector training + LoRA fine-tuning + 6 task evaluation）。此外，开发者也会"cherrypick"——凭经验/直觉选择pretrained模型（如选最新、最大或最知名的），但这种方法不稳定且不可预测。

  全栈执行例子（以grid search在16×A40上搜索GQA最优VLM为例）：
  - 算法层：标准VLM架构——Vision Encoder（如CLIP-ViT-L/14）提取image embeddings → MLP Feature Projector对齐到LLM token space → LLM Decoder（如Vicuna-1.5-7B）生成文本。对于每个VE×LLM组合，训练MLP projector + LoRA fine-tune LLM。共49个candidate的完整训练和评估流程。
  - 系统框架层：基于LLaVA pipeline，使用HuggingFace Transformers + PEFT (LoRA) + Flash Attention-2。无搜索优化——每个candidate被独立、完整地训练。所有candidate串行或小批量并行在16×A40上执行。使用Adam optimizer（lr=1e-4, minibatch=4），LLaVA-1.5-Instruction作为alignment data。
  - 编译框架层：论文未明确说明（标准PyTorch eager execution）。
  - kernel调度层：论文未明确说明（标准A40 GPU kernel执行：matmul + attention + MLP kernels）。
  - 硬件架构层：16× NVIDIA A40 GPU (48 GB GDDR6 each)。Grid search通过数据并行或pipeline并行方式利用多GPU，但每个candidate的内部训练不涉及跨GPU分布式训练。

  Baseline根本缺陷：(1) **搜索空间爆炸**——HuggingFace上>150,000个LLM，即使精选到7×7=49个candidate也需要5439 GPU hours，扩展到更大模型库不可行；(2) **每个候选评估成本高**——每个candidate需要>100 GPU hours进行完整alignment训练，而大部分candidate最终会被淘汰（资源浪费）；(3) **无法利用候选间相似性**——grid search将每个candidate视为独立，不考虑相似VE/LLM组合可能产生相似性能，重复评估冗余候选；(4) **chicken-and-egg问题**——未alignment的VLM无法评估zero-shot性能（feature projector未训练时LLM不理解image embeddings），必须训练后才能知道性能，无法预先筛选。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出Mordal——通过两个正交优化维度解决grid search的低效：(1) 减少候选数量（Candidate Clustering）; (2) 减少每个候选的评估时间（Efficient Evaluation）。两个维度各自包含协同设计：

  **(1) Candidate Clustering**——解决"搜索空间爆炸"和"候选间冗余"：
  Mordal使用CKA (Centered Kernel Alignment) 计算VE和LLM的表示相似度，通过两步聚类构建VLM候选cluster。关键insight：相似pretrained模型的组合会产生相似性能，因此只需评估每个cluster的representative candidate即可推断cluster整体的优劣。
  - 两步聚类策略：先对VE聚类（基于vision embeddings的CKA相似度）→ 再在每个VE cluster内基于medoid VE对LLM聚类（基于LLM last hidden state的CKA相似度）→ Cartesian product生成VLM候选cluster
  - Inter-cluster evaluation：仅评估每个cluster的medoid候选，淘汰性能差的cluster，大幅减少需评估的candidate数量
  - Intra-cluster evaluation：仅在remaining Top-K cluster内进行candidate-granularity评估
  - 为什么CKA有效：(a)可比较不同shape的表示（不同VE/LLM的输出维度不同，cosine similarity不可用）; (b)对MLP projection变换鲁棒（feature projector的MLP层不影响CKA的性质）

  **(2) Efficient Evaluation (Early Stopping + Scaling Prediction)**——解决"每个候选评估成本高"：
  两个互补机制减少单个candidate的评估时间：
  - **Early Stopping (SHA)**：在inter-cluster阶段使用Successive Halving Algorithm。每轮以固定budget (b)评估所有representative candidates，保留top 1/η，下一轮增加budget (b×η)，循环淘汰直到收敛。效果：差候选在少量数据训练后即被淘汰，资源集中于有潜力的候选。
  - **Scaling Prediction**：利用发现的observational scaling law——VLM alignment性能与训练数据量存在log-linear关系（即log(Err)与log(data_ratio)线性相关）。对每个intra-cluster candidate，从少量数据(如1/8)开始训练并逐步减少数据量，收集(log(r), log(Err))点对，拟合线性回归模型f_c，预测r=1（完整数据）时的性能。效果：无需完整训练即可预测最终性能。
  - 关键发现：log-linear scaling仅在训练数据量超过一定阈值后出现（consistent with prior work），因此Scaling Prediction从大ratio向小ratio递减（利用已有checkpoint节省计算）。

  全栈执行对比baseline（以Mordal在16×A40上搜索GQA最优VLM为例）：
  - 算法层：同一VLM架构和49 candidates。Mordal不改变模型训练本身，而是改变搜索策略——通过CKA-based clustering将49 candidates分为~10-15 clusters → Inter-cluster SHA评估representatives → 淘汰差cluster → 保留Top-3 cluster → Intra-cluster Scaling Prediction → 选出最优candidate。Search time从5439h降至483h（11.2× speedup），成功选出Top-1 candidate (SigLIP-Vicuna, 66.4% accuracy)。
  - 系统框架层：基于LLaVA + PyTorch + HuggingFace + PEFT + Flash Attention-2。Mordal提供统一Python接口 (`import mordal` → `mordal.query_for_model()`)，用户提供alignment data + target task data + model zoo。Mordal自动管理：clustering（CKA计算 + hierarchical clustering）→ inter-cluster training + SHA filtering → intra-cluster training + scaling prediction → best model selection。空闲GPU自动分配给未收敛candidate加速搜索。
  - 编译框架层：论文未明确说明（标准PyTorch eager execution）。
  - kernel调度层：论文未明确说明（标准A40 GPU kernel执行）。
  - 硬件架构层：16× NVIDIA A40 GPU (48 GB GDDR6)。Search time breakdown（Figure 11）：early stopping占大部分时间（因需实际训练部分数据），scaling prediction仅占小部分（仅对remaining candidates进行）。

  设计思路核心：Mordal的本质是将pretrained model selection从"穷举+完整训练"重构为"聚类粗筛+预测精排"的两阶段pipeline。CKA-based clustering利用表示相似度将候选分组（类似分层采样的思想），SHA在cluster级别做粗粒度淘汰，Scaling Prediction在candidate级别利用log-linear scaling做细粒度排序。两个关键科学发现支撑这个pipeline：(1) 相似pretrained模型的VLM性能也相似（使clustering可行）; (2) VLM alignment存在observational scaling law（使prediction可行）。Mordal证明了VLM pretrained model selection问题可以通过systematic algorithmic framework高效解决，而非依赖人工经验或暴力搜索。
