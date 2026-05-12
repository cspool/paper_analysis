论文标题：TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation

端侧SLM、云中心LLM的协同调度，复杂请求到云端、简单请求在端侧处理。


开源仓库确认：
    - 状态：未找到明确开源仓库
    - 链接：N/A
    - 说明：本地 PDF、DOI 页面元信息和 EuroSys 2026 accepted papers 页面均能确认论文条目，但论文正文没有给出 TailorLLM 的 GitHub、artifact appendix 或代码包链接。本次外部检索只找到 ACM / EuroSys 论文页面和不相关 LoRA 项目，未找到作者发布的官方实现。因此截至本次分析，不能确认 TailorLLM 已公开代码。相关公开页面：https://2026.eurosys.org/papers.html；https://doi.org/10.1145/3767295.3769346

1、论文工作：
    - 论文要解决的核心问题：LLM 推理服务的用户规模增长让纯云端 LLM 部署承担高云 GPU 成本；纯端侧 SLM 虽然低延迟、低成本，但在复杂任务上准确率不足。已有端云协同方法也难同时兼顾多任务准确率、端到端延迟和云计算成本：模型切分方法需要在端和云之间传输中间激活，弱网下通信开销高；token 级 SLM/LLM 协同类似 speculative decoding，需要云端频繁验证小模型生成的 token，一次问答可能触发多轮端云通信，网络 RTT 会累计成显著延迟，并抵消部分云成本节省。
    - 论文的主要贡献：论文提出 TailorLLM，一个基于低秩适配的 task-level 端云协同推理系统。它利用两个观察：用户请求往往集中在少数高频任务，且任务类型随时间呈现比 token 级生成更强的周期性。TailorLLM 在端侧运行 Llama3-1B 级 SLM，并为常见任务动态加载 LoRA / RFLoRA adapter；复杂、罕见、新类别或端侧 SLM 准确率不足的请求则 offload 到云端 Llama3-70B。核心算法包括离线的 RFLoRA，用共享冻结矩阵与方向 / 幅度分解减少 adapter 传输和存储；以及在线的 AdapterMgr，用 imitation learning 逼近 Belady cache replacement，管理端侧 LoRA RAM 缓存。
    - 论文所处背景：该工作位于 LLM serving、edge-cloud collaborative inference、parameter-efficient fine-tuning、LoRA adapter 管理和缓存替换策略的交叉点。论文面向资源受限端侧设备与云端 LLM 服务之间的协同推理场景，特别是用户请求具有长尾分布和周期性偏好的多任务应用，如翻译、摘要、情感分析、问答、语法判断等。

2、相对 Baseline 解决的问题与设计方法：
    - Baseline 的具体问题：cloud-only 基线准确率稳定但云端成本和网络响应延迟高；end-only 基线延迟低但 Llama3-1B 这类 SLM 在多任务，尤其数学推理等复杂任务上准确率显著不足。HSL 这类 token-level 端云协同需要小模型生成若干 token 后由云端 LLM 验证，通信频率高，RTT 增大时 TTFT / TPOT 明显恶化。Petals / model partitioning 基线将 LLM 层切到端和云上执行，但 decoder 堆叠模型的分割会引入跨设备中间结果传输，并受端侧显存和无线网络影响。标准 LoRA 虽比全模型轻，但当多个任务 adapter 需要在端云间更新时，22MB 级别的单任务 adapter 仍会形成显著传输和 RAM 压力。LRU 等简单缓存策略只利用最近访问，不能利用任务级周期模式。
    - 论文的设计方法：TailorLLM 把协同粒度从 token 级提升到 task 级。在线阶段先用 Contriever 语义编码、UMAP 降维和 HDBSCAN 聚类做开放类别任务识别；Allocator 根据任务类别、端侧 SLM 对该任务是否满足准确率要求、对应 LoRA 是否在本地缓存，决定本地执行还是云端执行；AdapterMgr 使用用户历史访问序列和当前 LoRA 缓存状态作为双模态输入，用单层 Mamba block 提取时间序列特征，并用 Belady 离线最优替换策略生成 imitation learning 标签，预测应替换哪个 LoRA slot。离线阶段，RFLoRA 观察到不同任务 LoRA 中 A 矩阵更趋同、B 矩阵更任务相关，因此冻结并共享 A，同时将预训练权重按列范数分解为 direction 和 magnitude，只训练 B 与幅度参数 m，使端云传输只需发送任务相关的 B 和 m。
    - 方法如何对冲 Baseline 缺陷：相对 cloud-only，TailorLLM 让约 70% 高频且端侧可胜任的任务在本地完成，直接减少云端 token 生成量和网络往返。相对 end-only，它用任务分类和准确率表识别 SLM 不擅长的任务，并将 GSM8K 等复杂任务转给云端，避免端侧小模型硬做导致准确率崩塌。相对 HSL，TailorLLM 不在每几个 token 后请求云端验证，而是对整个任务做一次路由，因此对 RTT 更不敏感。相对 Petals，TailorLLM 不把单个 decoder forward path 切分到端云两侧，避免频繁跨设备传递中间激活。相对标准 LoRA，RFLoRA 将 Llama3-1B 的单 adapter 大小从约 22MB 降到 11.56MB，并把 trainable parameters 降到约 0.273% 的全模型参数。相对 LRU，AdapterMgr 用历史周期模式和缓存状态联合预测，提升端侧 adapter hit rate，从而提高本地完成比例。
    - 关键 trade-off：TailorLLM 以任务集中性和可预测性作为前提：如果用户任务高度长尾、频繁出现新类别，或 30 类以上开放分类规模过大，分类准确率和 AdapterMgr 命中率都会下降，更多请求会回退到云端。RFLoRA 虽降低 adapter 传输，但 adapter 仍需在端云间传输，论文也承认弱带宽下可能带来高延迟或传输失败；当 SLM 基座模型更新时，已有 LoRA adapter 需要重新训练。系统还需要维护任务准确率表、LoRA 云端库、端侧 RAM 缓存和新任务聚类 / 微调流程，因此把一部分复杂度从推理路径转移到了离线 adapter 训练和在线缓存管理。

3、论文实现：
    - Baseline 如何实现：cloud-only 使用纯云端 Llama3-70B；end-only 使用纯端侧 Llama3-1B；HSL 使用端侧 SLM 生成 draft、云端 LLM 验证的 token-level 协同方案，实验中设置为每 5 个 token 验证一次；Petals 代表模型切分式端云协同，论文在端侧显存约束下将 Llama3-70B 按 5:65 层切分到端侧与云侧；TailorLLM-LoRA 用标准 LoRA 替换 RFLoRA 做消融；TailorLLM-LRU 用 LRU 替换 AdapterMgr 做消融。AdapterMgr 的调度对比还包括离线最优 Belady、LRU 和 Parrot。
    - 新设计如何实现：TailorLLM 原型包含云侧和端侧两部分。云侧保存 LLM 和 RFLoRA adapter library，负责处理复杂 / 未命中 / 新类别任务，也负责为积累到足够样本的新任务微调 adapter。端侧运行 SLM、任务分类器、Allocator、AdapterMgr 和 adapter cache。任务分类器由 Contriever embedding、UMAP 三维降维和 HDBSCAN 组成；AdapterMgr 的输入是长度 H=100 的历史访问序列和容量 w=5 的 LoRA 缓存状态，embedding 隐层维度 d=128，时间特征提取器为单层 Mamba block，训练标签来自 Belady，loss 为 BCE。RFLoRA rank r=16；A 矩阵在不同任务间共享并冻结，只为每个任务保存 / 传输 B 与 magnitude m。
    - 实验 / 实现平台：云侧平台为 4 张 NVIDIA RTX 3090 GPU，每张 24GB GDDR6X，Ubuntu 20.04；端侧平台为 NVIDIA Tesla T4 GPU，16GB 显存，并通过 visible resource constraint 将可见显存限制为 10GB 来模拟资源受限 edge device，端侧同样运行 Ubuntu 20.04。端云之间使用无线网络，标准测试中 RTT 约 47ms。模型为 Llama3-1B 作为 SLM、Llama3-70B 作为云端 LLM。评估数据集包括 GSM8K、MRPC、COLA、QNLI、RTE、SST-2、MNLI、QQP、BoolQ；训练 / 测试按 8:2 划分，并构造周期性多任务用户访问序列。AdapterMgr 额外在 MovieLens 行为数据和合成周期数据上评估。
    - 关键实验设置与指标：主要指标包括 cloud computing overhead、average multi-task accuracy、end-to-end latency，并细分 TTFT 和 TPOT；云成本按 GPT-4o API 价格估算，网络传输成本按 AWS $0.09/GB 估算。TailorLLM 相比 cloud-only 最多节省约 69.8% 云计算成本，端到端延迟最多降低约 62%，同时准确率接近纯云方案。RFLoRA 在 8 个非 GSM8K 任务上的平均准确率为 81.6%，使用约 0.273% 参数，略高于 LoRA / AdaLoRA / HydraLoRA，并低于 Llama3-70B 的 85.1%。分类模块在 10 / 15 类时准确率约 0.969 / 0.957，未分类率低于 5%；扩展到 30 类时准确率降到 0.736，未分类率升到 32.8%，说明设计更适合少数高频任务。端侧开销方面，任务分类约 0.45-1.53ms，LoRA switching 约 0.26ms，论文认为相对 22.6ms/token 的推理延迟占比约 2-7%。需要注意的是，正文中关于 Llama3-1B 与 Llama3-70B ms/token 的表述与“SLM 更快”的论证存在方向上不够清晰之处，本分析按论文给出的实验结论理解为本地 SLM 路径带来总体延迟优势。

4、pipeline/kernel解析：
    - 新pipeline/kernel是什么：论文没有提出新的 GPU kernel、attention kernel 或算子级 dataflow；最接近的新执行流是 TailorLLM 的 task-level end-cloud collaborative inference pipeline，以及配套的 RFLoRA adapter 训练 / 分发路径。这个 pipeline 的关键不是优化单次矩阵乘或 decode kernel，而是改变端云协同的粒度：从“每隔若干 token 让云端验证”改为“先识别任务，再决定整条请求在端侧 SLM+adapter 还是云端 LLM 上执行”。
    - 新pipeline/kernel的执行流例子：假设用户发起一个摘要请求。端侧首先用 Contriever 将请求编码成语义向量，UMAP 降维后交给 HDBSCAN 判断任务类别；如果类别已知，Allocator 查询该任务的端侧 SLM 准确率是否达标。如果达标且对应 RFLoRA adapter 已在端侧 RAM cache 中，端侧加载该 adapter 到 Llama3-1B，直接本地自回归生成结果，请求不再与云端逐 token 交互。如果 adapter 不在 cache 中，AdapterMgr 查看最近 H=100 个任务访问序列和当前 w=5 个 adapter slot，预测是否应从云端下载该任务 adapter 以及替换哪个 slot；下载时 RFLoRA 只传输任务相关 B 和 m，而共享 A 已预存在端侧。若任务是 GSM8K 这类复杂任务、分类器标记为新类别 / 未分类，或 SLM 对该任务不满足准确率要求，请求直接发送到云端 Llama3-70B 执行。随着更多新类别请求被收集成密集簇，云端可为该类别离线 fine-tune 新 RFLoRA adapter，并加入 cloud adapter library，后续再由 AdapterMgr 预取到端侧。
