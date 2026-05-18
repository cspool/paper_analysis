## Serverless LLM Inference

术语是什么？

Serverless LLM Inference是将serverless计算范式应用于大语言模型推理服务的架构模式：云平台托管大量用户上传的LLM模型，请求到达时按需分配计算资源（GPU/CPU）启动model instance执行推理，请求完成后回收资源。与传统的dedicated model serving不同，serverless场景具有两大特征：(1) small- to mid-sized models占主导——HuggingFace上87%下载量是≤8B参数的模型；(2) 请求模式高度稀疏和突发——LMSYS数据集中56%的模型每小时平均请求数<5。代表性系统包括ServerlessLLM (OSDI'24)、Medusa、DeepServe、ParaServe等。

从系统架构角度拆解术语：

SLINFER论文描述的serverless LLM推理流程：
1. **模型注册**：用户将LLM上传至云平台，平台将模型权重缓存在共享存储（如CPU memory pool）中以加速冷启动
2. **请求到达**：用户请求到达系统proxy，携带input tokens
3. **Instance分配**：系统检查是否存在该模型的运行中instance → 若有且资源充足则将请求路由到现有instance；若无则在可用硬件上启动新instance（cold-start latency ~1s for 7B model with fast loading）
4. **请求执行**：instance使用vLLM/OpenVINO等inference engine执行prefill+decode迭代，按SLO要求stream回output tokens
5. **Instance回收**：请求完成后instance保持alive一段时间（keep-alive threshold，如1s），超时后回收资源
6. **资源管理**：多model竞争有限GPU/CPU资源，系统需在高部署密度和SLO compliance间平衡

核心挑战：(1) GPU资源稀缺——ServerlessLLM在4×A100上serving 64个3B-13B LLM时33%请求SLO违规，因GPU数量不足；(2) memory over-provisioning——每instance独占整GPU但平均仅用23% memory；(3) CPU资源闲置——GPU节点上CPU大多数核心空闲。

术语一般如何实现？如何使用？

Serverless LLM推理系统的典型实现：
- **Event-driven架构**：请求到达触发instance创建/路由，请求完成触发资源回收
- **Fast model loading**：ServerlessLLM通过多级缓存（本地SSD→CPU memory→GPU memory）和并行I/O加速模型加载，将7B模型冷启动降至~1s
- **Keep-alive策略**：instance在请求完成后保持alive短暂时间以吸收后续请求，减少冷启动频率
- **资源抽象**：SLINFER将异构硬件（CPU/GPU）统一为resource pool，请求优先路由到低功耗CPU instance、SLO不满足时fallback到GPU

涉及论文标题：
- Towards Resource-Efficient Serverless LLM Inference with SLINFER
- FlexPipe: Adapting Dynamic LLM Serving Through Inflight Pipeline Refactoring in Fragmented Serverless Clusters

FlexPipe补充的serverless GPU碎片化视角：在Alibaba生产集群中，GPU平均订阅率达216%（两服务共享一张GPU），仅8.7%概率获得单张>85%空闲显存的GPU，同机4卡可用概率仅0.02%。78%的tensor parallelism请求被迫退化为pipeline parallelism。Serverless的反亲和调度（anti-affinity policies）将service replica分散到不同物理节点，与分布式LLM推理所需的tightly coupled GPU集群根本冲突。Always-on资源通常维持历史峰值60-75%的GPU容量，导致平均SM utilization仅17%，而弹性扩容的多秒延迟违反交互式LLM的亚秒SLO。
