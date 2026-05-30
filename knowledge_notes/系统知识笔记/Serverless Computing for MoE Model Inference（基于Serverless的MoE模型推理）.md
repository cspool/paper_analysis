## Serverless Computing for MoE Model Inference（基于Serverless的MoE模型推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Serverless Computing（无服务器计算）是一种云计算范式，由云提供商（AWS Lambda、Google Cloud Functions、Azure Functions、Alibaba Cloud Functions）自动管理服务器、容器和资源的分配与伸缩。用户仅需上传代码（封装为 Docker 镜像或函数），由平台按需分配计算资源并执行。在 MoE 模型推理中，每个 expert 被部署为一个独立的 serverless function；gating network 作为单独的调度函数将 token 路由到各 expert 函数；非 MoE 层也分别部署为独立函数。Serverless 平台的核心计费模式是基于内存使用量 × 执行时间（GB-second / GBs），无请求时零计费，适合处理 MoE 推理中 expert 负载高度倾斜（热门 expert 持续被调用，冷门 expert 可能长时间空闲）的场景。

从系统架构角度拆解术语：
Serverless MoE 推理的全链路工作流（基于 AWS Lambda + S3）：
1. **模型分区**：MoE 层采用 expert parallelism（每个 expert → 1 个 Lambda function），非 MoE 层采用 model parallelism（每层 → 1 个 function）。
2. **Docker 镜像构建**：每个模型分区封装为 Docker 镜像，推送到 ECR（Docker image manager）。模型参数存储于 S3（external storage）。
3. **函数部署**：通过 Step Functions（serverless function deployment manager）将各 Docker 镜像分配到 Lambda function 并部署。部署前需预先配置内存大小（如 128MB~3072MB），决定 vCPU 数量和计费单价。
4. **推理请求处理**：推理请求从 S3 被 Lambda function 读取 → 各 function 从 S3 加载模型参数到内存 → 执行计算 → 中间结果写回 S3 供下游 function 读取。
5. **通信模式**：函数间通信有两种方式：
   - Direct invocation：函数直接调用另一函数传输数据（受 payload size 限制，如 AWS Lambda ≤6MB）
   - Indirect transfer via external storage (S3)：通过 S3 中继数据，突破 payload 限制，但增加延迟和 S3 读写成本。
6. **计费**：每个 function 按使用的内存大小（GB）× 运行时间（秒）计费。冷启动（cold start）期间同样计费。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：Python 3.8 + PyTorch + Transformers 构建模型代码 → Dockerfile 定义环境 → AWS Lambda function 配置内存 [128, 3072] MB → 通过 boto3 SDK 或 Step Functions 编排函数间调用。
- Serverless MoE 的核心优势：按 GBs 粒度计费（vs. CPU cluster 按月/小时），无空闲资源成本。该论文实验中 serverless MoE 推理的 billed cost 比 CPU cluster 降低至少 75.67%。
- Serverless MoE 的核心挑战：(1) 函数部署和冷启动需要数分钟，无法在推理过程中动态调整内存配置，必须提前预测 expert popularity；(2) 函数是 stateless 的，直接调用时每次重新触发需重新加载模型参数，pipeline 通信难以设计；(3) payload size 限制直接通信的数据量。
- AWS Lambda 默认内存与 vCPU 成正比：更大内存配置 = 更多 vCPU = 更快计算，但也意味着更高的 GBs 单价。
- 开源参考项目：LambdaML (https://github.com/DS3Lab/LambdaML) 是 serverless ML 的基准框架，对每个 function 使用最大内存配置（3008MB）。

涉及论文标题：
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing
