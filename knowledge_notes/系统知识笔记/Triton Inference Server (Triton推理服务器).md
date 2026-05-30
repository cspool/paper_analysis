## Triton Inference Server (Triton推理服务器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Triton Inference Server（原名 TensorRT Inference Server）是 NVIDIA 开源的高性能推理服务框架，支持多种深度学习框架（TensorFlow、PyTorch、TensorRT、ONNX 等）和多 GPU 部署，提供 dynamic batching、concurrent model execution、模型版本管理、请求调度等生产级推理服务功能。论文 "Who Says Elephants Can't Run" 将优化后的 FasterTransformer MoE 模型部署在 Triton Inference Server 上，利用 Triton 的 dynamic batching 和弹性扩缩容实现云规模生产部署。

Triton 通过 C API / gRPC / HTTP 对外服务，内部管理多个 backend（FasterTransformer, TensorRT, PyTorch, ONNX 等）的模型实例生命周期。Key features: dynamic batching（将到达的独立请求组合为 batch 以提高 GPU 利用率）、concurrent model execution（多个模型或同一模型多个版本并行运行）、model ensemble（将多个模型串联为 pipeline）。

从系统架构角度拆解术语：

Triton Inference Server 处理 MoE 翻译请求的流程：
```
# Triton 作为推理服务入口
Client → HTTP/gRPC → Triton Server
                        │
                        ├── Scheduler: Dynamic Batching
                        │   ├── 收集到达的翻译请求
                        │   ├── 当 accumulated requests ≥ threshold 或 timeout
                        │   └── 组成 batch → dispatch to backend
                        │
                        ├── Backend Manager: FasterTransformer Backend
                        │   ├── 管理模型实例（instance group）
                        │   ├── GPU instance: 加载优化后的 FT MoE 模型
                        │   └── 请求分发（round-robin / minimal pending）
                        │
                        └── FasterTransformer Backend
                            ├── Tokenize (SentencePiece)
                            ├── Encoder forward (24 layers, MoE)
                            ├── Decoder forward (12 layers, MoE + Batch Pruning)
                            ├── De-tokenize
                            └── Return translation → Triton → Client
```

Triton 如何帮助 MoE 生产部署：(1) Dynamic batching 减少 CPU-GPU kernel launch overhead，增加有效 batch size；(2) Concurrent execution 允许同时服务多种语言对或模型版本；(3) Instance group auto-scaling 根据请求流量自动增减 GPU 实例；(4) Model versioning 支持 A/B 测试和 rolling update；(5) 与 Kubernetes 集成可实现真正的云规模弹性扩缩容。

术语一般如何实现？如何使用？

开源在 https://github.com/triton-inference-server/server。配置通过 `config.pbtxt` 文件定义模型、backend、batch size、instance group 等。论文中 Triton 部署 5.32B MoE INT4 模型到单卡 T4 GPU，通过 dynamic batching 聚合请求提升 batch size 至 20-64，月成本 $0.153/token（vs CPU $0.209/token）。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
