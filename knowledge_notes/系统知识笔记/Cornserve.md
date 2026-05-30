## Cornserve

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cornserve 是一个分布式 Any-to-Any (A2A) 多模态模型 Serving 平台。它支持通用 A2A 模型的推理部署，能够根据 Cornfigurator 生成的 physical plan 自动在 GPU 集群上实例化 executor、管理组件间数据流、执行请求路由。开源地址：https://github.com/cornserve-ai/cornserve。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Cornfigurator → Physical Plan → Cornserve Runtime
                                    ├── Plan Deployer (启动 executor 实例)
                                    ├── Request Router (按 routing prob 分发请求)
                                    ├── Data Transfer (NCD, ~10ms median)
                                    ├── Executor (运行 component 推理)
                                    └── Monitoring (per-executor 指标)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Cornserve 在 Cornfigurator 论文中被用作 evaluation platform——所有 baseline plan（vLLM, vLLM-Omni, ModServe, EPD）均通过 Cornfigurator 表达为 equivalent physical plan 后部署到 Cornserve 上运行，以消除框架实现差异对性能的影响。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
