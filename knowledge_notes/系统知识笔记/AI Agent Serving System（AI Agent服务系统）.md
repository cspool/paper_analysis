## AI Agent Serving System（AI Agent服务系统）

术语是什么？通过联网搜索让回答具体和精准。

AI Agent Serving System是专门为AI agent workload（多轮LLM推理+外部工具调用的动态推理）设计的在线服务基础设施。与传统的chatbot serving（单次LLM inference per request）不同，agent serving系统需要处理：(1) 每请求多次LLM call和tool call的交替迭代；(2) tool等待期间GPU idle的不规则资源利用；(3) 多轮交互积累的长context导致的KV cache膨胀；(4) 请求间计算量高度动态的bursty workload pattern。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

AI Agent Serving System的架构和请求流（以论文AgentBench + vLLM backend为例）：

```
// Agent Serving System架构
┌──────────────────────────────────────────────────────┐
│                   Agent Server                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Agent Worker │  │ Agent Worker │  │ Agent Worker │  │
│  │  (ReAct)    │  │  (Reflexion) │  │   (LATS)    │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         │                │                │          │
│         └────────────────┼────────────────┘          │
│                          │ LLM inference requests    │
│                          ▼                           │
│              ┌───────────────────────┐               │
│              │   vLLM Backend        │               │
│              │  (FCFS Scheduler +    │               │
│              │   Continuous Batching │               │
│              │   + Prefix Caching)   │               │
│              └───────────────────────┘               │
│                          │                           │
│         ┌────────────────┼────────────────┐          │
│         ▼                ▼                ▼          │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐     │
│  │Wikipedia │  │ Wolfram Alpha│  │   Python   │     │
│  │   API    │  │     API      │  │  Executor  │     │
│  └──────────┘  └──────────────┘  └────────────┘     │
│                  Tool System                         │
└──────────────────────────────────────────────────────┘
```

单请求执行流（ReAct + HotpotQA）：User Query → Entrypoint → Agent Worker → LLM call #1 (vLLM) → output: Action=Search["query"] → Tool: Wikipedia API (~1.2s, GPU idle) → observation → LLM call #2 (prefill reuses prefix cache, input增长了LLM/Tool history tokens) → ... → Final Answer。论文测量：agent serving峰值throughput远低于ShareGPT（ReAct HotpotQA/WebShop 2.6/1.2 QPS vs ShareGPT 6.4 QPS），单请求GPU idle最高54.5%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Agent serving系统通常：(1) Agent Server entrypoint接收请求，管理worker pool；(2) worker根据agent type维护状态机；(3) LLM backend使用vLLM/SGLang提供continuous batching和prefix caching；(4) Tool system管理外部API（Wikipedia、Wolfram Alpha）或本地执行（Python interpreter）；(5) 流量按Poisson distribution模拟。论文配置：vLLM 0.6.6 + PyTorch 2.6 + CUDA 12.8 + Llama-3.1-8B/70B-Instruct + GCP A100 GPU。开源：https://github.com/VIA-Research/AgentBench。

AIMS进一步将AI Agent Serving System扩展到**cloud-edge hybrid**场景：云端部署LLM（GPT-5/Claude Sonnet 4），本地边缘设备部署SLM（Qwen3-4B/Gemma3-4B，llama.cpp执行），通过Adaptive Iteration-level Model Selector动态决定每个subtask在SLM还是LLM上执行。与传统agent serving中模型选择固定在部署时不同，AIMS在每次LLM invocation前做路由决策（request-level classifier + subtask-level routing pipeline: SSE→SLE→CD→SD），以最大化SLM使用同时保持LLM级精度。调度器decision overhead仅占总时间3-7%，estimator栈约2GB VRAM。AIMS在EUROSYS '26上被提出。

涉及论文标题：
- The Cost of Dynamic Reasoning: Demystifying AI Agents and Test-Time Scaling from an AI Infrastructure Perspective
- AIMS: A Cost-Efficient Framework for LLM-based Agent Deployment in Cloud-Edge Hybrid Environments
