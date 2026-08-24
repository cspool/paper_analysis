# <span id="page-0-0"></span>EARL: Efficient Agentic Reinforcement Learning Systems for Large Language Models

Zheyue Tan Aalto University zheyue.tan@aalto.fi

Huining Yuan Tsinghua University yuanhuining0@gmail.com Mustapha Abdullahi Aalto University mustapha.abdullahi@aalto.fi

> Zelai Xu Tsinghua University zelai.eecs@gmail.com

Tuo Shi Aalto University tuo.shi@aalto.fi

Chao Yu Tsinghua University zoeyuchao@gmail.com

Boxun Li Infinigence-AI liboxun@infini-ai.com

#### **Abstract**

Reinforcement learning (RL) has become a pivotal component of large language model (LLM) post-training, and agentic RL extends this paradigm to operate as agents through multi-turn interaction and tool use. Scaling such systems exposes two practical bottlenecks: (1) context length grows rapidly during training, inflating memory usage and latency, and triggering out-of-memory (OOM) failures; and (2) intermediate tensors accumulate with context length, making cross-device data movement into a major system bottleneck.

We present *EARL*, a scalable system for efficient agentic RL. It introduces a *parallelism selector* that dynamically adapts model and training parallelism across RL stages based on sequence length and system load, and a *data dispatcher* that performs layout-aware, decentralized exchange of intermediate data batches. Together, these components increase throughput, reduce long-context failures, and enable stable large-scale training of agentic LLMs without relying on hard context length limits or length penalties.

