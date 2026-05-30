## Meta Lingua

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Meta Lingua是Meta (FAIR) 于2024年开源的轻量级LLM训练和推理库（https://github.com/facebookresearch/lingua, BSD-3-Clause）。定位介于NanoGPT（教学级）和Torchtitan（工业级）之间，专为快速研究和原型设计而构建。核心理念："验证idea X的可扩展性所需的最少功能集合是什么？"。纯PyTorch nn.Module实现，支持FSDP/HSDP/Data Parallel/Model Parallelism、torch.compile、activation checkpointing和FP8。内置MFU/HFU profiling、PyTorch .distcp checkpointing、FineWeb-Edu/DCLM数据加载和标准benchmark评测。架构生态：Transformer (Llama-style)、fastRNN (minGRU/minLSTM/Hawk)、Mamba SSM、Multi-Token Prediction等。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Attamba在Meta Lingua中的使用流程：
```
1. 环境搭建:
   git clone https://github.com/facebookresearch/lingua
   bash setup/create_env.sh
   conda activate lingua_<date>

2. 代码组织:
   apps/
   ├── attamba/              # Attamba项目目录
   │   ├── model.py          # Attamba模型定义（基于lingua.Transformer）
   │   └── config.yaml       # 模型/训练配置
   lingua/                   # 核心库
   ├── transformer.py        # Transformer/Llama基类
   ├── distributed.py        # parallelize_module() FSDP/TP包装
   ├── data.py               # DCLM dataloader
   └── profiler.py           # MFU/HFU profiling

3. 训练启动:
   torchrun --nproc-per-node 1 -m apps.attamba.train \
     config=apps/attamba/config.yaml \
     data.seq_len=1024 batch_size=16

4. Meta Lingua内部执行流程:
   a) 解析config.yaml → model config + training config
   b) 初始化Attamba模型（Mamba SSM blocks替代K/V投影）
   c) parallelize_module(model, FSDP/TP策略) → 分布式wrapper
   d) DCLM dataloader加载数据（10% subset, batch=16, seq_len=1024）
   e) Training loop: forward → loss → backward → optimizer step
   f) 自动profiling: MFU计算, memory tracking
   g) Checkpoint保存: PyTorch .distcp格式
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
安装：`git clone + conda env create`，支持单GPU调试到多GPU分布式训练。核心优势：(1) 代码极简——Transformer实现在单个文件中，易于修改如替换K/V投影为SSM block；(2) 内置profiling——自动计算MFU/HFU，无需手动配置；(3) 多架构支持——同一框架内可对比Transformer/Mamba/Hawk/minGRU等架构；(4) 模块化——可单独使用lingua核心库的组件。Attamba选择Meta Lingua的原因：需要快速迭代SSM+Attention混合架构的实验，尤其是修改attention内部组件（K/V投影替换）——这对代码灵活性要求极高，而Meta Lingua的单文件Transformer实现恰好满足此需求。

涉及论文标题：
- Attamba__Attending_To_Multi-Token_States
