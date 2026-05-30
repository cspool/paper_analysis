## LlamaFactory

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LlamaFactory（LLaMA-Factory）是由Zheng et al. (2024)提出的统一LLM微调框架，支持100+语言模型的高效微调。提供YAML配置驱动的训练pipeline，集成LoRA/QLoRA等参数高效微调方法、多种数据预处理策略和训练编排功能。在VisGym中，LlamaFactory被用作所有VLM supervised fine-tuning实验的训练框架，负责：(1) demonstration数据预处理（格式转换、tokenization）；(2) 训练编排（full-parameter fine-tuning、mixed-task training）；(3) 超参数管理（learning rate、batch size、training steps）。

从编译框架角度拆解术语，给出具体例子。
VisGym中LlamaFactory的训练pipeline配置：
```
# LlamaFactory training configuration for VisGym SFT:
model: Qwen2.5-VL-7B-Instruct
finetuning_type: full  # full-parameter fine-tuning
precision: bf16
global_batch_size: 64
learning_rate: 1e-5
max_steps: 1500  # single-task; 5000 for mixed-task
data:
  - demonstration trajectories from VisGym solvers
  - preprocessed: filtered failures, removed test-set overlaps
  - only easy difficulty demonstrations
output: fine-tuned VLM checkpoint
```

LlamaFactory的pipeline流程：
```
1. Data Preprocessing:
   Raw demonstration [{"instruction":..., "observations":[...], "actions":[...]}]
   → Convert to conversation format (user/assistant turns)
   → Tokenize + format for specific model (Qwen2.5-VL template)
2. Training:
   Model加载 → 应用fine-tuning config → DistributedDataParallel (if multi-GPU)
   → 标准LM loss on action tokens → Checkpoint保存
```

术语一般如何实现？如何使用？
开源地址：https://github.com/hiyouga/LLaMA-Factory，Apache 2.0 license。安装：`pip install llamafactory`。使用方式：(1) 通过YAML配置文件指定模型、数据、训练参数；(2) 通过CLI或Python API启动训练；(3) 支持Web UI（`llamafactory-cli webui`）进行可视化配置。VisGym使用LlamaFactory的命令行接口进行批量训练，支持single-task（每个任务独立模型）和mixed-task（单一模型联合训练所有任务）两种配置。

涉及论文标题：
- VisGym: Diverse, Customizable, Scalable Environments for Multimodal Agents

---
