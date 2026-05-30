## LMMs-Eval

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LMMs-Eval (Zhang et al., 2024) 是一个面向大型多模态模型（LMM）的统一评估框架，由南洋理工大学 S-Lab 维护。提供标准化的 evaluation pipeline：模型加载 → 数据集预处理 → 推理 → 指标计算，支持多种多模态 benchmark 和模型。核心价值在于为不同 LMM（特别是视频理解模型）提供可复现的公平对比基准。在 DIG 论文中，LMMs-Eval 被用作评估框架，集成 vLLM 后端加速推理，对 Qwen2.5-VL-7B/32B 和 Qwen3-VL-8B 在不同 benchmark（MLVU, LVB, VideoMME）和帧配置（8-768 frames）上的 accuracy 进行系统化评估。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
LMMs-Eval 的评估流程：
```
# 配置
model_config = {"name": "Qwen2.5-VL-7B", "backend": "vllm"}
dataset_config = {"name": "MLVU", "split": "dev", "n_frames": 32}

# Step 1: 模型加载 + 推理后端配置
model = load_model(model_config, backend="vllm")
# 可指定: transformers, vllm, lmdeploy 等

# Step 2: 数据集加载 + 预处理
dataset = load_dataset(dataset_config)
for sample in dataset:
    # 按 benchmark 规范构建 prompt
    prompt = build_prompt(sample, dataset.template)
    # 加载视频帧
    frames = frame_selection_strategy(sample.video, n_frames=32)
    # 推理
    response = model.generate(concat([frames, prompt]))
    # 提取答案 (A/B/C/D 从回答中解析)
    prediction = extract_answer(response)

# Step 3: 指标计算
accuracy = sum(pred == gt) / len(dataset)

# Step 4: 汇总报告
report = {dataset: accuracy for dataset in all_datasets}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源地址：https://github.com/EvolvingLMMs-Lab/lmms-eval。安装：`pip install lmms-eval`。使用：`python -m lmms_eval --model qwen2.5_vl --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct --tasks mlvu --batch_size 1`。支持功能：(1) 多 benchmark 统一接口（MLVU, LVB, VideoMME, EgoSchema, NExT-QA 等）；(2) 多推理后端（vLLM, HF Transformers, LMDeploy）；(3) 自定义帧采样策略（uniform, fps, 自定义）；(4) 自动化的分布式评估。在 DIG 中，LMMs-Eval 通过 vLLM 后端执行所有推理，包括 query identification 和 reward assignment 阶段的 LMM 调用。

涉及论文标题：
- Divide__then_Ground__Adapting_Frame_Selection_to_Query_Types_for_Long-Form_Video_Understanding
- HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models
