## VLMEvalKit

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VLMEvalKit (Duan et al., ACM Multimedia 2024) 是一个开源的大规模多模态模型（LMM）统一评估工具包，由上海人工智能实验室（OpenGVLab）维护。提供标准化的 evaluation pipeline：模型加载 → 数据集预处理 → 推理 → 指标计算，支持 70+ 多模态 benchmark（包括 MMBench, MME, SEEDBench, MMMU, MMStar 等）和 100+ 模型（包括 GPT-4o, Gemini, Qwen-VL, InternVL, LLaVA 等）。核心价值：(1) 统一评估协议确保不同模型公平对比；(2) 支持多种推理后端（HuggingFace Transformers, vLLM, LMDeploy 等）；(3) 自动化结果收集和报告。VLMEvalKit 在 ViEBench 中用于评估 End-to-end VLMs（无 tool-use 能力的模型如 GPT-4o, o3, InternVL3, LLaVA-OV 等），确保与 agentic models 的 accuracy 对比公平。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
VLMEvalKit 的评估流程：
```
# 配置
model_config = {model_name: "Qwen2.5-VL-7B", backend: "transformers"}
dataset_config = {dataset: "MMBench", split: "test"}

# Step 1: 模型加载
model = load_model(model_config)

# Step 2: 数据集加载 + 预处理
dataset = load_dataset(dataset_config)
for sample in dataset:
    # 构建 prompt（按各 benchmark 规范）
    prompt = build_prompt(sample, dataset.template)
    # 推理
    response = model.generate(concat([image, prompt]))
    # 结果提取
    prediction = extract_answer(response, dataset.extractor)

# Step 3: 指标计算
metric = dataset.metric(predictions, ground_truths)

# Step 4: 结果汇总
report = aggregate_results(all_datasets, all_models)
```
在 ViEBench 中，End-to-end VLMs 通过 VLMEvalKit 评估仅输出 Accuracy（不支持过程级指标），agentic models 使用各自官方仓库的评估 pipeline 输出完整七项指标。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源地址：https://github.com/open-compass/VLMEvalKit。支持功能：(1) 多模态 benchmark 统一接口；(2) 推理加速后端（vLLM, LMDeploy, SGLang）；(3) 自定义数据集和模型注册；(4) 自动化 leaderboard 生成。典型使用：`python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model Qwen2.5-VL-7B-Instruct`。局限性：仅支持 outcome-oriented metrics（accuracy, F1 等），缺乏过程级评估能力——这恰好是 ViEBench 的贡献所在。

涉及论文标题：
- Beyond_Accuracy__Evaluating_Grounded_Visual_Evidence_in_Thinking_with_Images__ViEBench
- Test-Time_Temporal_Sampling_for_Efficient_MLLM_Video_Understanding
