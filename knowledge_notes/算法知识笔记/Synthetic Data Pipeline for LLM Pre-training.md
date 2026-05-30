## Synthetic Data Pipeline for LLM Pre-training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Synthetic Data Pipeline 是 Hunyuan-Large 提出的四步合成数据生成流程，用于补充自然语料中欠缺的高质量训练数据（数学、代码、低资源语言、高教育价值领域）。流程包含四个步骤：

1. **Instruction Generation**：使用高质量种子数据源（网页、QA数据、代码仓库、书籍等）+ 多样化 instruction 生成 prompt → 生成覆盖多领域、多风格、多复杂度的 instructions
2. **Instruction Evolution**：通过三方面改进初始 instructions：(a) 增强清晰度和信息量，(b) 自指导增强低资源领域，(c) 提升难度层级
3. **Response Generation**：使用多个不同大小的专门模型为 evolved instructions 生成专家级 answers
4. **Response Filtering**：使用 critique model + self-consistency 检查（多答案一致性过滤）去除低质量或不一致的数据

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Hunyuan-Large 四步合成数据 pipeline
# 总产出：约 1.5T tokens 高质量合成数据

# Step 1: Instruction Generation
seeds = [web_pages, qa_data, code_repos, books, ...]
instructions = []
for seed in seeds:
    prompt = instruction_gen_template.format(seed=seed)
    instructions.append(llm.generate(prompt))  # 覆盖多领域+多风格

# Step 2: Instruction Evolution
evolved = []
for inst in instructions:
    inst = enhance_clarity(inst)               # (a) 清晰度提升
    if is_low_resource(inst.domain):
        inst = self_instruct_augment(inst)     # (b) 自指导增强
    inst = increase_difficulty(inst)           # (c) 难度提升
    evolved.append(inst)

# Step 3: Response Generation
synthetic_pairs = []
for inst in evolved:
    expert_model = select_expert_model(inst.domain, inst.complexity)
    response = expert_model.generate(inst)
    synthetic_pairs.append((inst, response))

# Step 4: Response Filtering
filtered_pairs = []
for inst, resp in synthetic_pairs:
    score = critique_model.score(inst, resp)    # 多维质量评分
    if score < threshold:
        continue
    if is_objective_qa(inst):
        responses = [model.generate(inst) for model in ensemble]
        if not self_consistency_check(responses):
            continue
    filtered_pairs.append((inst, resp))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 需要多个专门模型作为 generator 和 critique model（Hunyuan-Large 使用 70B dense critic）；(2) instruction 多样性依赖种子数据的覆盖面和 prompt 设计的多样性；(3) self-consistency 筛选对客观 QA 任务有效（生成多个答案投票），主观任务需要人审。Hunyuan-Large 的 1.5T 合成数据覆盖数学、代码、低资源语言和高教育价值领域。实际生成时需配合分类标签系统灵活调整各类数据比例。合成数据 pipeline 已在 LLaMA 3/3.1、Phi-3 等主流工作中广泛验证，是提升模型能力的关键手段。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
