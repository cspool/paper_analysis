## Knowledge Distillation for MoE Expert Merging (MoE 专家知识蒸馏合并)

术语解释
通过知识蒸馏将多个 MoE expert 的知识压缩到一个参数规模更小的模型中，使得推理时只需访问合并后的 expert，减少计算开销。BrownoutServe 的 united expert 训练是这一范式的典型应用。

术语是什么？
MoE expert 知识蒸馏合并的核心流程：
- **Teacher**: 一组原始 experts（k 个），各自拥有独立参数
- **Student**: 一个 united expert，参数规模与单个原始 expert 相同
- **Distillation target**: 最小化 student 与各 teacher 输出的 hidden states 之间的 MSE
- **结果**: Student 学会近似 k 个 teachers 的综合行为，推理时用 1 次 expert 访问替代 k 次

与标准知识蒸馏（Hinton et al. 2015）的区别：标准 KD 通常用于压缩一个大模型到小模型，而 MoE expert merging 是在模型内部横向合并多个同规模的 experts 到一个等价规模的 expert，保持推理路径的参数规模不变，但减少 expert 访问次数。

从算法pipeline角度拆解术语：
```
# 训练阶段（offline）
for each expert group in each transformer layer:
    # group = {Expert_a, Expert_b, Expert_c, Expert_d}  (k=4)
    UE = init_expert(same_param_size_as_one_expert)
    
    for x in distillation_dataset:
        # 收集所有 teacher 输出
        teacher_outputs = [Expert_i(x) for Expert_i in group]
        
        # Student 前向
        student_output = UE(x)
        
        # MSE 蒸馏损失
        loss = mean(||student_output - teacher_outputs[i]||^2 for i in range(k))
        loss.backward()
    
    save(UE)  # 保存 trained united expert

# 推理阶段（online）
# 原路径：token → expert_a(少量token) + expert_b(少量token) + ...
#          → 多次小batch kernel launch，GPU利用率低
# 蒸馏后：token → UE(合并后的token batch) → 1次大batch kernel，GPU利用率高
```

术语一般如何实现？如何使用？
- 训练数据：需准备蒸馏数据集，包含模型需要处理的典型输入分布
- 损失函数：MSE（BrownoutServe）、KL divergence（标准 KD）、或组合 loss
- 与相关方法的区别：MoDE（mutual distillation among experts）、KDEM（KD-enhanced expert merging）等进一步探索了 expert 间的相互蒸馏
- 限制：united expert 的参数容量固定，way=k 越大（合并越多 experts）精度损失越大

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs
