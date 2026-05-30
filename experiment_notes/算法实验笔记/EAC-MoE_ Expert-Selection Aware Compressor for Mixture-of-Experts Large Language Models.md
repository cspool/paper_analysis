## EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 EAC-MoE，结合两部分：(1) **QESC (Quantization with Expert-Selection Calibration)**：静态量化方法，对 MoE 模型进行逐层 GPTQ 量化后，使用 TopK-MSE 损失校准路由器，缓解低比特量化引起的 expert-shift 问题。MHSA 保持 4-bit，Router 保持全精度，Expert 量化到 2/2.5/3-bit（最终平均位宽 2.06/2.54/3.03-bit）。校准用 WikiText2 训练集的 128 条 2048 长度序列。(2) **PESF (Pruning based on Expert-Selection Frequency)**：动态专家剪枝，在推理时根据当前序列中各 expert 被选中的频率，剪枝低于阈值 α × (l×K/N) 的 expert。α=0.3 保守（~10% 加速无精度损失），α=0.7 激进（~30%+ 加速）。PESF 仅适用于 prefill 阶段。(3) **EAC-MoE = QESC + PESF** 组合，在量化基础上进一步剪枝。
  实验比较：(a) 量化对比：vs GPTQ、BSP、PMQ 在三种位宽下的 PPL 和 8 个 zero-shot 任务准确率；(b) 剪枝对比：vs EES、ODP 在准确率和加速比上；(c) 量化+剪枝组合 vs MC-MoE；(d) 消融实验：TopK-MSE vs MSE 损失；(e) 挑战任务：GSM8K 和 HumanEval；(f) 过拟合分析：混合精度方法用不同校准集在跨任务数据上的性能。

- 硬件平台是什么，配置是什么。
  量化过程在单张 A100 40G GPU 上执行。推理加速测试和部署演示在 RTX 3090 GPU 上进行。量化耗时：Mixtral-8x7B 总耗时约 1.32h（GPTQ 1.30h + Router Calibration 0.02h），其余模型类似。

- 模型是什么。数据集和bench分别是什么。
  模型：Mixtral-8x7B（8 expert, top-2）、Phi3.5-moe（16 expert, top-2）、Deepseek-moe-16b-base（64 expert, top-6, 含 shared expert）、Qwen1.5-MoE-A2.7B（60+4 shared expert, top-4）。数据集/benc"hmarks：(1) WikiText2 测试集 PPL；(2) 8 个 zero-shot 任务（EleutherAI LM Harness）：Winogrande、PIQA、ARC-Easy、ARC-Challenge、BoolQ、MathQA、HellaSwag、MMLU；(3) GSM8K（数学）、HumanEval（代码，pass@10，Bigcode-Evaluation-Harness 框架）。专家选择分析额外使用 19 个数据集涵盖 QA/CR、Math、Code、French 四类。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自 CAS 自动化所，发表在 ACL 2025。**论文未提供开源代码仓库**（截至查询时未找到公开 GitHub 链接）。算法核心流程如下：
