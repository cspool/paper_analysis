## Block-AP (Block-wise Training of All Parameters)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-AP (Block-wise Training of All Parameters) 是 EfficientQAT 提出的两阶段 QAT 框架的第一阶段，是在 block-wise reconstruction 框架下首次直接训练所有权重和量化参数的方案。与 BRECQ/OmniQuant/AutoRound 等方法不同——这些方法仅训练辅助量化参数（rounding参数、clipping阈值或步长）以限制优化空间防止过拟合——Block-AP 将标准 QAT 的"全训练"引入 block-wise 框架，同时训练原始权重 W、步长 s 和零点 z。Block-AP 的核心发现是：在 block-wise 重建中，全参数训练无需复杂的可训练参数设计（如 AdaRound 的学习取整），即可取得显著优于部分训练的结果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Block-AP 逐block训练流程
def block_ap(model, calib_data, epochs=2, lr_W=2e-5, lr_s=1e-4):
    for block_idx, block in enumerate(model.blocks):
        fp16_output = block(calib_data)           # 保存FP16输出（训练目标）
        # 初始化block内所有Linear层的s和z
        for linear in block.linears:
            init_scales_and_zeros(linear.W, bit=N, group_size=g)
        for ep in range(epochs):
            for batch in calib_data:               # 4096 samples
                # 前向：量化block内所有权重
                for linear in block.linears:
                    linear.W_int = clamp(round(linear.W/linear.s) + linear.z, 0, 2^N-1)
                    linear.W_hat = (linear.W_int - linear.z) * linear.s
                output = block(input, use_W_hat=True)
                loss = MSE(output, fp16_output)    # 重建损失
                # STE反向传播，更新W, s, z
                loss.backward()
                optim_W.step()  # lr_W: 2e-5(2bit), 1e-5(3/4bit)
                optim_sz.step() # lr_sz: 1e-4
    return model  # 输出量化模型 (W_q in N-bit, s in FP16, z in N-bit)
```
训练超参：batch_size=2, epochs=2, 校准数据4096样本（RedPajama, ctx=2048），训练后输出W_q(N-bit)、s(FP16)、z(N-bit)。Table 4消融：Block-AP单独使用即可将W2G64 Avg PPL从453.49降至8.53，Avg Acc从40.69%恢复至58.99%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-AP的实现依赖STE（采用LSQ+的三梯度公式Eq.3-5），通过自定义autograd Function将量化/反量化嵌入计算图。每block训练需以下内存：权重W(FP16) ≈ 202.4M参数、量化参数s+z ≈ 6.3M参数（Llama-2-7B单block）。Block-AP不使用Adam/AdamW等状态优化器——论文发现对于block-wise重建任务，SGD足以收敛且无需额外状态内存。与E2E-QP的关系：Block-AP提供高质量权重初始化，E2E-QP在此基础上做端到端微调；Table 4显示两者组合(Long) PPL=7.68 vs Block-AP单独=8.53 vs E2E-QP单独=9.33。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
