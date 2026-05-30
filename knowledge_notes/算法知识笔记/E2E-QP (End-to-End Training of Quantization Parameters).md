## E2E-QP (End-to-End Training of Quantization Parameters)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
E2E-QP (End-to-End Training of Quantization Parameters) 是 EfficientQAT 两阶段框架的第二阶段，冻结Block-AP产出的量化权重 W_q 和零点 z，仅端到端训练步长 s。其设计动机：Block-AP逐block独立训练忽略了跨block交互——各block输出的量化误差会级联放大，导致全局性能下降。E2E-QP通过端到端训练步长s来补偿这种跨block误差传播。由于步长s参数量极小（约占1.6% at g=64），E2E-QP的内存/时间开销远低于传统端到端QAT。例如Llama-2-70B W2G64的E2E-QP仅需34.2GB显存和14.3h训练时间（单A100-80GB）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# E2E-QP 端到端仅训练步长s
def e2e_qp(model, train_data, epochs=1, lr_s=2e-5):
    # 冻结所有权重
    for param in model.parameters():
        param.requires_grad = False
    # 仅解冻步长s（每个quantized group的scale factor）
    for step_size in model.step_sizes():
        step_size.requires_grad = True
    for batch in train_data:                     # 4096 samples, ctx=4096
        # 前向：仅反量化（无Eq.1量化过程）
        for linear in model.linears:
            linear.W_hat = (linear.W_q - linear.z) * linear.s
            # 梯度 ∂W_hat/∂s = W_q - z  （无需STE，直接解析梯度）
        output = model(input)                    # 端到端前向
        loss = cross_entropy(output, labels)     # LM损失
        loss.backward()                          # 仅s接收梯度
        optimizer_s.step()                       # lr_s: 2e-5(2bit), 1e-5(3bit)
    return model
```
关键设计：(1) E2E-QP中不执行量化操作(W_int=clamp(round(W/s)+z,...))，仅执行反量化(Ŵ=(W_q-z)*s)——因此W_q冻结不变；(2) s的梯度 ∂Ŵ/∂s = W_q - z 为解析梯度，无需STE近似；(3) Table 6显示训练s、z或s+z性能相近（s PPL=7.68, z PPL=7.69, s+z PPL=7.68），默认仅训练s以最小化额外位宽（z从N-bit展开为FP16会增加0.22 bits/param）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
E2E-QP 可适配多种下游场景：(1) 量化训练：使用RedPajama 4096样本，batch=32, epoch=1, lr_s=2e-5(2-bit)/1e-5(3-bit)；(2) 指令微调：替换训练数据为Alpaca，batch=16, 10000 steps, src_ctx=384, tgt_ctx=128；(3) 多模态微调：配合LLaVA-1.5 pipeline，冻结LLM + 预训练projector，再端到端微调LLM和projector（lr=2e-5(4-bit)/3e-5(2/3-bit)）。Table 8显示增大E2E-QP样本数可持续降低PPL（128→32764 samples: PPL 8.09→7.50），但Avg Acc在4096后不再显著提升。结合Block-AP + E2E-QP的完整EfficientQAT流程可使Llama-2-70B W2G64在单A100-80GB上41h内完成。

涉及论文标题：
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models

---
