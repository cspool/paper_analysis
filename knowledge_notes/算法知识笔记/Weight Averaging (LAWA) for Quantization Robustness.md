## Weight Averaging (LAWA) for Quantization Robustness

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Averaging（权重平均）是一种沿训练轨迹聚合多个 checkpoint 权重以改善模型泛化能力和 PTQ 鲁棒性的技术。核心操作：在训练过程中每隔固定步数保存 checkpoint，使用滑动窗口聚合最近的 K 个 checkpoint 的权重（均匀平均），输出平均后的模型。LAWA（LAtest Weight Averaging, Kaddour 2022）是最直接的变体——维护长度为 5 的滚动 FIFO 窗口，每个新 checkpoint 入队时最旧的出队，实时输出均匀平均后的权重。与 Model Soup（Wortsman et al., 2022，对多个独立训练 fine-tune 的模型取平均）不同，Weight Averaging 沿单一训练轨迹操作，无需多次训练。该论文的关键新发现：虽然 LAWA 在全精度下通常不如 lr cooldown，但在 3-bit / 4-bit 量化后，LAWA 可以匹配甚至超越 cooldown 的性能——因为 averaging 促进收敛到更平坦的极小值（wider minima），使模型对量化引起的权重扰动更具鲁棒性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LAWA 的在线算法流程：
```
buffer = []  # FIFO queue, max length = window
for step, batch in enumerate(dataloader):
    loss = model(batch).backward()
    optimizer.step(); optimizer.zero_grad()
    if step % checkpoint_interval == 0:
        buffer.append(deepcopy(model.state_dict()))
        if len(buffer) > window:
            buffer.pop(0)
        avg_state = average_state_dicts(buffer)  # element-wise mean
        lawa_model.load_state_dict(avg_state)
        # 评估 LAWA 模型的量化性能
        W_q = gptq_quantize(lawa_model, bits=3)
        quant_loss = eval_validation(W_q)
```
对于无法自由保存 checkpoint 的开源模型（如 OLMo-1B），使用连续发布 checkpoint 的增量平均：accumulate_n = Σ_{i=last_n} Θ_i / n，评估不同窗口长度 n 的效果。论文 Fig.24 显示 n=5 效果最好，平均后的模型同时降低全精度和量化后验证损失。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Weight Averaging 的实现：(1) PyTorch 中直接遍历 `state_dict` 逐参数 `(w1 + w2 + ...) / N`；(2) 通过 `torch.save`/`torch.load` 管理 checkpoint 文件。对大规模模型需注意 checkpoint 加载的 I/O 和内存开销。配置建议：checkpoint 保存间隔应足够密集（论文用 500 优化步），窗口长度 5 是默认配置，均匀平均优于指数移动平均。论文还发现 Model Soup（跨不同数据混合的独立训练模型平均）的量化误差低于任何单个成分（Fig.2），表明跨独立运行的 weight averaging 也有利于 PTQ。

涉及论文标题：
- Training Dynamics Impact Post-Training Quantization Robustness
