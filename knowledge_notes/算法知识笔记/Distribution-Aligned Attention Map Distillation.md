## Distribution-Aligned Attention Map Distillation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution-Aligned Attention Map Distillation（分布对齐注意力图蒸馏）是 Squat 论文针对量化自注意力模块注意力图结构退化提出的蒸馏技术。论文发现（Figure 3）：FP16注意力图中存在明显的初始token列特征（distinct column pattern），但量化后该特征消失。为此引入分布损失 `L_D = log(Σ_l Σ_h (attn_q · attn_f) / (||attn_q||₂ · ||attn_f||₂))`，通过对数缩放匹配原始损失尺度。该术语可推广为通过分布层面对齐（注意力图、特征图等）来指导量化的蒸馏技术。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def distribution_loss(q_model, fp_teacher, input_ids):
    with torch.no_grad():
        fp_attns = fp_teacher(input_ids, output_attentions=True).attentions
    q_attns = q_model(input_ids, output_attentions=True).attentions
    total_cos = sum(
        (q_attns[l][:, h] * fp_attns[l][:, h]).sum() /
        (q_attns[l][:, h].norm() * fp_attns[l][:, h].norm())
        for l in range(L) for h in range(H)
    )
    return log(total_cos)  # L_D
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
需要FP16教师模型提供参考注意力图。Squat中r_D=1.0（权重大于L_E）。推理时无此开销（仅训练阶段使用）。可与熵损失L_E联合使用获最佳效果。消融显示L_D单独使用比L_E更有效地恢复精度。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge
