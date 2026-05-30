## Time Embedding Layers in Diffusion Models (扩散模型中的时间嵌入层)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
时间嵌入层是扩散模型中将离散时间步 t（整数）转换为连续向量表示的全连接层（通常 1-2 层）。其作用是将时序信息注入模型：时间嵌入向量通过投影层（projection layers）变换后，在不同深度与 latent image representation 合并（通常通过加法或 FiLM 式调制），使模型在每个去噪阶段执行与时间步相适应的操作。QuEST 识别出这一机制在量化中的特殊重要性（Property ❶）：时间嵌入精度直接影响模型在不同时间步的执行正确性——量化的时间嵌入精度下降会导致时间步与模型功能不匹配，进而引起去噪序列震荡（oscillation），使 FID 恶化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
时间嵌入的计算过程伪代码：
```
# 标准扩散模型时间嵌入
def time_embedding(t, w1, w2):
    # t: 整数时间步 (如 150/200)
    # Step 1: 正弦位置编码
    half_dim = embedding_dim // 2
    emb = exp(arange(half_dim) * -log(10000) / (half_dim-1))
    emb = t.unsqueeze(-1) * emb.unsqueeze(0)
    emb = cat([sin(emb), cos(emb)], dim=-1)   # [1, embedding_dim]

    # Step 2: 线性变换（1 或 2 层）
    emb = linear(emb, w1)                      # 第一层
    emb = silu(emb)                            # 激活函数
    emb = linear(emb, w2)                      # 第二层 → time_emb
    return emb                                  # [1, model_dim]

# 注入 UNet（在各 ResBlock/Attention Block）
def forward(x, t):
    t_emb = time_embedding(t, w1, w2)   # 获取时间嵌入
    for block in unet_blocks:
        # 通过投影层将 t_emb 映射到与 x 相同维度
        scale, shift = proj_out(t_emb).chunk(2)
        x = block(x)
        x = x * (1 + scale) + shift     # 时间条件调制
    return x
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
扩散模型时间嵌入的量化策略：(1) QuEST 的 TLA：同时微调时间嵌入层的全精度权重和其激活量化参数，使量化时间嵌入输出与全精度版本对齐；(2) PTQ baseline：直接量化时间嵌入层权重和激活（不微调），导致 FID 上升 0.81-1.04；(3) 预计算方法：预先计算所有时间步的时间嵌入并直接查表使用——这忽略了量化模型中各模块的兼容性变化（不同模块量化后对相同时间嵌入的响应不同），因此性能不如微调；(4) 其他方法：TFMQ-DM 校准所有时间步的时间嵌入层和投影层；TDQ 通过简单网络学习跨时间步的动态量化参数。时间嵌入层参数极少（<1% 总参数），因此微调成本极低但同时效果显著。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
