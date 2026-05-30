## MoE-Infinity Activation-Aware Expert Offloading for Efficient MoE Serving

- 属于Serving调度的实现是什么？实验比较什么？
  MoE-Infinity 提出了 **Sparsity-Aware Expert Cache**（稀疏感知专家缓存），核心由 Expert Activation Matrix Collection (EAMC) 驱动的激活预测、基于预测的专家预取（prefetching）和缓存淘汰（eviction）三部分组成。在 batch size=1 的个人机器场景下，利用 MoE 模型解码阶段专家激活的高度稀疏性和请求内重用偏斜（skewed reuse），将频繁使用的专家缓存在 GPU 有限显存中，减少 PCIe 上的按需取专家 I/O。实验比较了 MoE-Infinity 与 DeepSpeed-Inference、vLLM、Ollama (Llama.cpp)、Mixtral-Offloading、BrainStorm 在多种 MoE 模型（DeepSeek-V2-Lite、Mixtral-8x7B、Switch-128x0.2B、NLLB-128x0.4B、Arctic-128x4B）和 290 个 LLM 任务（BIGBench/FLAN/MMLU）上的 TPOT（Time Per Output Token）延迟，取得了 3.1–16.7× 的延迟降低。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA RTX A5000 (24GB GPU 显存)，通过 PCIe 4.0 连接主机内存（带宽 32GB/s）。主机内存按模型规模配置：Switch 用 64GB、DeepSeek-V2-Lite 用 32GB、Mixtral 用 128GB、NLLB 用 256GB、Arctic 用 1TB。所有模型参数完整驻留在主机内存中，密集参数（attention 权重和 KV-cache）常驻 GPU 显存，专家参数按需/预取到 GPU。

- 开源Serving框架是什么。修改了什么。
  开源地址：https://github.com/EfficientMoE/MoE-Infinity。MoE-Infinity 基于 PyTorch 自建推理运行时，集成 FlashAttention 等 kernel 优化，支持 PyTorch 和 HuggingFace 格式的 checkpoint。核心修改是在标准 MoE 推理 pipeline 中插入了三层机制：
  1. **EAMC 追踪与匹配**：每次迭代记录 iteration-level EAM（iEAM，L×E 矩阵记录每层每个 expert 被路由的 token 数），累积为 request-level EAM（rEAM），并与 EAMC 中历史 rEAM 用余弦距离匹配找到最相似的激活模式。
  2. **PredictEAM 预测**：将匹配到的历史 rEAM 聚合、行归一化并施加 layer proximity decay（公式 1-(i-l)/L），生成 predicted EAM（pEAM），给出每个 expert 在未来层的激活概率。
  3. **缓存淘汰与预取**：淘汰时计算每个已缓存 expert 的 priority score = n_token / ((pEAM + ε) × (1 - layer_idx/L))，淘汰最低分 expert；预取时根据 pEAM 提前将下一层可能激活的 expert 从主机内存通过 DMA 传输到 GPU。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  已开源（GitHub: EfficientMoE/MoE-Infinity）。以下为一次推理请求的完整执行流程：
  ```
  ① 输入: 用户提交一个 prompt（如 "Explain quantum computing"），batch_size=1，部署模型为 DeepSeek-V2-Lite (64 experts/layer, 每 token 激活约 6 experts)。
  
  ② Prefill Phase:
     - Attention 计算（常驻 GPU）+ Router 计算所有 prompt token 的 expert 分配
     - 根据 Router 输出按需从 CPU 内存 fetch 激活的 experts 到 GPU expert buffer
     - MoE forward（GPU 计算）→ 累积 iEAM → 更新 rEAM
     - EAMC 匹配：将当前 rEAM 与 EAMC 中历史 rEAM 做余弦距离匹配，找到最相似的激活模式组
  
  ③ Decode Phase（逐 token 迭代）:
     每次迭代:
     a. GPU 执行 Attention(当前 token Q, KV-cache) → Router dispatch → 确定当前层激活 expert IDs
     b. Cache lookup: 检查激活 expert 是否已在 GPU cache 中
        - Hit → 直接使用
        - Miss → 触发 FetchOnDemand: CPU→GPU 通过 pinned memory + DMA 传输 expert 参数 (PCIe 4.0 32GB/s)
     c. 预测与预取: PredictEAM(iEAM, EAMC) → pEAM → 预取下一层高概率 expert 到 GPU
        （预取与当前层 MoE 计算重叠，隐藏 PCIe 延迟）
     d. 缓存淘汰: 若 cache 满，按 priority score 淘汰最小概率 expert
        priority = n_token / ((pEAM_prob + ε) × (1 - layer_idx/L))
     e. 更新 iEAM → 累积到 rEAM
     f. MoE forward → 输出 logits → 采样下一个 token
  
  ④ 输出: 生成的 token 序列返回给用户
  
  ⑤ Post-request: rEAM 写入 EAMC，若 EAMC 容量满则替换最相似的已有 rEAM（维持多样性）
  ```
  核心作用：在单张消费级 GPU 上运行远超其显存的 MoE 大模型（如 Arctic 900GB），通过激活感知的智能缓存将 GPU 空闲等待 PCIe 传输的时间从 baseline 的 254–2073ms 降至 51ms，使单 GPU 推理延迟接近多 GPU 全内存部署水平。
