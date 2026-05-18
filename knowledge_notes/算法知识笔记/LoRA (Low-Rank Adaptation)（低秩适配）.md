## LoRA (Low-Rank Adaptation)（低秩适配）

术语是什么？通过联网搜索让回答具体和精准。

LoRA (Low-Rank Adaptation) 由Hu等人(2021)提出，通过在预训练权重旁添加低秩分解矩阵（W' = W + ΔW = W + BA，其中B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)）实现参数高效微调。核心思想：模型权重更新ΔW实际具有内在低秩属性，大模型的自适应变化有效自由度远低于全参数空间维度。训练时W₀冻结，仅更新A（Kaiming初始化）和B（zero初始化），ΔW = BA在训练起始为零。训练后可将ΔW与W₀合并（W' = W₀ + BA）使推理时无额外延迟。LoRA具有"plug-and-play"特性：LoRA模块可独立于base model存储和重用，支持灵活的任务间切换，切换开销极低（Llama3-1B上<1ms）。

在**LLM端云协同推理**（TailorLLM）中：LoRA使端侧SLM（Llama3-1B）通过加载task-specific低秩矩阵即可在特定高频任务上达到接近cloud LLM（Llama3-70B）的精度，减少云端调用。每个adapter约22MB（r=16），可选择性应用到特定网络模块（如Q/K attention matrices）。TailorLLM通过RFLoRA进一步压缩adapter到~11.56MB。

在**扩散模型**（Difflow）中：LoRA被广泛用于注入特定视觉风格、角色或概念到预训练扩散模型中——每个LoRA仅adapt attention layers的Q/K/V/O projection权重，一个LoRA通常仅几MB（r=4-32），与base model独立加载/卸载。Difflow评估的edit应用使用了16种不同的LoRA weights作为style variants。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**LLM LoRA adapter 切换与推理**（TailorLLM）：
```
// LoRA-augmented linear layer: y = W₀x + BAx
// W₀ ∈ R^{d×k} frozen pretrained weight
// A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d,k)

// 训练阶段（云端RTX 3090）:
1: 对每个下游任务task_i的训练数据:
2:     A ~ Kaiming初始化   // 通常跨任务可共享
3:     B = zeros(d, r)     // zero初始化
4:     for each training step:
5:         ΔW = B @ A
6:         W' = W₀ + ΔW   // 仅A, B有梯度
7:         loss = CrossEntropy(model(x), y)
8:         B.grad, A.grad ← backward(loss)
9:         更新B, A

// 端侧推理切换（Tesla T4，<1ms）:
10: 加载task_i对应的B_i矩阵 → 与预存A合并
11: for each token in autoregressive decode:
12:     hidden = W₀ @ x + B_i @ (A @ x)
```

**扩散模型LoRA**（Difflow）：

LoRA在扩散模型attention层的权重修改：

```
// Original attention linear layer: y = Wx, W ∈ R^{d×k}
// LoRA-augmented: y = Wx + BAx
// B ∈ R^{d×r}, A ∈ R^{r×k}, rank r << min(d,k)

// 在U-Net attention层中:
1: Q = W_Q @ latent + B_Q @ (A_Q @ latent)   // LoRA adapted Query
2: K = W_K @ latent + B_K @ (A_K @ latent)   // LoRA adapted Key
3: V = W_V @ latent + B_V @ (A_V @ latent)   // LoRA adapted Value
4: attention_output = softmax(Q @ K^T / sqrt(d)) @ V
5: O = W_O @ attention + B_O @ (A_O @ attention)
```

在Difflow的edit应用中使用16 LoRA styles的pipeline流程：
```
1. 固定 control_image (Canny edge) + prompt → CLIP + ControlNet features
2. 固定 latent_noise (相同初始噪声保证结构一致)
3. for each LoRA_i in 16 styles:
4.     load LoRA_i weights (merge BA into attention projections)
5.     denoise latent with LoRA_i → styled output image_i
```

由于16个LoRA variants共享相同的latent_noise和conditioning inputs，Difflow将U-Net分解为input-dependent dGraphs（per-request unique）和input-independent dGraphs（16 styles共享），后者识别为loop-invariant并通过multi-value compile-time caching实现precomputation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LoRA权重在LLM领域通过HuggingFace PEFT库（`peft.LoraConfig`）配置和训练，支持指定target_modules（如q_proj, k_proj, v_proj, o_proj）、rank r和alpha scaling。训练后adapter以.bin/.safetensors保存B和A权重。推理时通过`model.load_adapter()`加载，切换开销极低（Llama3-1B<1ms），支持多adapter热切换。LoRA变体包括：DoRA（动态rank分配）、AdaLoRA（自适应layer-wise rank）、QLoRA（4-bit量化+LoRA）、HydraLoRA（非对称架构）、RFLoRA（冻结共享A+方向-幅度解耦）。开源实现：https://github.com/huggingface/peft。

在扩散模型领域，LoRA权重通常以.safetensors格式分发（如Civitai上的模型），通过Diffusers的`load_lora_weights()`加载。开源工具如kohya-ss/sd-scripts用于训练自定义LoRA。LoRA的应用产生大量具有相同base inputs但不同fine-tuning weights的correlative requests，Difflow利用invariant tensor elimination将input-independent计算在compile-time precompute（16 multi-value cached outputs），显著减少运行时计算。

涉及论文标题：
- TailorLLM: Collaborative End-Cloud Inference of Large and Small Language Models Based on Low-Rank Adaptation
- Difflow: A Data-Characteristic-Aware Serving System for Diffusion Models

---

