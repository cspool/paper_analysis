## MoH: Multi-Head Attention as Mixture-of-Head Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoH（Mixture-of-Head Attention）将 multi-head attention 中的每个 attention head 视为 MoE 框架中的 expert，通过一个 router 为每个 token 动态选择 Top-K 个 attention head 进行激活。核心组件：（1）**Heads as Experts**——将 h 个 attention head 视为 experts，router 对每个 token 产生 routing score g_i，仅 Top-K 个 routed head 被激活（g_i 非零），输出为加权求和 MoH(X, X') = Σ g_i · H^i · W_O^i；（2）**Shared Heads**——指定前 h_s 个 head 为共享 head（始终激活），捕获跨上下文的通用知识（如语法规则），剩余 h − h_s 个 head 为动态路由 head；（3）**Two-Stage Routing**——共享 head 的路由分数由 W_s 计算，路由 head 的路由分数由 W_r 计算并经 Top-K 筛选，再通过 W_h 产生 α_1, α_2 系数动态平衡两类 head 的权重；（4）**Load Balance Loss**——L_b = Σ P_i · f_i，防止路由坍塌到少数 head；（5）总 loss L = L_task + β · L_b（β=0.01）。
  - 实验比较：（a）**ViT 图像分类**（ImageNet-1K）：MoH-ViT-S/B 基于 TransNeXt 框架，仅替换 multi-head attention 为 MoH，与 DeiT、Swin、PVTv2、CoAtNet、FocalNet、CAFormer、TransNeXt 等对比 Top-1 Acc；（b）**DiT 图像生成**（ImageNet-1K 256×256）：MoH-DiT-S/B/XL 替换 DiT 中的 attention，对比 FID/sFID/IS/Precision/Recall；（c）**LLM 从头训练**：MoH-LLM-S(186M)/B(881M) 对比 vanilla LLM，6 个 benchmark（SciQ/PIQA/WinoGrande/OpenbookQA/LogiQA/TruthfulQA）；（d）**LLaMA3-8B Continue-Tuning**：MoH-LLaMA3-8B vs LLaMA3-8B，14 个 benchmark（MMLU/CEVAL/CMMLU/GSM8K/TruthfulQA/HellaSwag/LogiQA/BoolQ/LAMBADA/SciQ/PIQA/WinoGrande/NQ/ARC-C）；（e）**Ablation**：shared heads 消融、two-stage routing 消融、shared heads ratio 消融（13.9%~74.0%）、activated head ratio 消融（50%~80%）、inference time 对比（seq len 256/512，head num=32）。

- 硬件平台是什么，配置是什么。
  - ViT 训练：8 GPUs（论文未明确说明 GPU 型号，基于 TransNeXt 训练设置），自动混合精度（AMP）
  - DiT 训练：论文未明确说明 GPU 配置
  - LLM 从头训练：Megatron 框架，Tensor Parallel=1，Pipeline Parallel=1，batch size 4M tokens，序列长度 2048
  - LLaMA3-8B Continue-Tuning：Tensor Parallel=2（第一阶段）/ 1（第二阶段），Pipeline Parallel=1（第一阶段）/ 8（第二阶段），batch size 16M tokens，序列长度 8192

- 模型是什么。数据集和bench分别是什么。
  - 模型：MoH-ViT-S(50M)/B(90M) 基于 TransNeXt；MoH-DiT-S(33M)/B(130M)/L(458M)/XL(675M) 基于 DiT；MoH-LLM-S(186M, 12 layers, hidden=768, heads=12)/B(881M, 24 layers, hidden=1536, heads=16)；MoH-LLaMA3-8B（从 LLaMA3-8B continue-tune）
  - 数据集：ImageNet-1K（~1.2M images, 1000 classes）；LLM 训练用 RedPajama(Books 4.24%/Wikipedia 3.50%/ArXiv 4.37%/StackExchange 3.19%/C4 10.94%)、Dolma(61.28%)、Pile(12.48%) 按采样比例混合；LLaMA2 tokenizer（65,536 vocab）
  - Benchmark：ViT→ImageNet-1K Top-1 Acc；DiT→FID/sFID/IS/Precision/Recall；LLM→SciQ/PIQA/WinoGrande/OpenbookQA/LogiQA/TruthfulQA；LLaMA3→MMLU/CEVAL/CMMLU/GSM8K/TruthfulQA/HellaSwag/LogiQA/BoolQ/LAMBADA/SciQ/PIQA/WinoGrande/NQ/ARC-C

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/SkyworkAI/MoH（Apache 2.0），预训练权重在 HuggingFace（Chat-UniVi/MoH-ViT-*、Chat-UniVi/MoH-DiT-*、Chat-UniVi/MoH-LLaMA3-8B）
  - 算法伪代码流程：
    ```
    # 输入: X ∈ R^{T×d_in} (T tokens, d_in dims)
    # 超参: h (总head数), h_s (共享head数), K (激活路由head数)
    
    # 1. 计算共享head路由分数
    s_s = Softmax(W_s @ X)  # W_s ∈ R^{h_s×d_in}, s_s ∈ R^{h_s×T}
    
    # 2. 计算路由head路由分数
    s_r = Softmax(W_r @ X)  # W_r ∈ R^{(h-h_s)×d_in}, s_r ∈ R^{(h-h_s)×T}
    
    # 3. Top-K 选择路由head
    topk_indices = TopK(s_r, K)  # 对每个token选K个路由head
    
    # 4. 两阶段系数
    [α_1, α_2] = Softmax(W_h @ x_t)  # W_h ∈ R^{2×d_in}
    
    # 5. 组装 routing score g_i
    for i in 1..h_s:     g_i = α_1 * s_s[i]
    for i in h_s+1..h:   g_i = (i in topk_indices) ? α_2 * s_r[i-h_s] : 0
    
    # 6. 计算每个head的attention输出
    for i in 1..h:
      Q_i = X @ W_Q^i, K_i = X' @ W_K^i, V_i = X' @ W_V^i
      H^i = Softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
    
    # 7. MoH 加权求和输出
    MoH(X, X') = Σ_{i=1}^{h} g_i · H^i · W_O^i
    
    # 8. Load Balance Loss
    P_i = mean(Softmax(W_r @ X)[i-h_s])  对路由head
    f_i = mean(token选择head_i的indicator)
    L_b = Σ P_i * f_i
    
    # 总loss: L = L_task + 0.01 * L_b
    ```
    关键张量计算：对于每个 token x_t，router 计算 routing score 选择 Top-K head。shared head 始终参与计算，routed head 按需激活。输出是 activated head 的加权和。在 ViT 和 DiT 中，head 激活预算在各层不均匀分布——浅层激活较少 head，深层激活较多 head。
