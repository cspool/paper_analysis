## SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SSMLoRA 提出基于状态空间模型（SSM）增强的低秩适应方法。核心是 Time Module，沿时间轴连接跨层插入的低秩矩阵，通过状态转移方程实现层间信息传递。Time Module 内部包含四个矩阵：W_a (d×r)、W_b (r×d)、W_c (r×r, 状态矩阵)、W_d (r×r, 控制矩阵)。前向过程：(1) 低秩投影 `x_new = x × W_a`；(2) 状态导数 `h_t' = h_{t-1} × W_c + x_new × W_d`；(3) Taylor 展开近似 `h_t = h_t' + h_{t-1}`；(4) min-max 归一化后将 `h_t_norm` 作为偏置：`y = x × W_0 + (x_new + h_t_norm) × W_b`。与 S4 不同，SSMLoRA 使用 Taylor 展开直接离散化而非离散化状态矩阵 W_c 和 W_d，避免了 FFT 开销（需时可开放）。稀疏插入策略：只在 attention 的 query 和 value 矩阵交替间隔插入 Time Module（层 l 激活 query，层 l+1 跳过，query 和 value 分别独立时间轴）。初始化策略：W_a 用 scaled Gaussian，W_b/W_c/W_d/h 为零初始化，使模型初始退化为稀疏 LoRA。
  - 实验比较：(1) GLUE benchmark（CoLA/SST-2/MRPC/STS-B/QQP/MNLI/QNLI/RTE）上 RoBERTa-base 对比 Fine-tune/BitFit/LoRA/QLoRA/MixLoRA（Table 1）；(2) LLaMA2-7B/13B 对比 LoRA（Table 2: RTE/BoolQ/WSC/WiC/MultiRC/COPA）；(3) 长文本能力——DeBERTaV3-base 在 SQuAD/NarrativeQA 上对比 Fine-tune/BitFit/LoRA（Table 3），RoBERTa-base 在 RACE 上对比 LoRA（Table 4）；(4) 不同长度区间 NarrativeQA 性能（Figure 2/Table 5）；(5) 稀疏性消融——RoBERTa-large 和 GPT-2 在 GLUE 上对比 LoRA 的不同 rank r=1/2/4/8/16（Table 6/7），SuperGLUE 对比（Table 8）；(6) 内存效率——LLaMA2-7B 上序列长度 16→7000 的 GPU 内存和推理耗时对比（Table 12）；(7) 参数效率对比（Table 9）；(8) 训练 wallclock time（Table 10/11）。

- 硬件平台是什么，配置是什么。
  - 小型模型（DeBERTaV3-base/RoBERTa-base/RoBERTa-large/GPT-2）：单卡 NVIDIA RTX 3090 (24GB)。
  - 大型模型（LLaMA2-7B/LLaMA2-13B）：单卡 NVIDIA RTX A6000 (48GB)。
  - 学习率范围：[5e-4, 1e-6]，采用动态学习率调度和 early stopping。LoRA 类方法统一超参：α=16, rank=8（消融实验除外），dropout=0.1。

- 模型是什么。数据集和bench分别是什么。
  - 模型：RoBERTa-base (124M)、RoBERTa-large (355M)、GPT-2、DeBERTaV3-base、LLaMA2-7B、LLaMA2-13B。
  - GLUE benchmark：CoLA (MCC)、SST-2 (Acc)、MRPC (F1/Acc)、STS-B (Pearson/Spearman Corr)、QQP (F1/Acc)、MNLI-m/mm (Acc)、QNLI (Acc)、RTE (Acc)。
  - SuperGLUE benchmark：RTE/BoolQ/WSC/WiC/MultiRC/COPA/CB/ReCoRD。
  - 长文本/推理：SQuAD (F1/EM)、NarrativeQA (ROUGE-L)、RACE (Acc)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/yuhkalhic/SSMLoRA（NAACL 2025 main conference）
  - 安装：Python 3.10, `pip install -r requirements.txt`。训练命令：`python src/main.py --dataset BoolQ`。
  - 算法 pipeline 张量计算流程（以 RoBERTa-base 为例，单层 attention 的 query 矩阵 W_Q 插入 Time Module）：
    ```
    # 初始化
    W_a: d×r = randn(d, r) * scale_gaussian  # scaled Gaussian
    W_b: r×d = zeros(r, d)  # 零初始化
    W_c: r×r = zeros(r, r)  # 状态矩阵，零初始化
    W_d: r×r = zeros(r, r)  # 控制矩阵，零初始化
    h:   1×r  = zeros(1, r)  # 状态向量，零初始化（时间轴起点的 Time Module）

    # Time Module 前向传播（每个被激活的 attention 层）
    x_new = x @ W_a                # [batch, seq, d] × [d, r] → [batch, seq, r]
    h_prime = h @ W_c + x_new @ W_d  # [1, r] × [r, r] + [batch, seq, r] × [r, r] = [batch, seq, r]
    h_new = h_prime + h            # Taylor 展开近似：[batch, seq, r]
    # min-max 归一化（per-batch）
    h_norm = (h_new - h_new.min()) / (h_new.max() - h_new.min() + 1e-8)
    y_lora = (x_new + h_norm) @ W_b  # [batch, seq, r] × [r, d] → [batch, seq, d]
    y = x @ W_Q + y_lora           # 最终输出

    # 状态传递（h 脱离计算图，仅 W_c/W_d/x 参与训练）
    h = h_new.detach()

    # 稀疏插入策略（query 和 value 独立时间轴，交替间隔）：
    # 对于 attention 中的 query 矩阵 W_Q：
    #   Layer l:   插入 Time Module（激活），接收来自激活层的 state
    #   Layer l+1: 跳过（不插入）
    #   Layer l+2: 插入 Time Module（激活）
    # ...
    # 对于 attention 中的 value 矩阵 W_V：独立的另一条时间轴，同样的交替间隔模式
    # 非 attention 层（如 FFN、classifier）：标准 LoRA 稠密插入（W_a + W_b，无 W_c/W_d/h）
    ```
    关键特性：(1) 参数仅约为 LoRA 的 50%（因交替间隔稀疏 + 仅 q/v 插入）；(2) FFT 可选——公式 (2) 计算为矩阵乘法而非卷积，但如果需要可启用 S4 的 FFT 加速；(3) 跨层状态传递使模型能关联不同层的低秩映射信息；(4) 零初始化使训练起点退化为稀疏 LoRA，渐进学习 SSM 连接。

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：RWKV（Receptance Weighted Key Value）模型架构，结合RNN和Transformer优势。核心设计包括：(1) Token Shift——每个block的时间混合输入由当前和前一个时间步的线性插值产生（`r_t = W_r · (μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1})`）；(2) WKV Operator——通道级时间衰减线性注意力，`wkv_t = Σe^{-(t-1-i)w+k_i}⊙v_i / Σe^{-(t-1-i)w+k_i}`，将Transformer的O(T²d)降为O(Td)；(3) Output Gating——使用σ(r_t)⊙wkv_t作为输出；(4) Channel-Mixing Block——使用squared ReLU激活：`o'_t = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)`。训练时使用time-parallel模式（类似Transformer并行），推理时使用time-sequential模式（类似RNN逐token递归）。
  - 实验比较：(1) FLOP匹配的零样本NLP评估——与Pythia/OPT/BLOOM在12个benchmark上比较；(2) 扩展上下文微调——从1024→2048→4096→8192逐步增加序列长度；(3) 长序列LRA benchmark与S4等模型比较；(4) Enwik8字符级语言建模；(5) 推理时延和内存与Transformer家族（BLOOM/OPT/GPT-Neo/Pythia）在CPU和GPU上比较；(6) 与ChatGPT/GPT-4在RTE/WNLI/GoEmotions等任务上的提示工程对比。

- 硬件平台是什么，配置是什么。
  - 训练：StabilityAI提供的GPU集群（论文未具体说明GPU型号和数量）
  - 推理benchmark：CPU (x86) 和 GPU (NVIDIA A100 80 GB)，使用float32精度

- 模型是什么。数据集和bench分别是什么。
  - 模型规模：RWKV 169M (12层/d=768)、430M (24层/d=1024)、1.5B (24层/d=2048)、3B (32层/d=2560)、7B (32层/d=4096)、14B (40层/d=5120)
  - 训练数据：The Pile (800GB, 330B tokens)，训练1个epoch
  - 评估benchmark：ARC (Easy/Challenge)、BoolQ、COPA、HeadQA、HellaSwag、LAMBADA、OpenBookQA、PIQA、ReCoRD、SciQ、Winogrande、LRA (ListOps/Text/Retrieval/Image/Pathfinder/Path-X)、Enwik8

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/BlinkDL/RWKV-LM
  - 预训练模型：https://huggingface.co/RWKV
  - Chat接口：https://github.com/BlinkDL/ChatRWKV
  - 算法pipeline伪代码（每个RWKV block的前向过程）：

```
# 输入: x_t (当前token), x_{t-1} (上一token), 状态 (a_{t-1}, b_{t-1}, p_{t-1})
# === Time-Mixing Block ===
# Token Shift
r_t = W_r @ (μ_r * x_t + (1-μ_r) * x_{t-1})       # [d]
k_t = W_k @ (μ_k * x_t + (1-μ_k) * x_{t-1})       # [d]
v_t = W_v @ (μ_v * x_t + (1-μ_v) * x_{t-1})       # [d]

# WKV Operator (channel-wise, 每个通道独立)
q = max(p_{t-1}, u + k_t)                           # 数值稳定
wkv_t = (e^{p_{t-1}-q} * a_{t-1} + e^{u+k_t-q} * v_t) /
        (e^{p_{t-1}-q} * b_{t-1} + e^{u+k_t-q})    # [d], element-wise

# 状态更新
q' = max(p_{t-1} - w, k_t)
a_t = e^{p_{t-1}-w-q'} * a_{t-1} + e^{k_t-q'} * v_t   # [d]
b_t = e^{p_{t-1}-w-q'} * b_{t-1} + e^{k_t-q'}        # [d]
p_t = q'

# Output Gating
o_t = W_o @ (σ(r_t) * wkv_t)                         # [d]

# === Channel-Mixing Block ===
r'_t = W'_r @ (μ'_r * x_t + (1-μ'_r) * x_{t-1})     # [d]
k'_t = W'_k @ (μ'_k * x_t + (1-μ'_k) * x_{t-1})     # [d]
o'_t = σ(r'_t) * (W'_v @ max(k'_t, 0)²)              # [d], squared ReLU
```
  关键张量形状：输入x为[B,T,d]（训练）或[1,1,d]（推理逐token）；W_r/k/v/o为[d,d]；μ为[d]（可学习token shift参数）；w为[d]（可学习通道时间衰减，非负）；u为[d]（当前token bonus参数）。训练时WKV通过串行扫描（可用parallel scan优化至O(B log T d)），推理时递归更新仅需O(d)空间和O(d)时间。
