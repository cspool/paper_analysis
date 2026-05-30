## SSM-augmented Low-Rank Adaptation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SSM-augmented Low-Rank Adaptation 是一类将 State Space Model (SSM) 的状态转移机制集成到 LoRA 低秩适配中的方法，SSMLoRA 是其代表性实现。核心创新：在标准 LoRA 仅关注"当前层输入→低秩空间→输出"的逐层独立适配基础上，引入沿时间轴跨层传递的状态向量 h_t，使第 l 层的低秩映射能够利用来自 l-1 层的上下文信息。关键技术组件：(1) Time Module 包含标准 LoRA 矩阵 W_a/W_b + SSM 状态矩阵 W_c/W_d + 持久化状态向量 h；(2) 状态更新采用 Taylor 展开离散化 `h_t = h_{t-1}·W_c + x_new·W_d + h_{t-1}`；(3) 归一化后的 h_t 作为低秩空间的偏置调整输出 `y = xW_0 + (x_new + h_t_norm)·W_b`。设计目标：通过跨层状态共享提升参数利用率，使稀疏插入（~50% 参数）仍能维持甚至超越稠密 LoRA 性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# Time Module 完整 pipeline（SSMLoRA 核心计算单元）
class TimeModule:
    """
    d: model hidden dim, r: low-rank dim (typically r=8, r << d)
    State space 沿时间轴传递: 同类矩阵共享时间轴，异类矩阵独立时间轴
    """
    def __init__(self, d, r):
        self.W_a = nn.Linear(d, r, bias=False)   # LoRA 降维
        self.W_b = nn.Linear(r, d, bias=False)   # LoRA 升维（零初始化）
        self.W_c = nn.Linear(r, r, bias=False)   # SSM 状态矩阵（零初始化）
        self.W_d = nn.Linear(r, r, bias=False)   # SSM 控制矩阵（零初始化）
        self.h = torch.zeros(1, r)               # 状态向量（零初始化）

    def forward(self, x: [B,S,D]) -> [B,S,D]:
        x_new = self.W_a(x)                         # Step 1: [B,S,D]→[B,S,R]
        # 广播 h 匹配 batch+seq
        h_prev = self.h.expand(B, S, R)
        h_prime = h_prev @ self.W_c.weight.T + x_new @ self.W_d.weight.T  # Step 2: 状态导数
        h_new = h_prime + h_prev                     # Step 3: Taylor 展开
        self.h = h_new[:, -1:, :].detach()           # 取最后 token 状态传入下层
        # Step 4: min-max 归一化
        h_min, h_max = h_new.min(), h_new.max()
        h_norm = (h_new - h_min) / (h_max - h_min + 1e-8)
        output = (x_new + h_norm) @ self.W_b.weight.T  # Step 5: 偏置调整 + 升维
        return output

# 初始状态：W_b=W_c=W_d=h=0 → y = xW_0 + 0 → 等价原模型
# 随着训练进行：状态转移逐渐学习跨层关联
```
零初始化策略保证训练起点退化为稀疏 LoRA，避免早期引入噪声。随着训练进行，模型通过 W_c/W_d 逐渐学习跨层状态关联，实现从"稀疏独立适配"到"SSM 连接适配"的平滑过渡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/yuhkalhic/SSMLoRA (NAACL 2025, Python 3.10 + PyTorch)。训练命令：`python src/main.py --dataset BoolQ`。关键设计决策：(1) Q/V 独立时间轴——防止不同语义角色的状态互相干扰；(2) min-max 归一化——`(h - min)/(max - min + ε)` 稳定数值，将 h_t 限定在合理范围；(3) 无需 FFT 优化——r 仅为 8，r×r 矩阵乘法开销可忽略。实验验证 GLUE 上 ~50% 参数达稠密 LoRA 性能；NarrativeQA 1000+ tokens 序列 ROUGE-L 超 LoRA 2.1%；RACE high-difficulty Acc 67.37 > LoRA 65.64。

涉及论文标题：
- SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

---
