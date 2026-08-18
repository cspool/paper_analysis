## Expert Substitution（专家替换）与 Expert-Cache Router（专家缓存路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Substitution 是 SMoE（NJU/Tsinghua/Honor，ISCA'26）提出的第三种 MoE 专家调度范式（前两种为预取类与剪枝类）：利用 router gate score 反映的专家重要性差异，把被激活但分数低（low-score）的专家替换为 GPU 显存中已缓存、gate score 与之相近的未激活专家，从而在不损失精度（α 阈值内）的前提下减少 CPU→GPU 的 PCIe 专家加载与 CPU 专家计算。核心观察：fine-grained MoE 中只有少数 top-score 专家显著影响输出，low-score 专家分数与未激活专家相当（routing noise + load balancing 使 tail expert 行为趋同），替换它们几乎不损失精度、甚至因抑制 noisy activation 而提升（论文表 VI 手动降分实验验证）。配套算法是 Expert-Cache Router（Algorithm 1）：按超参 α（substitution threshold）与第 k+1 高分 S_{k+1} 分档——score > (1+α)S_{k+1} 为 top-score 专家保留；(1−α)S_{k+1} ≤ score < S_{k+1} 且已在 GPU 或属 top-score 集的为可替换候选 E_s；S_{k+1} ≤ score < (1+α)S_{k+1} 为 low-score 专家 E_l——用 E_s 中最高分者替换 E_l 中不在 GPU 者，不足部分回退为 PCIe 加载/CPU 计算。优化目标为逐层 max |G∩E_a| + min(|E_l\G|, |E_s|)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Expert-Cache Router（Algorithm 1，单 token t，k=top-k，α 阈值）
S_t = gate_scores(t) sorted desc;  S_{k+1} = (k+1)-th score
T = (1+α)S_{k+1};  L = S_{k+1};  R = (1−α)S_{k+1}
for e in experts:
    if score(e) > T:        O[t] += e; C += e              # top-score，保留
for e in experts:
    if L ≤ score(e) < T:    B_t += e                        # low-score（E_l）
    elif R ≤ score(e) < L and (e in GPU or C): A_t += e     # 可替换候选（E_s）
if |A_t| ≥ |B_t|:  O[t] += top |B_t| of A_t                 # 全量替换
else:              O[t] += A_t; 剩余 |B_t|−|A_t| 个 low-score 走 PCIe/CPU
# 配合：score-aware eviction 保留高分专家扩大 E_s；top-score prefetch 保证 top 专家在 GPU
```
示例：Qwen2-57B-A14B（k 较大）中某层激活 5 个专家，原 GPU 命中 2/5；替换 low-score 专家 d,e 为 GPU 驻留的 f,g 后命中 4/5，预取量从 3 降到 1，GPU 命中率提升到 71%（S3 设置）。α 选取：min_α A(α) s.t. T(α) ≤ R（TPOT 预算），多项式拟合 T(α) 后一维搜索；S1/S2/S3 分别取 0.35/0.3/0.25。注意与 DIAMoND 的 Adaptive Expert Selection（冲突感知动态替换，用于边缘 MoE 推理的另一机制）名称相近但机制不同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SMoE 已开源：https://github.com/goingshr/SMoE（figshare: https://doi.org/10.6084/m9.figshare.31982136）。Python 3.13 free-threading（no-GIL）环境，Ubuntu dependency.sh 装 Rust 工具链并源码编译 tokenizers；运行入口 run.sh（环境变量传参）/ main.py，config JSON 字段：replaceScoreRatio（等价论文 --alpha，替换比例）、window_size（null=LRU）、if_prefetch、if_usecpu、if_replace。模型：deepseek-moe-16b / Qwen2-57B-A14B-Instruct / XVERSE-MoE-A4.2B；GPU：3080Ti 12GB / 4060Ti 16GB / A6000 48GB（PCIe 3.0/4.0）。效果：TPOT 相对最优 baseline 降 24%（batch=1）/35%（batch=3），S3 达 48%；GPU 命中率 >60%；α≤0.35 精度无损。

涉及论文标题：
- SMoE: An Algorithm-System Co-Design for Pushing MoE to the Edge via Expert Substitution
