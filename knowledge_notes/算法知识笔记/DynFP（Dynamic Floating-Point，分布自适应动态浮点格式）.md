## DynFP（Dynamic Floating-Point，分布自适应动态浮点格式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- DynFP 是 UNICORE 提出的逐 group 可配置的低比特（4-bit/3-bit）浮点格式，定义 DynFP(W_E, W_M, Z, I)：W_E/W_M 为指数/尾数位宽（自适应 E/M 分配，如 E3M0、E2M1、E1M2，优先动态范围或精度），Z 为把冗余负零码重映射为 E3M2 正常域内有价值的值（更细分辨率中间值或更大 outlier 扩展动态范围；若 Z 超 group 格式最大值则符号吸收进 scale），I 为可选的 gap-insertion 空位插入标志（把指数码 E 在位置 ℓ 处拆成 E_hi/E_lo，映射 Φ(1,ℓ,E)=E_hi·2^(ℓ+1)+E_lo，在指数阶梯中插入受控空隙以匹配非均匀分布）。动机：LLM 张量级分布平滑，但 32 元素 group 级分布重尾、非对称、紧聚（图 11），单一静态 FP4 无法表示，静态格式导致精度损失；DynFP 让每 group 选自己的浮点布局。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
  - 数值语义（DynFP 解码伪代码）：
    ```
    if E==0 and M==0 and S==1:  v = Z                       # 负零重映射为有用值
    elif E==0 and M!=0:         v = (-1)^S * 2^(1-B) * M     # subnormal
    else:                       v = (-1)^S * 2^(Phi(I,l,E)-B) * (1+M)  # normal，I-flag 时指数经 Phi 插入空位
    ```
    例：E1M2 布局（2 尾数位）动态范围受限，I-flag 开启时指数码经 Φ 插入 0 bit 扩展覆盖范围；Z 把负零码映射为 0.5 以上 E3M2 正常域值（避免 reintroduce subnormal，Z≥0.5）。权重侧用离线贪心搜索（96 候选 → k=16 palette）选每 group 格式，K/V 侧用 crest factor κ 在线选格式；元数据 = 4-bit（权重）/1-bit（K/V）每 group 格式索引 + 8-bit scale，有效位宽 4.375 bits（比 MXFP4 高 2.9%）。效果：UNICORE-Q 在 4/4/16 各模型 PPL 最低（OPT-6.7B 10.93 vs INT 11.18），zero-shot 平均准确率多数配置最优。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
  - 实现：硬件侧 Unified Format Converter 用 LUT（(格式索引, 数值) 为索引）把 DynFP 编码映射到等效 E3M2 表示，负零经 mux 选 Z，1 cycle 解码；软件侧 artifact Software/Accuracy/（PyTorch，unicore_kernel/quant_utils/ae_scripts）实现格式搜索与量化，运行各表 shell 脚本自动下载模型/数据集。使用：对每个权重/K/V group 选择最优 E/M 布局 + Z + I-flag，使低比特浮点表示匹配 LLM 重尾/非均匀分布；离线权重量化无需激活校准（Llama-2-7B 单 RTX 6000 Ada 约 2 分钟），在线 K/V 量化用 κ 阈值映射（<0.2% QKᵀ FLOPs）。开源：https://github.com/CLab-HKUST-GZ/isca53-unicore。

涉及论文标题：
- UniCore: A Bit-Width Scalable GEMM Unit for Unified LLM Inference
