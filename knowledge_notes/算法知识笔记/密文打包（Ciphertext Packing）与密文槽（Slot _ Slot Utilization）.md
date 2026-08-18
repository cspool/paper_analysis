## 密文打包（Ciphertext Packing）与密文槽（Slot / Slot Utilization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 密文打包是把多个明文标量（图像像素、通道值、batch 元素等）塞进单个 CKKS 密文的 N/2 个槽（slot）中，使一次同态运算（Add/PMult/Rot）同时作用于多个数据（SIMD），摊薄 HE 原语（NTT、旋转、keyswitch）的高额成本。槽利用率（slot utilization）= 有效数据槽数 / 总槽数，反映 SIMD 效率：利用率 50% 意味着密文数翻倍、内存与下游计算膨胀。
- 打包的核心矛盾（本论文核心动机）：卷积在密文域引入两类数据依赖，都靠旋转解决——(1) 空间依赖（intra-channel，同通道相邻像素）、(2) 通道依赖（cross-channel，多输入通道聚合）。不同打包布局决定：(a) 旋转次数（内旋转/外旋转数量）、(b) 槽利用率、以及 (c) 密文数量与内存占用。HE-CNN 中旋转贡献约 70% 端到端延迟（Fig.1(b)），因此打包布局是性能一级因素。
- 本论文指出现有打包方案的通用缺陷：静态启发式、逐层无关；跨层碎片化（1×1 缩减层后槽利用率掉到 12.5%-31.25%）；旋转优化不完整（只减内或只减外）。FEnc² 的 Conv-aware Encoding + AAC 正是对这三点的统一求解。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 本论文 Algorithm 1 的槽映射公式（Conv-aware Encoding 的核心，BS×C×H×W 输入按 S×S 块打包）：
```
输入 X ∈ R^{BS×C×H×W}，块大小 S，M=max(pad(H),pad(W))，m=M/S
对块内坐标 (u,v)，0≤u,v<S：
    X_{uv} ← 收集所有 m×m 块中的 (u,v) 元素 → R^{C×BS×m²}
    展平并映射：X_{ijk}^{(u,v)} → slot l
      其中 i = ⌈l/(BS·m²)⌉（通道）、j = l mod BS（batch）、
            k = ⌈l/BS⌉ mod m²（块内坐标）
加密得满装密文 ct_{uv}（共 S² 个密文）
```
- Annotations：S=1,BS=1 退化为 row-major/Orion 式布局（无外旋转、内旋转最大）；S=M 退化为 CryptoNets 像素式布局（无内旋转、外旋转最大）；中间 S 平衡两类旋转。打包密度决定后续层密文数：利用率 100% vs 25% 时同一层密文数差 4 倍（Fig.5 瓶颈 8→2→8 例子）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：客户端按框架返回的"索引-槽映射"做 CKKS 编码加密（本论文 FEnc² 只凭公开元数据 H,W,C,BS + 模型结构自动生成映射，无需运行期 profiling）；服务端按该布局执行同态电路。已有方案族：LoLa/CHET（row-major）、Gazelle/Fast-HEAR/Multiplexed/Orion/Hyena+（interleaving+BSGS 多通道）、HELayers（block tiling+多图打包）、Fhelipe（层后合并稀疏槽）、CryptoNets（pixel-wise）、FEnc²（统一 fragment 编码，包含前两者为特例）。用途：加密 CNN/Transformer 推理的输入编码与层间布局管理；衡量指标：旋转数、槽利用率、密文数、内存。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
