## 哈希编码（Hash Encoding，多分辨率哈希编码 / Instant-NGP 式哈希网格）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hash encoding（Müller et al., "Instant neural graphics primitives with a multiresolution hash encoding", SIGGRAPH 2022）用多分辨率哈希表存场景特征：每个采样 3D 点按 L 层分辨率量化到网格顶点，用空间哈希函数查表得到各层特征向量并线性插值，拼接到 MLP 输入。相比 RFF 或 dense voxel grid，它用固定大小哈希表覆盖无限细节、训练快（秒级）内存小，是 Instant-NGP 类 grid-based pipeline 的核心。NeRArch-Sim 论文把 Encoding 阶段定义为"把采样位置转成特征向量"，hash encoding 与 RFF 是 MLP-/grid-based 管线的代表编码；其在 NeRArch-Sim 中作为分类学算子（HashEncoding，参数 num_levels、hash_table_size、feature_dim），并配套专用硬件（hash 地址生成、查找单元）。注意与本仓库中"HashEncode for LLM Attention（HATA）"、"Hash Encoding of Operator Fusion Schemes（STOF）"是不同上下文里的同名概念，勿混淆。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- NeRArch-Sim 用 NeuRex pipeline 作例子：`e = HashEncoding(dim, num_levels=16, graph=g)`，随后 MLP 输入维度 = e.out_dim。伪代码：
```
# 对采样点 x（3D）
feat = []
for l in 1..L:                       # 16 层分辨率
    v = quantize(x, res_l)           # 按层分辨率量化到整数网格
    hash_idx = hash(v) % table_size  # 空间哈希查表
    feat_l = trilinear_interp(table_l, v)   # 表内插值
    feat.append(feat_l)
h = concat(feat)                     # 拼成 MLP 输入特征
```
- 硬件/内存视角（NeRArch-Sim 表 VIII）：NeuRex 的 DRAM 流量由 position streaming（469MB）与 hash subtable loading（16MB，细分辨率层的不规则查找）主导；Grid Cache（64KB，DRAM Rd 9.8GB）与 Subgrid Buffer（128KB，DRAM Rd 16MB）服务哈希查找；NeuRex 的 Subgrid Buffer 因细分辨率层近似随机哈希查找产生最高 bank conflict 开销（表 IX：conflicts 7.7M、stalls 384K、overhead 2.25%）。算子级优化"restricted hashing（在 subgrid 边界内处理射线）"是 NeuRex 的 region-level reuse 优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件：tiny-cuda-nn/Instant-NGP 的 `MultiResHashEncoding`（NeRArch-Sim 环境依赖 tiny-cuda-nn）；Nerfstudio 的 instant-ngp 模型即此编码。硬件：NeRArch-Sim 的 Encoding 硬件库含 Address generator、Tree reducer、Index generation/computation unit 等（对应 NeuRex/CICERO/SRender 的哈希相关单元）；SRender 复用 Hash Index Generators、新增 Point Rearrangement/Distance Compute/Comparison units。哈希表大小、分辨率层数、特征维度均可配置（hash_table_size、num_levels、feature_dim）。

涉及论文标题：
- NeRArch-Sim: A Unified Simulator for Benchmarking and DSE of Neural Rendering Accelerators
