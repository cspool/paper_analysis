## Moran's I（空间自相关，Spatial Autocorrelation）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Moran's I 是度量空间数据自相关的统计量，量化空间数据的聚类/分散程度，取值 [-1,1]：-1 完全分散、0 空间随机、1 完全聚类。定义：
$$I = \frac{N}{W} \cdot \frac{\sum_{i=1}^{N}\sum_{j=1}^{N} w_{ij}(x_i-\bar{x})(x_j-\bar{x})}{\sum_{i=1}^{N}(x_i-\bar{x})^2}$$
其中 N 为元素数、w_ij 为 i/j 的空间权重（论文用共享边界与 k 近邻计算）、W=ΣΣw_ij、x̄ 为均值。本论文创新性地把 Moran's I 应用到 RNG 评估：把 RNG 输出组织成位图（bitmap），用 Moran's I 检测位图上数据的聚类/条带化——真随机位图的期望 Moran's I=0（无空间相关），非零值暴露输出中的空间结构（如 SRAM 阵列布局不对称导致的偏置条带）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
µRNG 应用流程（SRAM 熵源评估）：① 把 4KB SRAM 上电态组织为位图（每个 cell 一个像素：0/1/不稳定）；② 对每个环境 corner（温度 × 压摆率）计算 Moran's I；③ 比较聚类程度。关键实测：名义条件 Moran's I=0.032（基本空间随机，SRAM 可行熵源）；+85°C 慢电压爬坡时 Moran's I=0.127（明显条带化——布局不对称偏置在慢爬坡下暴露）；低温慢爬坡 Moran's I=0.015（数据保持竞争）。伪代码：
```
for corner in corners:                       # 温度 x 压摆率
    bitmap = sram_powerup_state(4KB, corner) # 上电态位图
    w = spatial_weights(bitmap)              # 共享边界 + k 近邻
    I = morans_I(bitmap, w)                  # 公式(5)
    report(corner, I, per_bit_entropy)
```
Annotations：x_i 为第 i 个 cell 的位值；w_ij 刻画 cell 邻接关系；慢压摆率下单元偏向布局决定态形成条带 → Moran's I 显著非零；低温下数据保持使熵崩溃（每 bit 熵 0.004 @-68°C）而条带不明显。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：计算邻接权重矩阵（网格图共享边 + k 近邻）、按公式算 I；成熟实现见于空间统计库（R spdep::moran、Julia SpatialDependence.jl、Python libpysal 等，可用于置换检验/蒙特卡洛 p 值）。本论文把它与碰撞测试、熵估计一起作为 NIST 之外的弱点发现测试。使用场景：检测 RNG/熵源输出中统计测试发现不了的"空间结构"（布局偏置、聚类）；尤其适合 SRAM/位图类熵源。结论：SRAM 熵源受慢压摆率结构印记与低温数据保持两个攻击者可控的不安全源影响，归为 Class 2。

涉及论文标题：
- μRNG: A Framework for Assessing Randomness in Intermittent Computing Devices
