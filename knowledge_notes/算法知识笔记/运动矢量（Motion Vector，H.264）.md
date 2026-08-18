## 运动矢量（Motion Vector，H.264）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
运动矢量是视频编码中记录"当前块相对参考帧的坐标位移"的元数据。H.264/AVC 中编码器把帧划分为 16×16 宏块（可再分割为 8×8、4×4 等更小分区），对每个分区在参考帧中做运动估计（motion estimation，搜索最佳匹配块），将坐标偏移编码为运动矢量。运动矢量只建模线性平移，无法表达旋转、缩放、遮挡/去遮挡与非刚性运动——这些由残差捕获。解码端解析运动矢量并对参考帧做运动补偿（motion compensation）生成预测块。在 SLICE 中，运动矢量网格（4×4 块粒度，每元素一个 MV）被当作"该区域是否静态/运动剧烈程度"的代理信号：MV 均值=0 表示无运动（可跨帧复用）；MV 幅值大表示快速运动区域（帧内易模糊，SR 增益小）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 SLICE 的 Patch Analysis（Algorithm 2）中，运动矢量网格 G^mv 经平均池化聚合为每 patch 的统计量，参与两类决策：
```
mv_mean = AvgPool2D(G^mv, kernel=P/4, stride=P/4)   # G^mv 是 4×4 块粒度网格，P=16 → 核/步长 4
# ① 复用判定：mv_mean==0 且像素域残差均值==0 → M^reuse
# ② SR 打分：score = 0.9·hf_ratio + 0.1·(1 − clip(mv_mean/10, 0, 1))
#    运动项 clip(mv_mean/10)：MV 幅值大 → 该项接近 0 → 分低 → 不选做 SR
```
例子（270p 帧，P=16）：某静态背景 patch 内 4×4 块 MV 全为 0 → mv_mean=0 → 与残差共同判定复用；某快速运动物体 patch 的 mv_mean≈25 → 归一化 clip(25/10)=1 → 运动项 0 → 仅靠残差项得分，难以进入 TopK。论文实测 270p/540p 视频中 MV 幅值超过 10 的块仅占 3.6%/6.2%，故除以 10 是合理的归一化。Fig.7(c) 显示 patch 平均 MV 幅值越大，SR 相对插值的期望 PSNR 增益越小（时序错位与模糊所致），这是把 MV 作为负向信号的理论依据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
编码器侧由 x264/x265/FFmpeg 等实现运动估计（全搜索/菱形/HEPS 等算法），MV 以差分形式写入码流语法元素；解码侧硬件/软件解码器解析 MV 做运动补偿。SLICE 用扩展版 Compressed Video Reader（基于补丁化 FFmpeg，https://github.com/Yaojie-Shen/Compressed-Video-Reader）在 H.264 解码过程中导出每块 MV 网格，模拟 SoC 硬件解码器未暴露的码流侧信号；由于只读取不改写 bitstream，解码仍走标准硬件解码器。论文用 MV 与残差联合而不是单用 MV（MV-only baseline 质量更差，因为会优先选中静态背景而非高频区域）。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
