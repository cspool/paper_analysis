## Patch 复用与 HR Cache（Patch Reuse & HR Cache）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
利用视频时域冗余：连续帧间静态背景等区域在运动矢量与残差均为零（预测完美、内容无变化），可直接把上一帧超分结果（HR cache）中对应位置的像素拷贝过来，完全跳过该 patch 的 SR 推理与插值。零 MV+零残差像素占比：Vimeo90K 30.1%、Kinetics-400 25.5%、K600 25.3%、K700 20.7%（Fig.6），表明真实视频中存在大量可复用区域。复用正确性由实验保证：Fig.8 显示任意复用比例下，把上一帧超分 patch 贴入当前帧造成的 ΔMSE 以 <0.2% 幅度集中在零附近。复用还提升质量：SR 增强的细节从早期帧传播到后续帧（SLICE 0.35dB vs SLICE-noreuse 0.78dB 质量损失）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 的复用判定 + 上采样阶段的使用：
```
# 判定（全 GPU）：R = (mv_mean==0) ∩ (res_pixel_mean==0) → M^reuse
# 使用：PATCHWISEUPSCALE 中 reuse patch 从 HR cache（上一帧 SR 结果常驻 GPU）按坐标直拷；
#       MERGEPATCHES 中把水平相邻复用 patch 合成连续带整段拷贝，减少拷贝开销
```
例子：某直播视频的固定演播室背景 patch 在连续几十帧中 MV 与残差均为 0 → 每帧都从 HR cache 直拷，SR 推理量随复用率进一步下降；I2 视频复用率 79.43% → SLICE 达 52.19 FPS。Fig.8(b) 显示加速比随复用率超线性增长：极端整帧可复用场景达 98.24×（运行时间约为全帧 SR 的 1.02%）。与 frame-skip 方案的区别：SLICE 每帧都分发 patch 级更新，而 frame-skip 整帧跳过会导致新增高频区域得不到 SR 更新（Hybrid baseline 因此质量更低）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
软件实现 = GPU 显存中的上一帧 HR 结果缓存 + 按 mask 的 banded 拷贝；缓存与合并全部留在 GPU 侧，避免 CPU-GPU 往返（即使 Jetson 统一内存架构下也仍存在逻辑内存拷贝）。复用判定顺序优先于 SR 选择（先找可复用 patch 缩小候选集），以最大化复用收益。GOP 敏感：GOP 越长 inter 序列累积运动估计误差、残差变大、复用率下降（Fig.17），但更长 GOP 也减少 intra 帧的全帧 SR 次数。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
