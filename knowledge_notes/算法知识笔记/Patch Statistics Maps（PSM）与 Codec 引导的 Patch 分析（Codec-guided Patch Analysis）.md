## Patch Statistics Maps（PSM）与 Codec 引导的 Patch 分析（Codec-guided Patch Analysis）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PSM（Patch Statistics Maps）是把解码器产出的码流元数据网格（MV 幅值、像素域残差、频域高频/总能量）经平均池化聚合为每 patch 的统计量图（mv_mean / res_pixel_mean / hf_ratio），作为"该 patch 是否值得 SR 推理"的运行时信号。Codec 引导 = 用标准 bitstream 中解码器本就要解析的元数据做 SR 调度决策，无需服务器辅助、无需修改 bitstream，因此硬件视频解码器可被完整使用，调度运行时开销可忽略。SLICE 的 patch 分析三步：先识别复用 patch（MV=0 且像素残差=0），再按 score 的 TopK 选 SR patch（预算 k=35%），其余插值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Algorithm 2 全流程（P=16，全 GPU）：
```
# ① 生成 PSM（AvgPool2D 各一次完成聚合）
mv_mean      = AvgPool2D(G^mv,  kernel=P/4)        # MV 网格 4×4 块粒度 → 核/步长 4
res_pixel_mean = AvgPool2D(|G^pix|, kernel=P)      # 像素粒度 → 核/步长 16
hf_ratio     = AvgPool2D(G^hf, kernel=P/4) / AvgPool2D(G^t, kernel=P/4)  # 频域块粒度
# ② 识别复用 patch
R = (mv_mean==0) ∩ (res_pixel_mean==0);  M^reuse[R]=True
# ③ 打分并 TopK 选 SR patch
score = α·hf_ratio + β·(1 − clip(mv_mean/10, 0, 1))     # α=0.9, β=0.1
S = TopK(score, k=35%);  M^SR[S]=True                   # GPU TopK kernel
```
例子：270p 帧 30×17 个 patch 中，hf_ratio 高的纹理 patch 得分靠前入选 SR；hf_ratio≈0 的平坦 patch 被插值；复用 mask 先行剔除静态 patch，避免 SR 名额浪费。Fig.15 设计空间显示 35% 为吞吐/质量折中（40% 仅多 0.08dB）；Fig.18 显示默认权重最优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PSM 聚合用 PyTorch 的 AvgPool2D（像素粒度核/步长 P、4×4 块粒度核/步长 P/4），一次池化出全帧每 patch 均值；TopK 用 GPU 排序/选择 kernel。元数据网格来源：扩展版 Compressed Video Reader（补丁化 FFmpeg）在 H.264 解码时导出 G^mv/G^pix/G^hf/G^t，仿真硬件解码器未暴露的码流侧信号；部署时若解码器开放相关接口可直接读取。复用判定特意用像素域残差（而非变换系数）以排除 inter 帧中的 intra 块被误复用。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
