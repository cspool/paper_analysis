## GPU Patch 聚批与 TopK 选择（unfold 聚批 / AvgPool2D PSM / GPU TopK kernel）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLICE 把 patch 级上采样调度全部放在 GPU 上执行的运行时方案：① 用 AvgPool2D 一次池化把码流元数据网格聚合为每 patch 统计量（PSM）；② 用 GPU TopK kernel 按推理面积预算（35%）选出得分最高的 patch；③ 用 unfold 把 patch 网格转为紧凑 GPU 张量、按 SR mask gather 出需推理的 patch 组成 batch，一次/少数几次 EDSR forward；④ 按行分带（row-wise banded）GPU 拷贝合并写 framebuffer。设计目标：避免 CPU 往返与 kernel 启动开销，让 patch 提取、调度、SR、合并全链路驻留 GPU。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PSM 聚合（各一次 AvgPool2D，P=16）
mv_mean      = AvgPool2D(G^mv,  kernel=4,  stride=4)   # 4×4 块粒度 MV 网格
res_pixel_mean = AvgPool2D(|G^pix|, kernel=16, stride=16)  # 像素粒度
hf_ratio     = AvgPool2D(G^hf, kernel=4, stride=4) / AvgPool2D(G^t, kernel=4, stride=4)
# 调度决策
M^reuse = (mv_mean==0) & (res_pixel_mean==0)
score   = 0.9·hf_ratio + 0.1·(1 − clip(mv_mean/10,0,1))
M^SR    = TopK(score, k=35%)                            # GPU TopK kernel
# 聚批推理
patch_batch = unfold(frame, 16, 16)                     # (N_patch, 3, 16, 16)
selected    = patch_batch[M^SR]                         # gather → (≈35%·N_patch, 3, 16, 16)
HR_patches  = EDSR_fp16(selected)                       # 单次/少数几次 forward，输出 4× (…,64,64)
# 合并：reuse 行按水平相邻合成连续带整段拷贝；插值 patch 各自拷贝；SR patch 按目标位置写回
```
例子：270p 帧 30×17=510 个 patch，TopK 选约 178 个组成 batch 单次 EDSR forward；静态 patch 走 HR cache 带拷贝；其余 bicubic。论文强调不循环逐 patch 构建 batch，避免 CPU 往返与 kernel 启动开销，这是吞吐优化的关键实现细节。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 PyTorch 标准张量算子实现：torch.nn.functional.avg_pool2d、torch.topk、tensor.unfold / F.unfold、布尔 mask 索引（gather）、GPU 上的 banded 拷贝。论文未说明是否自定义了 CUDA kernel（记为论文未明确说明——实现描述为 PyTorch 算子级）。合并阶段把水平相邻的复用 patch 合成连续 band 整段拷贝，以提升带宽效率、降低每 patch 更新开销。

涉及论文标题：
- SLICE A Selective Local Inference Framework with Codec Exploitation for Accelerating Video Super-Resolution
