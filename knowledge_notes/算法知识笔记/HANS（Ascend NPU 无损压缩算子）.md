## HANS（Ascend NPU 无损压缩算子）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
华为为 Ascend NPU 开发的无损压缩算法/算子（ENEC 论文称之为 HANS，闭源、仅提供 Python API 压缩张量；gitee.com/ascend/op-plugin 有相关 PR）。联网证据显示 CANN ops-math 仓库含 hans_encode 算子（aclnnHansEncode 接口）：对张量做指数字节的 PDF（概率分布）统计，按 PDF 分布做无损压缩，结果存 device 内存或卸载到 host 侧；输入支持 FLOAT16/BFLOAT16/FLOAT32（ND 格式，元素数需为 64 的倍数且 ≥32768），输出 PDF 分布 (1,256) INT32、尾数部分、定长压缩部分（fixed）与变长压缩部分（var）；支持 Atlas A2/A3 系列。ENEC 论文的评价：HANS 在压缩比与吞吐上都有限，且闭源（ENEC 是首个在 Ascend 上开源且达到 SOTA GPU 压缩器性能的无损压缩器）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# HANS encode（据 CANN ops-math 文档与论文描述）
pdf = PDF_stats(exponent_byte_of(tensor))     # 指数字节概率分布 (1,256)
fixed, var, mantissa = encode(tensor, pdf)    # 指数分定长+变长两段压缩，尾数单独输出
```
Annotations：HANS 与 ENEC 同为"按指数统计做无损压缩"路线，但 HANS 采用定长+变长混合结构（含变长部分 → 需要不规则访存/分支，可能限制其吞吐）；ENEC 全定长。ENEC 论文中的 NPU 基线对比（910B2）：ENEC 压缩吞吐为 HANS 的 1.36×（BF16）到 2.47×（FP32），解压 2.11×；压缩比两者接近（BF16 1.35-1.37 vs 1.33-1.35）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：CANN 算子（hans_encode/hans_decode），aclnn 接口调用，PyTorch 侧经 op-plugin 暴露 Python API（ENEC 论文用它做张量级压缩测试）。使用：Ascend 上模型权重无损压缩；由于只有 Python API，论文用 msprof 在 kernel 级测其性能。局限：闭源、不能修改/定制；定长+变长混合在 Ascend 上吞吐受限；ENEC 定位为它的开源替代（性能相当或更优）。

涉及论文标题：
- ENEC: A Lossless AI Model Compression Method Enabling Fast Inference on Ascend NPUs
