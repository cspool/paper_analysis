## im2col（Image to Column，卷积到矩阵乘变换）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
im2col 是把卷积运算等价重写为矩阵乘法（GEMM）的数据重排变换：把每个卷积窗口（filter 覆盖的输入 patch）展平成一列（或一行），把输入特征图重排成"展开矩阵"，使所有卷积窗口的计算变成一次标准 GEMM。它牺牲存储（展开后数据冗余、内存膨胀，经典地约放大 k×k×C_in 倍）换取"GEMM 高度优化"的好处——GEMM 在 CPU/GPU/TPU 上有成熟的高性能实现与硬件支持。对卷积神经网络训练/推理的框架（Caffe、早期 cuDNN、TensorFlow）与稀疏加速器评估都常用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
im2col 的展开过程（单输入图 I，C_in 通道，H×W；卷积核 K 个 filter，每 filter c×k×k；输出 C_out=K 通道）：
```
输入: I[C_in][H][W], Filter[K][C_in][k][k], 步长 s, padding p
# 1. 构建展开输入矩阵 Col[C_in*k*k][H_out*W_out]
#    每输出位置 (oh, ow) 一列：按 (c_in, kh, kw) 顺序展平该窗口的像素
for oh in range(H_out):
    for ow in range(W_out):
        col = []
        for c_in in range(C_in):
            for kh in range(k):
                for kw in range(k):
                    col.append(I[c_in][oh*s+kh][ow*s+kw])
        Col[:, oh*W_out+ow] = col
# 2. 权重矩阵 FilterMat[K][C_in*k*k]：每个 filter 展平成一行
# 3. 一次 GEMM：Out[K][H_out*W_out] = FilterMat @ Col
# 4. 结果按 (oh, ow) 重排回输出特征图 Out[K][H_out][W_out]
```
HiT 的用法：把 ResNet50/VGG16 的三个卷积层（如 3×3、512 通道层）离线经 im2col 转换成矩阵乘法，得到 MS×MS workload（激活与权重均为 40% 密度非结构化稀疏），再用稀疏 GEMM 加速器执行。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：主流深度学习框架内置 im2col（PyTorch 的 unfold / Caffe 的 im2col 层；cuDNN 用隐式 GEMM 避免显式展开内存开销）；HiT 论文中 im2col 是离线数据预处理（把卷积层转成标准矩阵乘，遵循 [48]），作为 MS 评估 workload 的构造步骤——转换后输出仍保持卷积层语义，但可以复用矩阵乘加速器与稀疏数据流。注意：im2col 使非零布局从"卷积结构"变为"GEMM 矩阵结构"，稀疏加速器（HiT 等）据此按行列索引匹配执行，而无需感知卷积窗口拓扑。

涉及论文标题：
- HiT: A Unified Sparsity-Adaptive Architecture for High-Throughput Matrix Multiplication
