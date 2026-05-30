## Domain Representation Extraction

术语是什么？
Domain Representation Extraction（域表示提取）是 M3oE 框架的底层模块，负责将不同域的异构输入特征映射到统一的表示空间，同时保留域特定信息和跨域共性信息。其核心操作是：(1) 对每个域 d，将域特定权重矩阵 W_d 和共享权重矩阵 W_sh 做 element-wise product：Ŵ_d = W_d ⊙ W_sh，结合了域独特性和跨域共性；(2) 通过共享线性变换 W_c 将所有域的表示映射到同一 embedding 空间；(3) 额外引入 domain-agnostic mapping f_DA（一个多层神经网络）对原始输入做域无关映射并作为残差加入，以调节统一表示空间、抑制来自其他域的噪声。这一设计受 STAR（Sheng et al. 2021, CIKM）中 star topology 的启发——STAR 使用因子化的域共享和域特定网络来处理多域输入。

从算法pipeline角度拆解术语：
```
输入: x_d (域d的原始特征)

// Step 1: 域特定 + 共享权重融合
W_hat_d = W_d ⊙ W_sh                  // element-wise product

// Step 2: 域特定线性变换 + 共享偏置
z_d = W_hat_d @ x_d + b_d + b_sh

// Step 3: 统一空间映射
u_d = W_c @ z_d + b_c

// Step 4: 域无关残差连接
h_d = u_d + f_DA(x_d)                 // f_DA 为多层MLP

输出: h_d (统一表示空间中的域表示)
```
其中 W_sh、W_c 和 f_DA 在所有域上共享参数，学习跨域通用模式。

术语一般如何实现？如何使用？
Domain Representation Extraction 本质上是输入层的特征工程 + 域适配，通常作为模型的第一层。W_d ⊙ W_sh 的 element-wise product 设计在 PyTorch 中可直接用 `*` 运算符实现。f_DA 的域无关映射作为残差连接，其作用类似于正则化项——即使域特定信息有限，也能通过通用映射提供稳定的基准表示。这一层的输出 h_d 被后续的 Multi-View Expert Learning Layer 中各模块共享使用，因此其质量直接决定上层解耦和融合的效果。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---
