# **Abstract**

In recent times, dynamic-shape neural networks have gained widespread usage in intelligent applications to address complex tasks, introducing challenges in optimizing tensor programs due to their dynamic nature. As the operators' shapes are determined at runtime in dynamic scenarios, the compilation process becomes expensive, limiting the practicality of existing static-shape tensor compilers. To address the need for effective and efficient optimization of dynamic-shape neural networks, this paper introduces MikPoly, a novel dynamic-shape tensor compiler based on micro-kernel polymerization. MikPoly employs a two-stage optimization approach, dynamically combining multiple statically generated micro-kernels using a lightweight cost model based on the shape of a tensor operator known at runtime. We evaluate the effectiveness of MikPoly by employing popular dynamicshape operators and neural networks on two representative accelerators, namely GPU Tensor Cores and Ascend NPUs. Our experimental results demonstrate that MikPoly effectively optimizes dynamic-shape workloads, yielding an average performance improvement of 1.49× over state-of-the-art vendor libraries.

∗ Corresponding author.

![](_page_0_Picture_10.jpeg)

[This work is licensed under a Creative Commons Attribution-](https://creativecommons.org/licenses/by-nc-nd/4.0/)NonCommercial-NoDerivs International 4.0 License. *ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA* © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0385-0/24/04. https://doi.org/10.1145/3620665.3640390

*CCS Concepts:* • **Software and its engineering** <sup>→</sup> **Compilers**; • **Computing methodologies** → **Neural networks**.

*Keywords:* Tensor Compilers, Deep Learning

#### **ACM Reference Format:**

Feng Yu, Guangli Li, Jiacheng Zhao, Huimin Cui, Xiaobing Feng, and Jingling Xue. 2024. Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization . In *29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24), April 27-May 1, 2024, La Jolla, CA, USA.* ACM, New York, NY, USA, 16 pages. https://doi.org/10.1145/3620665.3640390

