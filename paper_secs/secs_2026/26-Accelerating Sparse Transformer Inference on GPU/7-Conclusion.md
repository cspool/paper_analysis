# 7 Conclusion

In this paper, we propose STOF, an efficient framework with flexible masking and operator fusion for optimizing sparse Transformer on GPU. First, we propose a unified MHA module that implements row-wise and block-wise kernels with unique storage formats and optimizations. Then, we propose an operator fusion module that enables fusion expansion and parameter tuning as well as mapping the fusion schemes to compilation templates. The experimental results show that STOF outperforms the state-of-the-art works in terms of MHA computation and end-to-end inference. For future work, we plan to extend STOF to support PaddlePaddle[3](#page-11-17) and to incorporate it transparently into the compiler stack.

