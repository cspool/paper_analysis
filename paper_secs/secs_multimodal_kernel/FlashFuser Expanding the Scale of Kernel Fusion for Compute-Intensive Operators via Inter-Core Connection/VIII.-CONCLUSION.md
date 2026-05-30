# VIII. CONCLUSION

In this paper, we presented FlashFuser, the first compiler framework that overcomes this limitation by leveraging the inter-core connection capabilities of modern GPUs. By introducing a DSM communication abstraction, using a dataflow analyzer to evaluate data placement and costs, and leveraging an efficient search engine to explore the vast search space, FlashFuser systematically generates highly efficient fused kernels. On an NVIDIA H100 GPU, our evaluation shows that FlashFuser delivers kernel speedups of up to 3.3× against highly-tuned libraries and 4.1× against state-of-the-art compilers. These gains, driven by a 58% reduction in memory access, lead to a 1.24× end-to-end speedup.

## ACKNOWLEDGMENT

We thank Dr. Size Zheng for providing the source code of Chimera. This work was supported by the National Key R&D Program of China under Grant 2022YFB4501400, and the National Natural Science Foundation of China (NSFC) Grants (62222210 and 62532006) and Shanghai Qi Zhi Institute Innovation Program SQZ202316. Any opinions, findings, and conclusions in this paper are those of the authors only and do not necessarily reflect the views of our sponsors.

