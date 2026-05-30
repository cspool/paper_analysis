# <span id="page-3-3"></span>3.3 Problem with Existing Solutions

To address the aforementioned redundancy in MoE execution, several solutions[9, 12, 25] have been proposed. However, these works either fail to explore the the potential of structured sparsity in model parameters, which can be accelerated by sparse ALU, or waste the memory bandwidth for dual-side sparse matrix multiplication.

Among them, Megablocks[25] provides a block sparse representation and a customized kernel, while vLLM[9] proposes a kernel that combines the computation processes of all experts into a single kernel to address data flow redundancies in MoE layers. However, these solutions overlook the opportunity to leverage structured sparsity in model weights. Moreover, their highly customized designs make it challenging to incorporate the structured sparsity efficiently.

Meanwhile, other research has developed kernels optimized for structured sparsity, delivering notable performance improvements over SOTA kernels for dense matrices or unstructured sparse matrices. However, kernels like BBS[11] and nmSPARSE[37] fail to utilize SpTC for further acceleration. Solutions such as cuSPARSELt[6] and DFSS[15] leverage SpTC but impose a sparse ratio limit of 50%. VENOM[12] allows for a flexible sparse ratio while utilizing SpTC with a V:N:M format, specifically optimized for sparse-dense matrix multiplication scenarios. As depicted in Figure 6, when encountering a sparse column in model weights, it skips the multiplication with the corresponding row in inputs. This approach maintains an efficient memory access pattern with coalesced memory access, as illustrated in **①**.

<span id="page-4-0"></span>![](_page_4_Figure_2.jpeg)

Figure 7. Samoyeds Dual-side Sparse Data Format.

However, challenges arise when both weight and input matrices are sparse. In such situations, as shown in Figure 6, the skipped row and the sparse column in inputs break the data into smaller tiles, adversely reducing performance. For instance, the data may be loaded into shared memory in formats ②, ③, and ③. Formats ② and ③ involve loading either sparse column data or skipped row data unnecessarily, leading to severe I/O amplification at high sparse ratios. Moreover, the data in format ④ are not contiguous in memory, leading to uncoalesced memory access and reducing GPU memory I/O bandwidth.

