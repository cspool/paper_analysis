# <span id="page-10-0"></span>A.2. The Necessity of Automated Kernel-Generation

MxMoE mitigates the combinatorial explosion of mixed-precision configurations by automating kernel generation [4.3.](#page-5-1) To illustrate the effectiveness of this approach, we compare our strategy with two alternative solutions:

• Developing a universal kernel to handle all precision combinations: This approach would compromise kernel performance. We provide a breakdown demonstrating the limitations of this method relative to micro-kernel specialization in MxMoE. Specifically, the kernel for W4A4-per-channel could theoretically share the same software pipeline with W4A4-group128, but enforcing universality significantly degrades performance. As shown in Tab. [6,](#page-10-3) we test different kernels under the shape [8192, 8192, 8192], and the specialized kernel always outperform unified kernels. The reason is that unifying the two pipelines requires introducing runtime condition checks, which hinder loop unrolling in the MAC-loop. Moreover, to support group-size=128, the per-channel kernel's tile-size selection is constrained, making configurations such as tile k = 256 infeasible.

Table 6. Performance comparison of different W4A4 quantization kernels on GPU TOPS

<span id="page-10-3"></span>

| Kernel Type                    | W4A4 per-channel TOPS | W4A4 group128 TOPS |  |  |
|--------------------------------|-----------------------|--------------------|--|--|
| W4A4 per-channel (Specialized) | 1070.5303             | N/A                |  |  |
| W4A4 group128 (Specialized)    | N/A                   | 667.3349           |  |  |
| Unified Kernel                 | 929.1997              | 412.0268           |  |  |

• Developing separate kernels for each configuration: While handcrafted kernels could match performance, they require substantial engineering effort. If a given hardware platform supports five quantization candidates (e.g., w2a6, w4a16, w8a8, w4a4, w4a4 with group-size 128), implementing individual kernels for all possible configurations would require 5! = 120 kernels. In contrast, our micro-kernel specialization approach requires implementing only 5 configurable micro-kernels, which are automatically combined by the kernel generator to form optimized fused operators.

Table 7: W5A5 mixed-precision scheme allocated by MxMoE. Qwen1.5-MoE, layer 5.

<span id="page-11-0"></span>

| Expert | Gate  |         |         |       | Up      |         | Down  |         |         |  |
|--------|-------|---------|---------|-------|---------|---------|-------|---------|---------|--|
|        | w-act | w gsize | a gsize | w-act | w gsize | a gsize | w-act | w gsize | a gsize |  |
| 0      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 1      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 2      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 3      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 4      | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |  |
| 5      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 6      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 7      | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |  |
| 8      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 9      | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 10     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 11     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 12     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 13     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 14     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |  |
| 15     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 16     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 17     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 18     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 19     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 20     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 21     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 22     | 8-8   | -1      | -1      | 8-8   | -1      | -1      | 8-8   | -1      | -1      |  |
| 23     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 24     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 25     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 26     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 27     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 28     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 29     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 30     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 31     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 32     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 33     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 34     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 35     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 36     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 37     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 38     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 39     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 40     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 41     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 42     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 43     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 44     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |  |
| 45     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 46     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |
| 47     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |  |
| 48     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |  |

## Submission and Formatting Instructions for ICML 2025

Table 7: MxMoE W5A5 scheme (continued)

| Expert |       | Gate    |         |       | Up      |         | Down  |         |         |
|--------|-------|---------|---------|-------|---------|---------|-------|---------|---------|
|        | w-act | w gsize | a gsize | w-act | w gsize | a gsize | w-act | w gsize | a gsize |
| 49     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |
| 50     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |
| 51     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |
| 52     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |
| 53     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |
| 54     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |
| 55     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |
| 56     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |
| 57     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 4-4   | 128     | 128     |
| 58     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 4-4   | 128     | 128     |
| 59     | 4-4   | 128     | 128     | 4-4   | 128     | 128     | 8-8   | -1      | -1      |
| 60     | 4-4   | -1      | -1      | 4-4   | -1      | -1      | 8-8   | -1      | -1      |