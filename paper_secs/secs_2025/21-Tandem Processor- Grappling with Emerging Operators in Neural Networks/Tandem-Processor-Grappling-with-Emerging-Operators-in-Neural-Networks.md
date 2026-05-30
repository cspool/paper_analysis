# Tandem Processor: Grappling with Emerging Operators in Neural Networks

Soroush Ghodrati Sean Kinzer Hanyang Xu Rohan Mahapatra Yoonsung Kim<sup>§</sup> Byung Hoon Ahn Dong Kai Wang<sup>‡</sup> Lavanya Karthikeyan Amir Yazdanbakhsh<sup>b</sup> Jongse Park<sup>§</sup> Nam Sung Kim<sup>‡</sup> Hadi Esmaeilzadeh

Alternative Computing Technologies (ACT) Lab

### **Abstract**

With the ever increasing prevalence of neural networks and the upheaval from the language models, it is time to rethink neural acceleration. Up to this point, the broader research community, including ourselves, has disproportionately focused on GEneral Matrix Multiplication (GEMM) operations. The supporting argument was that the large majority of the neural operations are GEMM. This argument guided the research in Neural Processing Units (NPUs) for the last decade. However, scant attention was paid to non-GEMM operations and they are rather overlooked. As deep learning evolved and progressed, these operations have grown in diversity and also large variety of structural patterns have emerged that interweave them with the GEMM operations. However, conventional NPU designs have taken rather simplistic approaches by supporting these operations through either a number of dedicated blocks or fall back to general-purpose processors.

This work sets out to challenge the conventional wisdom in neural accelerator design and explore the architecture of an on-chip companion, dubbed Tandem Processor, that complements the rather optimized GEMM unit in neural accelerators. This processor needs to be specialized to keep up with the GEMM unit; and yet needs to be programmable to address the (1) structural and (2) operational variations. To strike a balance between specialization and programmability, on the one hand, we specialize its memory access logic with a novel ISA/microarchitecture that alleviates the register file and its associated load/store operations. On the other hand,

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s).

ASPLOS '24, April 27-May 1, 2024, La Jolla, CA, USA © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-0385-0/24/04 https://doi.org/10.1145/3620665.3640365

the calculations of the non-GEMM layers are only supported through primitive arithmetic/logic vector operations. Therefore, programmability is offered at the mathematical level. The enhancements due to the specialization of the memory access logic in the Tandem Processor and its tight integration with the GEMM unit sustain the throughput and the utilization of the neural accelerator. Comprehensive evaluations of the proposed design based on the end-to-end execution of seven diverse DNNs including emerging language models show significant performance improvements and energy reduction enabled by leveraging the Tandem Processor. We provide the RTL code that is synthesizable both for FPGA and ASIC implementations in addition to the associated compiler as part of the open-source GeneSys project (https://actlab-genesys.github.io/). We also present the chip floorplan and post-layout analysis. This work is the result of 10 years of effort in building real NPUs that support end-to-end neural network execution.

CCS Concepts: • Hardware  $\rightarrow$  Hardware accelerators; Application specific processors; • Computer systems organization  $\rightarrow$  Single instruction, multiple data; • Computing methodologies  $\rightarrow$  Neural networks.

Keywords: Neural Processing Unit (NPU), Domain Specific Architecture (DSA), Accelerator, Deep Neural Networks (DNN), End-to-End Acceleration, Non-GEMM Layers, Large Language Models (LLM), Single Instruction Multiple Data (SIMD), Instruction Set Architecture (ISA), Tandem Processor

#### **ACM Reference Format:**

Soroush Ghodrati, Sean Kinzer, Hanyang Xu, Rohan Mahapatra, Yoonsung Kim, Byung Hoon Ahn, Dong Kai Wang, Lavanya Karthikeyan, Amir Yazdanbakhsh, Jongse Park, Nam Sung Kim, and Hadi Esmaeilzadeh. 2024. Tandem Processor: Grappling with Emerging Operators in Neural Networks. In 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2 (ASPLOS '24), April 27-May 1, 2024, La Jolla, CA, USA. ACM, New York, NY, USA, 18 pages. https://doi.org/10.1145/3620665.3640365

<span id="page-1-0"></span>![](_page_1_Figure_2.jpeg)

Figure 1. Neural operators in representative DNNs over the years.

