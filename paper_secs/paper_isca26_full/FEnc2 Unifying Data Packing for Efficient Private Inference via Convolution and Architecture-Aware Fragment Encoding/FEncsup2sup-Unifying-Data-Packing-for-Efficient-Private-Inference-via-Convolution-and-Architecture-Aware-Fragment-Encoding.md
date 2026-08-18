# FEnc<sup>2</sup>: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding

Ran Ran<sup>1</sup>, Zhaoting Gong<sup>1</sup>, Nuo Xu<sup>2</sup>, Yuanchao Xu<sup>3</sup>, Fan Yao<sup>4</sup>, and Wujie Wen<sup>1</sup>

<sup>1</sup>North Carolina State University

<sup>2</sup>University of Minnesota

<sup>3</sup>University of California, Santa Cruz

<sup>4</sup>University of Central Florida

<sup>1</sup>{rran, zgong6, wwen2}@ncsu.edu

<sup>2</sup>xu001536@umn.edu, <sup>3</sup>yxu314@ucsc.edu, <sup>4</sup>fan.yao@ucf.edu

Abstract—Fully Homomorphic Encryption (FHE) enables privacy-preserving machine learning but incurs extreme computational and memory overhead. These costs stem not only from slow low-level primitives such as Number Theoretic Transform (NTT), rotation, and key-switching, but also from inefficient ciphertext packing at the application level. Existing packing strategies typically preserve either neighboring data elements or feature-grouping information, but not both, leading to wasted ciphertext slots, excessive rotations, and inflated ciphertext counts. We propose  $FEnc^2$ , a unified and principled fragment-based encoding framework that optimizes slot utilization, rotation complexity, and ciphertext density for CKKS-based private convolutional neural network inference. Rather than applying static or layer-isolated heuristics, FEnc<sup>2</sup> introduces (1) Conv-aware Encoding, which analytically selects an optimal fragment (block) size to decouple spatial dependencies and jointly minimize inner-outer rotations across layers, and (2) Arch-aware Ct Compression, which dynamically restores ciphertext density after feature- or channelreduction layers. Together, these transformations reshape encrypted workload structure, reducing homomorphic operations by one to two orders of magnitude. With full memory capacity utilized (i.e., at maximum batch size), FEnc<sup>2</sup> achieves end-toend latency speedups over the state-of-the-art Orion of up to  $228.83 \times (\tilde{G}P\tilde{U})$  and  $226.06 \times (CPU)$  for LeNet (MNIST), and up to  $4.55 \times$  (GPU) and  $9.43 \times$  (CPU) for MobileNet (ImageNet). Importantly,  $FEnc^2$  is hardware-agnostic but architecturally transformative: by optimizing encrypted tensor layout before execution, it reduces ciphertext count and workload pressure on hardware, complementing primitive-level optimizations (e.g., NTT/keyswitch accelerators). This demonstrates that applicationlevel data layout is a first-order architectural design dimension for encrypted inference and a critical enabler for next-generation FHE systems.

Index Terms—CKKS, Data Encoding, Fully Homomorphic Encryption, Hardware Acceleration, Private Machine Learning.

