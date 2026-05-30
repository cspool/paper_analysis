# <span id="page-17-0"></span>**B Implementation Detail of Section [4](#page-3-2)**

## **B.1 Hardware**

<span id="page-17-2"></span>Please reference Table [5.](#page-17-2)

| Item   | Type                                 | Quantity |
|--------|--------------------------------------|----------|
| CPU    | Intel Xeon Silver 4314 CPU @ 2.40GHz | 24       |
| GPU    | NVIDIA Tesla A800 80G PCI-e          | 1        |
| Memory | 16GB ECC DDR4@2666MHz                | 15       |

Table 5: Hardware Information

#### **B.2 Software**

We utilize pyTorch for the basic framework with torch.compile. We use an implementation of SwiGLU for GLU, and the implementation from Transformers [\(Wolf et al.,](#page-15-6) [2020\)](#page-15-6) MixtralModel for MoE methods.

