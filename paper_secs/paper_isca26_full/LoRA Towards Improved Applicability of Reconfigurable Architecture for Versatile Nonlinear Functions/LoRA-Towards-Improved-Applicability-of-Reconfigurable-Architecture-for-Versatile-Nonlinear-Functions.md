# LoRA: Towards Improved Applicability of Reconfigurable Architecture for Versatile Nonlinear Functions

Yuan Dai\*, Guibin Zou\*, Yuanda Yang, Huan Lin, Jiahang Lou, Yiwen Luo, Xinyu Cai, Wenbo Yin, Wai-Shing Luk, Lingli Wang<sup>™</sup>

State Key Laboratory of Integrated Chips and Systems, Fudan University, China Email: daiy21@m.fudan.edu.cn, llwang@fudan.edu.cn

Abstract—Coarse-Grained Reconfigurable Architecture (CGRA) emerges as a competitive accelerator for computation-intensive applications. However, most CGRAs primarily focus on linear operations, which makes it challenging to efficiently support nonlinear functions that are increasingly important in emerging applications. This limitation hinders the potential of CGRAs to accelerate modern computation-intensive applications, particularly those that rely on complex nonlinear functions.

In this paper, we propose several key approaches to enable CGRA to support versatile nonlinear functions. First, we develop a Chebyshev-based approximation algorithm that employs a novel segmentation strategy, along with considering the properties of the target function to identify a suitable Chebyshev polynomial for each sub-interval. Second, based on the Logarithmic Number System, we design a lightweight unit to efficiently compute the polynomials and complex operations. Finally, we develop a comprehensive end-to-end framework LoRA with architectural and compiler support for implementing nonlinear functions on the CGRA. Experimental results show that LoRA can provide efficient nonlinear function implementation and achieve  $23.33 \times$  and  $2.18 \times$  average performance improvements compared to an STM32H750 microcontroller unit and a stateof-the-art (SOTA) CGRA. In addition, LoRA achieves a 2.13× average energy efficiency gain compared to the SOTA CGRA.

Index Terms—Reconfigurable Hardware, Nonlinear Function, Chebyshev Polynomial, Logarithmic Number System

## I. Introduction

The demise of Dennard scaling and the impending end of Moore's Law have spurred the rapid advancement of accelerators in various domains [11], [14], [28], [30]. Although these specialized accelerators can achieve optimized performance and energy efficiency, they are inflexible. In addition, such a specialization leads to dark silicon, since many accelerators remain underutilized across diverse workloads.

To overcome the above limitation, the Coarse-Grained Reconfigurable Architecture (CGRA) provides a balanced solution for performance, efficiency, and post-silicon reconfigurability. Consequently, CGRA has attracted considerable attention from both academia [16], [22], [26], [27], [38], [42], [51], [56], [61], [64], [70], [78] and industry [24], [59].

This work is supported by National Science and Technology Major Project (2021ZD0114701). We thank all the reviewers for their time and thoughtful feedback on this paper.

Conventionally, a CGRA consists of word-level reconfigurable processing elements (PEs) connected by reconfigurable interconnects. Given a targeted computation-intensive loop kernel, the compiler transforms it into a Data Flow Graph (DFG) and then maps it to the CGRA for execution. However, as shown in Table I, with the rapid development of applications, the applicability of conventional CGRA becomes limited. This is because these emerging applications often contain nonlinear functions (e.g., elementary functions) that current CGRAs can not support. As a result, most existing CGRAs offload these computations on the host CPU, leading to performance degradation.

TABLE I OVERVIEW OF NONLINEAR FUNCTIONS IN DIFFERENT APPLICATIONS

| Categories                 | Nonlinear Functions                                                   | Typical Application                                                           |
|----------------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------------|
| Activation<br>Function     |                                                                       | Deep Neural Network [34], [36], [65]<br>Large Language Model [20], [37], [57] |
| Trigonometric<br>Functions | $\sin(x)$ , $\cos(x)$ , $\sinh(x)$ , $\cosh(x)$ , $\arcsin(x)$ , etc. | Discrete Cosine Transform [1]<br>Synthetic Aperture Radar [13]                |
| Others                     | $log_b(x), x^y$ , etc.                                                | Digital Signal Processing [49]                                                |

As shown in [21], a simple way to enhance CGRA applicability is by adding off-the-shelf Intellectual Properties (IPs) for each nonlinear function [19], [21]. However, this approach is inefficient because: (1) hardware overhead grows with the number of IPs, and (2) while well-developed IPs for elementary functions provide accurate results, many realworld applications are error-tolerant [13], [18]. For example, PICACHU [56] uses Taylor expansions to approximate four nonlinear functions (exp, log, sin, cos) in large language models (LLMs), but requires more PEs as approximation accuracy increases. This problem becomes more significant when a compound nonlinear function is involved. NX-CGRA [54] transforms complex nonlinear operations into basic arithmetic (e.g., addition, multiplication) via its compiler, executing them in parallel on its Very-Long-Instruction-Word-based CGRA. While flexible, this approach demands more resources, resulting in higher latency and power consumption. Consequently, there is a great demand to design a reconfigurable and general-

<sup>\*</sup>Co-first author.

purpose approximating unit within the CGRA, making it applicable for versatile nonlinear functions.

Prior research proposes various methods for approximating nonlinear functions, including LUT-based and polynomialbased approaches [4], [7], [35], [56], [80]. In this paper, we combine polynomial and LUT approaches to approximate nonlinear functions. However, designing a flexible approximating unit for a general-purpose CGRA faces two challenges: (1) The approximation approach should be universal to be adopted by versatile nonlinear functions with user-defined input ranges, rather than being limited to a subset of functions [23], [56]. (2) The overhead of multiply-addition operations required by high-degree approximation polynomials cannot be ignored [4], [50], requiring an efficient polynomial computing approach.

To address the above challenges, this paper presents several contributions, where all the source codes are available from<sup>1</sup> :

- Algorithm. Based on the Chebyshev polynomial, we propose an efficient piecewise approximation algorithm, considering: (1) algebraic properties of the targeted function; (2) properties of the underlying architecture; (3) breakpoints for the nonuniform segmentation.
- Architecture. We integrate a lightweight reconfigurable functional unit *XCore* to CGRA for the nonlinear function, where the results of the proposed approximation algorithm configure this unit. By adopting the logarithmic number system (LNS), several nonlinear functions and the expensive multiply-addition operations in high-degree polynomials can be simplified.
- Framework. We propose *LoRA*<sup>2</sup> , an end-to-end framework with architectural and compiler support for efficient nonlinear function computation in CGRA. By leveraging the proposed algorithm-hardware co-optimization, our CGRA is future-proof and can handle a wide range of nonlinear functions in upcoming applications.

Paper Organization: The background and motivation are shown in Section II, followed by the error analysis in Section III. Next, we introduce the architecture of *XCore* in Section IV and the Chebyshev-based approximation algorithm in Section V, respectively. An overview of the *LoRA* framework is provided in Section VI. The evaluation setup and results are presented in Section VII and Section VIII. Finally, we make the conclusion in Section IX.

