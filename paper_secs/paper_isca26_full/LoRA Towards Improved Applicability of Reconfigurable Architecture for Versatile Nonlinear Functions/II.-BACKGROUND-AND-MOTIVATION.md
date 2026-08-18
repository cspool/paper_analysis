# II. BACKGROUND AND MOTIVATION

This section provides an overview of the related technologies and highlights our motivation.

## *A. Nonlinear Functions in Hardware*

Nonlinear functions are typically implemented by three categories of approaches.

- *1) Iterative-based Approach:* Based on the selected coordinate system and rotation/vectoring mode, the Coordinate Rotation Digital Computer (CORDIC) algorithm approximates the targeted function iteratively [10], [47] with a low hardware overhead thanks to its simple operations (e.g., *add* and *shift*). However, CORDIC requires more iterations to achieve better accuracy, resulting in a significant increase in operation latency. Besides, it only supports a limited input range, and some versions require additional processing before/after rotation.
- *2) LUT-based Approach:* For the LUT-based approach [35], [77], the output values of the targeted function are precomputed and stored in memory. Then, the input values act as the index to address the memory for the required outputs. Although the LUT-based approach reduces computation latency and simplifies implementation, it requires a large memory to achieve high accuracy. In addition, the LUT-based approach must trade off area overhead against the supported input range, thereby bounding the approximation to a specific region of the function, which limits its applicability.

The polynomial-based approach approximates nonlinear functions, such as the Taylor series [21], [50], [56], using degree-*n* polynomials. However, it faces limitations: (1) Computational Overhead: High-degree polynomials require multiple multiply-add (MAD) operations. While Horner's rule [33] simplifies computation, the overhead remains substantial. For instance, a 6th-degree Taylor polynomial for the *exp* function requires at least 6 MADs [4]. (2) Input Range: Taylor series accuracy is high only near the expansion point [21], limiting its effectiveness for applications with diverse input ranges. Although PICACHU [56] decomposes ranges, the associated computational overhead is still significant.

Given the complexity and wide input range of the target function, using a single polynomial for approximation is challenging. A more flexible approach is piecewise approximation with different polynomials. To reduce computational overhead, most existing works focus on single and quadratic degree polynomials, balancing the number of intervals [4], [23], [25], [39], [45], [69]. This approach can be treated as the combination of LUT and polynomial approaches, but requires less memory than the LUT-based approach, since there are only *breakpoints* and *coefficients* of each polynomial to be stored. Consequently, the architecture is reconfigurable for different functions by adjusting the *coefficients*. However, there are several optimization challenges: (1) Data format: Many prior approaches are designed for specific data formats (e.g., fixedpoint or floating-point) [25], [39], [45], while *LoRA* should support more formats for future flexibility. (2) Nonuniform segmentation: As demonstrated in [4], nonuniform segmentation concentrates more pieces where the function changes quickly or has high curvature, resulting in lower approximation errors than uniform partitioning. Hence, an efficient segment strategy is crucial when the number of intervals is limited.

Motivation: To address the limitations above, we propose a piecewise approximation using Chebyshev polynomials. In addition to maintaining flexibility for versatile nonlinear functions across various input ranges, a critical challenge is to

<sup>1</sup>LoRA is part of the Fusion framework, whose source code is available at: https://github.com/Dai-dirk/COFFA

<sup>2</sup>We name it *LoRA* because it is a Logarithmic-arithmetic-based Reconfigurable Architecture for executing nonlinear functions.

reduce the computational overhead of high-degree polynomials. Consequently, we employ the logarithmic number system to address this challenge.

