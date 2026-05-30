# Abstract

The quantization of Large Language Models (LLMs) poses significant challenges due to the heterogeneous nature of feature point distributions in low-bit quantization scenarios, including salient points, normal outliers, and massive outliers. These challenges are particularly pronounced in supporting both weight-only and weight-activation quantization modes, as existing methods often focus on a single mode and fail to address the diverse feature characteristics holistically, resulting in suboptimal model accuracy and hardware efficiency trade-offs.

To tackle these limitations, we introduce Amove, a novel codesign framework that synergistically integrates data type and hardware architecture design for efficient LLM quantization. Our approach is threefold: First, we conduct a comprehensive analysis of quantization granularity and propose a residual approximation mechanism that balances model accuracy and memory overhead under fine-grained quantization. Second, we design a flexible finegrained grouped vectorized data type, enabling seamless support for both weight-activation and low-bit weight-only quantization modes within a unified framework. Third, we implement the hardware architecture of Amove on both GPU tensor core and systolic arraybased architectures. The Amove-enhanced tensor core achieves an average speedup of 2.13× and a 1.70× reduction in energy consumption over the state-of-the-art OliVe design. Furthermore, an Amove-based accelerator achieves up to 2.67× speedup and 1.68× energy reduction over the state-of-the-art accelerator.

<sup>∗</sup>Co-corresponding authors.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MICRO '25, Seoul, Republic of Korea

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1573-0/25/10 <https://doi.org/10.1145/3725843.3756113>

CCS Concepts • Computing methodologies → Neural networks; • Computer systems organization → Data flow architectures.

