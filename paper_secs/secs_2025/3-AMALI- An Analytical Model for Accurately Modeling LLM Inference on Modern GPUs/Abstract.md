# Abstract

Large language model (LLM) inference applications are surging in recent years, which largely relies on modern GPUs. On the other hand, GPU analytical model is a commonly used tool for architects to precisely identify bottlenecks quickly with deep insights. However, existing GPU analytical models fall short of accurately modeling LLM inference applications on modern GPUs, because of unsuitable tensor core modeling, ignoring constant cache as well as instruction cache modeling and abstracting away important details for LLM inference applications.

To address this problem, we propose a novel analytical model dubbed AMALI to accurately model LLM inference on modern GPUs with three innovations. First, we develop an instruction modifier and throughput based tensor core model by accurately capturing the math pipe throttle stalls to enhance the architecture modeling for modern GPUs. Second, we propose analytical models for constant cache and instruction cache by developing micro-benchmarks to measure CUDA kernel launching latencies. This significantly improves AMALI's accuracy compared to real GPU hardware. Finally, we design a multi-warp model by leveraging warp instruction number distribution to reflect LLM inference application characteristics.

We validate AMALI on an A100 GPU by using typical LLM inference applications. The results show that AMALI reduces the MAPE (mean absolute percentage error) from 127.56% to 23.59%

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

ISCA '25, Tokyo, Japan

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-1261-6/25/06

<https://doi.org/10.1145/3695053.3731064>

compared to the state-of-the-art GCoM model. We further showcase that AMALI can be used to explore architecture design space by designing the tensor core capability of H100. The results show that AMALI accurately predicts the end-to-end performance improvements with the enhanced tensor core capability.

