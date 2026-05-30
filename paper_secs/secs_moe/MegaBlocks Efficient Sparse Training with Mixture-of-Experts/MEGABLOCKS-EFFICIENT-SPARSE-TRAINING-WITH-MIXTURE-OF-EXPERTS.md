# MEGABLOCKS: EFFICIENT SPARSE TRAINING WITH MIXTURE-OF-EXPERTS

Trevor Gale <sup>1</sup> Deepak Narayanan <sup>2</sup> Cliff Young <sup>3</sup> Matei Zaharia <sup>1</sup>

## ABSTRACT

We present MegaBlocks, a system for efficient Mixture-of-Experts (MoE) training on GPUs. Our system is motivated by the limitations of current frameworks, which restrict the dynamic routing in MoE layers to satisfy the constraints of existing software and hardware. These formulations force a tradeoff between model quality and hardware efficiency, as users must choose between dropping tokens from the computation or wasting computation and memory on padding. To address these limitations, we reformulate MoE computation in terms of block-sparse operations and develop new block-sparse GPU kernels that efficiently handle the dynamism present in MoEs. Our approach never drops tokens and maps efficiently to modern hardware, enabling end-to-end training speedups of up to 40% over MoEs trained with the state-of-the-art Tutel library and 2.4× over dense DNNs trained with the highly-optimized Megatron-LM framework.

