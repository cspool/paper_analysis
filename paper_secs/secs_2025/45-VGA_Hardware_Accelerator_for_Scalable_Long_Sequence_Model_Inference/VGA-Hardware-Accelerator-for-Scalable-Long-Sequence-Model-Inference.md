# VGA: Hardware Accelerator for Scalable Long Sequence Model Inference

Seung Yul Lee *Seoul National University* Seoul, Republic of Korea triomphant1@snu.ac.kr

Hyunseung Lee *Seoul National University* Seoul, Republic of Korea hs lee@snu.ac.kr

Jihoon Hong *Seoul National University* Seoul, Republic of Korea athlon77@snu.ac.kr

SangLyul Cho *Seoul National University* Seoul, Republic of Korea chosanglyul@snu.ac.kr

Jae W. Lee *Seoul National University* Seoul, Republic of Korea jaewlee@snu.ac.kr

*Abstract*—Effectively modeling relationships between distant elements in a long input sequence is an important task that remains challenging to this day. The state-of-the-art models for processing sequential data are self-attention-based transformer models. However, the computational complexity of self-attention is quadratic to the input sequence length, which often becomes the limiting factor in scaling the sequence length. Recently, state space model (SSM)-based global convolution models, which replace attention with convolution, have been found to be effective for modeling long sequences, with a sub-quadratic complexity using Fast Fourier Transform (FFT). However, they show suboptimal performance on data-parallel accelerators like GPU, due to the regions of extremely low compute utilization with memory bandwidth-bound operations. To address this inefficiency, this paper proposes the Vandermonde matrix Generating Accelerator (VGA), a custom accelerator that performs FFT-based convolution in an area/power-efficient manner. VGA introduces Complex number Compute Units (CCUs) to fully utilize the high on-chip SRAM bandwidth, and parameters are generated on the fly to drastically reduce the required SRAM capacity. VGA achieves 76× (48×) higher area (power) efficiency than NVIDIA A100 GPU when executing the global convolution operator of H3, a state-of-the-art SSM-based model.

