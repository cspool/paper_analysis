# <span id="page-2-1"></span>2.2 Characteristics of MoE Inference

Here, we outline three key characteristics of MoE inference, which shed light on the inference bottlenecks discussed in §[2.3.](#page-2-2)

Synchronous all-to-all communications. In this process, all-to-all communication is synchronous, meaning that computation (including FFN and aggregation) can only begin once every GPU has completed data transmission. This leads to the GPU computation resource idleness.

Reversed all-to-all communications. Within the same forward pass, the two all-to-all communications are reversed. For each data transfer from GPU to in the first communication, there is a corresponding data transfer from GPU to in the second. The data sizes in these transfers are identical, as the FFN architecture ensures that the input and output data sizes are the same.

Non-overlapping communication and computation. Communication and computation processes do not overlap; each step can only commence after the previous one is completed.

