# A.8 Notes

- FPGA synthesis may take 2-4 hours depending on design complexity
- GPU memory requirements scale with model size and batch size
- Cross-device communication latency depends on PCIe generation and utilization
- Results may vary with different FPGA boards due to timing variations
- Power measurements require external monitoring tools for highest accuracy

## A.9 Hardware Platform and Toolchain

We implement DFVG on a heterogeneous system consisting of Xilinx V80 FPGAs and NVIDIA RTX 4090 GPUs. The FPGA logic is synthesized using Vivado 2022.2, and the runtime communication uses PCIe with custom memory-mapped buffers. The host controller is written in C++ with support for non-blocking draft and verify streams.

#### Token Stream Serialization Format

Tokens generated on FPGA are serialized into a lightweight stream format containing token IDs, timestamps, and confidence scores. Each entry is 16 bytes, with 4 bytes for token ID, 4 for confidence, and 8 for alignment. The stream is transferred via DMA to GPU memory and queued for batch verification.

## C. Future Directions

While DFVG demonstrates the effectiveness of FPGA-GPU cooperation for single-round speculative decoding, future work includes:

- Supporting multi-round speculative decoding by pipelining multiple draft-verify iterations.
- Extending support to other model families such as Mistral, DeepSeek, and Mamba.
- Integrating with LLM agent frameworks (e.g., LangChain) to accelerate reasoning workflows.
- Exploring compiler-level fusion of decoding logic with attention sparsity patterns.

