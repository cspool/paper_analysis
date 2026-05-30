# 6 Implementation

We implement ElasticMoE as three separate modules, Coordinator, IMM, and HMM, that communicate over inter-process communication (IPC) using UNIX domain sockets via the ZMQ framework. The Coordinator exposes TCP APIs for receiving user requests and forwards them to the active inference instance in the IMM using standard OpenAI-style inference APIs. The IMM manages multiple inference instances, implemented as almost-unmodified vLLM instances, ensuring compatibility with a wide variety of MoE models.

Within the HMM, the control plane is implemented in Python, while the data plane is written in C++ using Huawei CANN APIs for performance-critical operations. These C++ primitives (e.g., zero-copy sharing, peer-to-peer transfers, virtual page remapping) are exposed to Python via PyBind11. To maximize compatibility, the HMM also supports the vLLM model loader backend, allowing all models supported by vLLM to be seamlessly integrated into ElasticMoE.

