# A. Mixture-of-Experts Models

Mixture-of-Experts (MoE) improves scalability by activating a small subset of specialized subnetworks per input, reducing computation and memory costs. A gating mechanism selects experts per token, enabling efficient large-scale and multi-task learning. Recent MoE models such as Mixtral [2], Qwen [11], and DeepSeek [1, 12] have shown significant performance and scalability gains.

MoE optimization has been explored across inference and training. Offloading methods target hybrid CPU-GPU deployment for throughput efficiency [7, 13]; serving-oriented works improve expert scheduling and placement [14–16]; and training strategies integrate data, tensor, and expert parallelism to enhance scalability [6, 17–19]. These approaches highlight the need for adaptable solutions under diverse constraints in memory, computation, and communication.

