# IX. CONCLUSION

This work presented MLX, a unified algorithm-architecture co-design for structured LLM acceleration. By combining semantic FFT compression with hierarchical butterfly decomposition, MLX exposes predictable, accuracy-tunable sparsity. We identify a common staged dependency pattern across FFT, BSMM, and related structured operators, and build Multi-Layer Execution to fold these deep operators into compact spatial dataflow. MLX integrates skip-hop routing, tag-based scheduling, and decoupled pipelines to sustain high utilization under staged execution. Experiments on Llama2-7B and InternLM2-7B show 57%-72% compute reduction with minor accuracy loss. Our 12 nm prototype demonstrates competitive gains over edge GPUs and prior sparse accelerators, while scaling to larger meshes and longer sequences. Overall, these results suggest that MLX extends beyond butterfly sparsity and provides a general substrate for efficiently accelerating a broader class of structured operators.

#### ACKNOWLEDGMENT

This work was supported by National Key R&D Program of China (Grant No.2023YFB4503500), Jiangsu Provincial Frontier Technology RD Program (Grant No.BF2024029), CAS Project for Young Scientists in Basic Research under Grant YSBR-029, National Natural Science Foundation of China (Grant No.62502498), and Beijing Natural Science Foundation (Grant No.L234078).

