# 4 Method

In general, we adopt the zigzag computation order proposed in FlexGen [42]: loading the weights from CPU<sup>6</sup> and performing the computation layer by layer. For the prefill stage, we perform all the computation on GPU and offload KV cache to CPU for all the micro-batches<sup>7</sup>. For the decode stage, within each layer, we propose a fine-grained GPU-CPU-I/O pipeline schedule (§4.1) to increase the utilization of GPU, CPU, and I/O in *decode* stage. We also build a performance model (§4.2) based on the HRM we extended from the Roofline Model to help search for the best hyper-parameters for the pipeline schedule, including the assignment of devices to perform different computations, the batch size, the micro-batch size and the ratio of weights to be placed on GPU statically. Note that for the memory-constrained scenarios we target in this

paper, CPU attention is consistently better than GPU attention, according to our performance model. We also conduct an ablation study in §6.3 to show how best policy changes under different hardware configurations.

## <span id="page-5-0"></span>4.1 GPU-CPU-I/O Pipeline Schedule

