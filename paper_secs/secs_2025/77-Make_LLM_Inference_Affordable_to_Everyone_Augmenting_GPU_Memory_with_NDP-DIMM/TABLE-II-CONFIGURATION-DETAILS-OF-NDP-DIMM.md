# TABLE II CONFIGURATION DETAILS OF NDP-DIMM

| CONFIGURATION DETAILS OF NDF-DIMM.                                                   |                    |                                             |  |
|--------------------------------------------------------------------------------------|--------------------|---------------------------------------------|--|
| NDP core                                                                             |                    |                                             |  |
| Configuration: 256 multipliers, reduction tree-based accumulator, Buffer size: 256KB |                    |                                             |  |
| One NDP core per DIMM                                                                | Frequency: @ 1 GHz | area overhead: 1.23mm <sup>2</sup> per core |  |
| DIMM Parameters                                                                      |                    |                                             |  |
| DDR4-3200, 32GB/DIMM×8, 2 DIMMs/channel                                              |                    |                                             |  |
| 4 rank/DIMM, 2 bank groups/rank, 4 bank/BG                                           |                    |                                             |  |
| DIMM Timing                                                                          |                    |                                             |  |
| tRC=76, tRCD=24, tCL=24, tRP=24, tBL=4                                               |                    |                                             |  |
| tCCD S=4, tCCD L=8,tRRD S=4, tRRD L=6, tFAW=26                                       |                    |                                             |  |
| DIMM-Link Parameters                                                                 |                    |                                             |  |
| 25Gb/s/Lane, 1.17 pJ/b, 8 × Lanes (25GB/s per Link)                                  |                    |                                             |  |
|                                                                                      |                    |                                             |  |

NDP-DIMMs within a window is under 5%, indicating a balanced assignment. Nevertheless, when surpassing the window size, the performance disparity among different NDP-DIMMs varies from  $1.2 \times$  to  $2.5 \times$ . Consequently, we can leverage the neuron activity within a window to guide the remapping of cold neurons. As shown in Algorithm 1, we initially gather the activated times for each neuron i within a window and calculate the total activated neurons in NDP-DIMM j based on the current neuron mapping  $C_{j,i}$ .  $C_{j,i}$  is a binary matrix that denotes if neuron i is mapped on NDP-DIMM j. We then sort the total activated neurons for NDP-DIMMs within the window and adjust neuron mappings between DIMM pairs accordingly. Specifically, the NDP-DIMM with the largest number of activated neurons is paired with the one that has the fewest activated neurons. Finally, the most activated neurons in the NDP-DIMM pair are remapped to achieve balance. As depicted in Figure 8b, we record the activated neurons within a window into the neuron activity table, and calculate the activity for each NDP-DIMM based on the mapping results. As the count of activated neurons in DIMM-1 exceeds that of DIMM-2, neuron 5 from DIMM-1 is remapped to DIMM-2 for load balance between the two NDP-DIMMs. This strategy offers two advantages: first, the fixed inter-DIMM communication traffic is directed to different bridges to prevent congestion; second, the greedy remapping approach can quickly achieve balance with minimal data transfer.

#### V. EVALUATION

## A. Experimental Setup

1) Hermes System: The proposed Hermes system integrates a single NVIDIA RTX 4090 GPU with 24GB of graphic memory and 330 tensor TOPS (FP16) to process hot neurons. Additionally, we provide 8 NDP-DIMMs, each including 32GB DDR4 memory as the extension of GPU memory. We use PCIe 4.0 to support data interaction between NDP-DIMMs and GPU memory with a bandwidth of 64GB/s. The kernel

![](_page_8_Figure_8.jpeg)

Fig. 9. Performance comparison with existing offloading-based systems.

![](_page_8_Figure_10.jpeg)

Fig. 10. The effectiveness of activation sparsity and NDP design on Hermes.

performance of the NVIDIA RTX 4090 is measured using NVIDIA Nsight Compute [40]. Furthermore, we develop an in-house simulator by modifying Ramulator 2.0 [35], [48] to evaluate the performance efficiency of NDP-DIMM devices. For the NDP core, we implemented it in RTL and synthesized it using the Synopsys Design Compiler [56] with the TSMC 7nm technology. Table II shows the configuration details of adopted NDP-DIMMs.

- 2) Baseline Systems: We selected several offloading-based inference systems, such as Huggingface Accelerate [22], [23], FlexGen [50], and Deja Vu [34], as the baselines. FlexGen and Deja Vu are restricted to OPT models. Moreover, Deja Vu, initially optimized for LLM activation sparsity within highperformance distributed systems, has been adapted to support offloading-based serving systems. In contrast to Hermes, these methods depend solely on the basic host memory to expand capacity without offering additional computational resources. We also provided a system (Hermes-host) that offloads cold neurons to the host CPU while handling hot neurons on GPU, demonstrating the necessity of NDP-DIMMs. Hermes-host follows the configuration in [53], which equips an Intel i9-13900K processor as the host CPU (providing a maximum bandwidth of 89.6 GB/s), and also uses a single NVIDIA RTX 4090 as the GPU for hot neurons. Additionally, to highlight the significance of activation sparsity in boosting Hermes system efficiency, we also compare Hermes against a straightforward NDP-DIMM extended system (referred to as Hermes-base) that does not leverage activation sparsity in LLMs.
- 3) Workloads: We chose OPT-13B, OPT-30B, OPT-66B [63], LLaMA2-13B, LLaMA2-70B [57], and Falcon-40B [4] as target models. For the OPT series models, we utilized their native ReLU activations to achieve activation sparsity. For the LLaMA2 and Falcon models, we use the open-source models<sup>2</sup> that substituted their original activation functions with ReLU [38], [64]. Furthermore, we added additional ReLU functions before generating QKV to achieve activation sparsity in self-attention blocks. Evaluation results

<sup>&</sup>lt;sup>2</sup>The modified LLMs can be found at https://huggingface.co/SparseLLM, including both LLaMA2 and Falcon models

show that these alterations result in negligible accuracy loss (under 1%). Furthermore, we adopt ChatGPT prompts [39] and Alpaca [47] as the datasets to evaluate the end-to-end performance, following configurations in [53], [59].

*4) Evaluation Metric:* Given our focus on local deployment scenarios, we primarily optimized LLM inference with small batch sizes. We concentrated on the average number of tokens generated per second (tokens/s) to evaluate model inference efficiency. Hereafter, the number above each bar in each figure indicates the end-to-end generation speed (tokens/s). In our experiments, we used batch sizes between 1 and 16, and kept the lengths of both input and output sequences fixed at 128.

