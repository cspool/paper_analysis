# *A. Latency and Memory Consumption*

The most important metrics in machine learning model deployment are execution time (i.e., latency) and memory consumption. A shorter latency ensures a more timely response from the service, whereas lower memory consumption indicates lower resource usage and potential to accommodate larger batch sizes. In this subsection, we compare MoE inference performance along these two axes. The mini batch size is set to 8 for language modeling and 48 for machine translation. We use a dense model of similar FLOPs as the baseline for comparison.

Latency. Figures [2](#page-0-1) shows the latency and memory consumption of the MoE Transformers of interest and that of their dense counterparts. Although in theory, MoE Transformers exeucte a similar number of FLOPs compared to the baseline dense models, in practice they are significantly slower. For the Language Model, the dense model requires 74.2ms, whereas the MoE Transformer requires more than 1.09s. For Machine Translation, the dense model executes the encoder and decoder in 101ms and 32ms, respectively, but the MoE Transformer requires 2.26s and 90ms.

Figure [5](#page-3-1) breaks down latency under different scenarios. The latency gap has been previously attributed to the frequent allto-all communication collective in MoE models. [\[21\]](#page-11-1) While all-to-all collectives does increase latency under multi-node deployment, we note that this is not the only source of latency.

![](_page_3_Figure_8.jpeg)

<span id="page-3-3"></span>Fig. 4. MoE vs Dense model memory footprint comparison during inference. The MoE models require significantly more memory usage when deployed on GPUs. Besides the large memory consumption due to the expanded model capacity (introduced by expert parameters), it also requires more memory for activation. (results for batchsize=48 for MT, and batchsize=8 for LM. Note that these are the largest batch sizes that are feasible to run under the baseline implementation.)

![](_page_3_Figure_10.jpeg)

<span id="page-3-1"></span>Fig. 5. MoE Model latency breakdown. Besides all-to-all communication, other components of the model, such as gating function and expert execution, are also inefficient. Communication overhead increases significantly when more than one node is involved.(Results for batch size=8 for LM and batch size=48 for MT).

In Section [III-B,](#page-3-2) we will discuss these extra sources of latency.

Memory. We also observe a large increase in memory consumption for MoE models (see Figure [4\)](#page-3-3). For LM, the dense model only requires 2.2GB on each GPU whereas the MoE model requires 18.88GB at its peak, an increase of 8.58×. For MT, the dense and MoE models use 7.02GB and 21.16GB, respectively, an increase of 3.01×.

We perform a detailed analysis by separating static and dynamic memory usage. Static memory consumption refers to memory allocated to model parameters, whereas dynamic memory consumption refers to memory allocated on demand, usually by network activations. Due to the fact that each GPU accommodates more than one expert during inference, the increase in static memory is expected. However, we observe that the peak dynamic memory consumption also increases significantly in both cases, which is surprising.

