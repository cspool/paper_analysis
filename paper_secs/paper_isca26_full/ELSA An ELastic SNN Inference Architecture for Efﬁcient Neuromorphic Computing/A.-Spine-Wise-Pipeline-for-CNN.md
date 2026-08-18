# A. Spine-Wise Pipeline for CNN

Fig. 13(a) illustrates the spine-wise pipeline in ELSA. Spines within a convolution are data-independent of each other, allowing their computations to be performed concurrently. To start the computation of the next layer as early as possible, spines are processed in a specific order, rather

than the conventional top-to-bottom, left-to-right sequence. The calculation order is indicated by the arrows in Fig. 13(a). For example, we present a timeline showing the computation times of three convolution layers in Fig. 13(a). In the timeline, layer 3 starts to calculate spine  $S_1$  after spine  $S_2$  in layer 2 finishes the calculation, as the  $S_1$  in layer 3 is data-dependent on  $S_1, S_2, S_4, S_5$  in layer 2, which is illustrated by the dark orange regions in Fig. 13. A more general formulation of the control algorithm in Output Scheduler for spine-wise pipeline is provided in Algorithm 1, where the padded spine is excluded from the input spine. Therefore, the calculation of padded spines is skipped (line 1). Then, ELSA generates the position of output spines with the order shown by the blue arrows in Fig. 13 (lines 4-12). Finally, the calculation of output spines that are data-dependent to padded spines is delayed until the last input valid spine arrives (lines 14-18).

#### B. Token-Wise Pipeline for Transformer

Fig. 13(b) displays the token-wise pipeline in ELSA. Since the data dependence only exists within the same token, ELSA processes spikes token by token, making the token-wise pipeline among SNN layers. In detail, ELSA executes spike operators (listed in Tab. I) token-wise, triggering the next operator immediately after completing the first token of the current operator. Since generating a single token in the ssoftmax requires all query and key tokens to be available, ELSA stalls the pipeline to wait for QK spike-attention.

#### C. Data Storage, Transfer and Management

To enable spine/token-wise pipelining, ELSA employs a hierarchical data management strategy: 1) Intra-core storage. Partial sums are accumulated in membrane buffers attached to each ST-BIF neuron, serving as local state registers across time-steps. When a neuron fires, the control module reads the membrane and spike tracer and forwards them to the ST-BIF circuit for spike generation and state update. 2) Intercore transfer. Spikes delivered to spines/tokens in the next layer are packed into *flits* and temporarily stored in FIFO Queues, which act as *pipeline registers* between adjacent cores to enable non-blocking, in-order transmission.

