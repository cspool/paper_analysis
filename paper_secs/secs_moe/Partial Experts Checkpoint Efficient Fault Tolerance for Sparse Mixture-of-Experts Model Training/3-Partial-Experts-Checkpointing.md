# 3 Partial Experts Checkpointing

In light of the substantial increase in checkpoint size predominantly attributed to the multiplicity of FFN experts within the MoE model, we introduce the concept of Partial Experts Checkpointing (PEC) to significantly downsize the checkpoint. In the PEC approach, a subset of experts—specifically,  $K_{pec}$  of the N experts per MoE layer—is saved, while the non-expert parts are preserved in their entirety. This strategy results in a checkpoint size comparable to that of a dense model when  $K_{pec}$  is set to 1, as illustrated in Figure 4. As a device-agnostic checkpointing mechanism, PEC is generally applicable across various MoE model training scenarios.

#### 3.1 Analysis

**3.1.1** Checkpoint Size. To accurately assess the efficacy of PEC on reducing checkpoint size, we initially define the size of a conventional checkpoint, denoted as  $C_{full}$ , which saves the states of all model parameters. The formulation is as follows:

$$C_{full} \approx (P_{ne} + P_e) \cdot (B_w + B_o) \tag{5}$$

where  $P_{ne}$  and  $P_e$  denote the number of parameters in the non-expert and expert parts of the model, respectively. Each parameter contributes fixed bytes of weight  $(B_w)$  and optimizer state  $(B_o)$ .

PEC, by contrast, only saves a subset of experts at each checkpoint, leading to a reduced checkpoint size, denoted as

 $C_{pec}$ , which is formulated as:

$$C_{pec} \approx (P_{ne} + \frac{K_{pec}}{N}P_e) \cdot (B_w + B_o) \tag{6}$$

where  $K_{pec}$  denotes the number of experts saved per MoE layer, and N denotes the total number of experts in each MoE layer. Given that the expert part typically constitutes the majority of the model parameters in existing MoE models, PEC's capability to reduce checkpoint size is considerable.

<span id="page-5-0"></span>**3.1.2 Impact on Model Accuracy.** It is critical to consider that recovering training from a PEC checkpoint may impact the model accuracy, as it causes a loss of expert updates contributed by the training input tokens. Specifically, the recovery process can retrieve the latest model states of the non-expert part and  $K_{pec}$  experts from the latest checkpoint, while the remaining  $N - K_{pec}$  experts can only be recovered to their states saved at the previous checkpointing.

To quantitatively assess the potential impact on accuracy attributed to PEC, we introduce a novel metric, the **Proportion of Lost Tokens (PLT)**. The PLT metric is designed to quantify the average proportion of tokens lost across all the MoE layers throughout the training, formulated as follows:

$$PLT = \frac{1}{N_{moe}} \sum_{i=1}^{N_{moe}} \frac{\sum_{j=1}^{N_{fault}} L_{i,j}(I_{ckpt}, K_{pec}, F)}{T_i \cdot TopK_i}$$
(7)

where  $N_{moe}$  denotes the number of MoE layers within the model, and  $N_{fault}$  denotes the count of faults encountered during the training.  $L_{i,j}$  refers to the measured number of the ith MoE layer's lost tokens caused by the jth fault, which is influenced by the checkpointing interval  $I_{ckpt}$ , the number  $K_{pec}$  of saved experts, and the function F for partial experts selection (e.g. sequential or load-aware methods). The product of the number of input tokens  $T_i$  and  $TopK_i$  of MoE gating indicates the total number of tokens processed by all experts in the ith MoE layer during the training. It is worth noting that the actual count of tokens processed by all experts is typically less than  $T_i \cdot TopK_i$ , primarily due to the token dropout imposed by the expert capacity [31].

To investigate the correlation between PLT and model accuracy, we conduct experiments that train GPT-125M-8E models with varying PEC configurations (different values of  $K_{pec}$  and  $I_{ckpt}$ ) on Wikitext dataset [39]. Each model's training process is designed to encounter a fault at the midpoint, followed by a recovery from the saved PEC checkpoint.

As evidenced in Figure 5, the final validation loss of the models experiences fluctuations (4.8808-4.8856) yet remains comparable to the non-fault case (4.8851) when PLT is below 3.75%. It substantiates the efficacy of PEC in minimizing checkpoint size without harming model accuracy in the case of limited PLT (more comprehensive accuracy evaluations are in Section 6.3). Additionally, the results highlight a correlation between smaller  $K_{pec}$  and larger  $I_{ckpt}$  with increased PLT, which may impact the model accuracy.

#### 3.2 Partial Experts Selection

As PEC only saves a subset of experts at each checkpoint, selecting which experts to save is important. Different functions of partial expert selection can lead to variations in PLT and the recovered model states, thereby potentially impacting the final model accuracy.

More importantly, considering that experts within each MoE layer are distributed across various ranks and devices via EP, as depicted in Figure 4, the partial expert selection significantly affects the workload distribution across ranks, thus impacting the time cost of checkpointing. The most imbalanced workload scenario, for instance, involves checkpointing "Expert(1-0, 3-0, 5-0, 7-0)", all located in "Ranko". In this case, the checkpointing duration is primarily prolonged by "Ranko", which bears the heaviest workload.

**Sequential Selection.** Given the challenges associated with the selection of partial experts, we propose a sequential selection strategy that sequentially alternates the target experts, incorporating an interleaved schedule across MoE layers and EP ranks. For instance, as illustrated in Figure 4, at the first checkpointing time, "Rank0" saves "Expert(1-0, 7-0)", "Rank1" saves "Expert3-1", and "Rank2" saves "Expert5-2". At the next checkpointing time, "Rank0" saves "Expert5-0", "Rank1" saves "Expert(1-1, 7-1)", and "Rank2" saves "Expert3-2". With this strategy, PEC can achieve a balanced checkpointing workload while maintaining an acceptable PLT.

**Load-aware Selection.** We extend our investigation into the function of partial expert selection by incorporating a load-aware approach that prioritizes the checkpointing of  $K_{pec}$  experts, characterized by the highest number of unsaved updates. Based on our empirical results, load-aware selection achieves model accuracy on par with sequential selection but necessitates more complicated control mechanisms and incurs higher costs, making it a less favorable option.

