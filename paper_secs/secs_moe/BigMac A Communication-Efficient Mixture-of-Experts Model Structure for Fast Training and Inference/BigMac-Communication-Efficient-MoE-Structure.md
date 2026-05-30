# BigMac: Communication-Efficient MoE Structure

In this paper, we propose BigMac, a novel MoE structure that eliminates the well-known All-to-All communication bottleneck. Note that BigMac builds atop the success of fine-grained MoE models such as DeepSeekMoE and Qwen, where it also assigns a large number of small experts for each MoE layer, as shown in Figure 2c. However, beyond this similarity, BigMac has the following two main differences in structure that reflect its design rationales, compared to fine-grained ones.

- Low-dimensional communication: we scale down the input/output tokens of experts to decrease the hidden dimension of the tokens to transfer, which greatly reduces the All-to-All communication overhead.
- Performance assurance: to adapt to the decreased dimension of input/output tokens, we have to re-design the structure of each expert, using reversed projections to avoid the expert parameter count decreasing synchronously with the dimension and to align with the finegrained MoE in terms of the total parameter count, to avoid diminishing the model quality.

Below, we will detail the BigMac's design with necessary notations, summarized in Table 2.

#### **DCCA: Low-dimensional Communication Strategy**

BigMac's efficient communication strategy is motivated by the estimation of the All-to-All communication overhead in each MoE layer of the fine-grained MoE models. This overhead can be described as

$$C = 2 \times top\_k \times \frac{ep - 1}{ep} bsh, \tag{1}$$

which is proportional to the standard hidden dimension h. For the fine-grained MoE model, as shown in Figure 2b, the model follows a communicate-descend-ascend-communicate (CDAC) manner, namely, the dimension of the tokens will be scaled down by a descending projection after the first All-to-All communication, and further be scaled up before the second All-to-All communication. Therefore, actually the fine-grained MoE model always transmits the token at the highest dimension, contributing to the serious overhead analyzed previously. Inspired by this fact, we ask a key question: is it possible for models like fine-grained MoEs communicate at low-dimensional level while maintaining the overall performance without degradation?

To this end, as shown in Figure 2c, at each MoE layer, BigMac moves the descending and ascending projections outside of every small expert and places the descending projection before the first All-to-All operation for remarkably scaling down tokens sent to their best-fit experts. This change allows the communication to happen at the lowest dimension. Following this, we place the ascending projection after the second All-to-All operation to scale up the tokens to their standard sizes. In contrast to the above CDAC manner used in fine-grained MoE models, BigMac follows a descend-communicate-communicate-ascend (DCCA) manner. Within the DCCA execution, the whole process of the MoE module is described by the following equation:

$$x' = xW'_{\downarrow}; \quad y' = \sum_{i \in T} p_i(x)E_i(x'); \quad y = y'W'_{\uparrow}.$$
 (2)

Here, x and y represent the output and input of two consecutive attention layers,  $W_{\downarrow}'$  and  $W_{\uparrow}'$  are the descending and ascending projection matrices, T refers to the set of  $top_{-}k$  experts for token distribution,  $E_{i}$  refers to the expert computation in BigMac, and  $p_{i}$  refers to the gate-value of activating the  $i_{th}$  expert. Note that we can choose to use either x or x' as the input of the gating function for token routing. Here, we choose x, the vector before downscaling for routing, since the routing function is computationally efficient and a high-dimensional input vector generally leads to more accurate routing. In conclusion, DCCA reduces C in Equation 1 into a much smaller C' by changing h to rh, where r is the downscaling factor. Later, we will explain the value assignment to r and overall communication savings.

#### **BigMac Expert Design**

Based on the DCCA strategy, following the expert structure of the fine-grained MoE models is impractical. Otherwise, the expert will have much fewer parameters and consequently hurt model quality. Recall that expert computation can be described as  $E(x) = \sigma(xW_{h \times h - f})W_{h - f \times h}$ , where

 $\sigma$  is an activation function, h is the dimension of the input/output tokens and  $h_-f$  refers to the intermediate dimension. Compared to CDAC, DCCA significantly reduces the input/output dimension h, resulting in a smaller E(x) with the same intermediate dimension  $h_-f$ .

As a result, to align BigMac's model size to that of fine-grained MoE models, we should increase the dimension  $h_-f$ . The specific structure of the expert designed for adapting the DCCA strategy is shown in Figure 2c. From the appearance, it is closer to the conventional MoE structure in Figure 2a, and it can be seen as swaping the two projection matrices of fine-grained MoE experts in Figure 2b. In this way, the expert in the fine-grained MoE in Equation 3 can be replaced with the one in BigMac as shown in Equation 4:

$$E_i(x) = \sigma(xW_{i,\downarrow})W_{i,\uparrow},\tag{3}$$

$$E_i(x) = \sigma(xW_{i,\uparrow})W_{i,\downarrow}.$$
 (4)

It can be verified that the BigMac expert involves the same size and same computational complexity compared with the expert in fine-grained MoE.

#### **Advantages Beyond Efficient Communication**

Except communication efficiency, BigMac further possesses many beneficial characteristics.

Enabling Dropless Token Routing. In both training and inference phases of MoE, the token routing imbalance problem occurs frequently, and it results in a severe straggler problem. To reduce the overhead brought by the imbalanced routing, most of the existing MoE models will set a threshold for the expert capacity (Fedus, Zoph, and Shazeer 2022), determined by the expert capacity factor f, which is often set in a range from 1 to 1.25. Each expert will drop the tokens exceeding the expert capacity. It was demonstrated in (Sanseviero et al. 2023) that the quality of the MoE model can be continuously improved by increasing the capacity factor, which implies that token dropping is harmful for model's generation. To ensure the performance, the recently proposed DeepSeekMoE and Mixtral remove the expert capacity limit, at the cost of high communication overhead (Dai et al. 2024; Jiang et al. 2024; Xue et al. 2024). Fortunately, the communication overhead has been greatly mitigated in BigMac. The increased token transmission brought by removing expert capacity limit will not significantly affect the overall training or inference efficiency.

Enabling Flexible Selection of top\_k. The number of activated experts,  $top\_k$ , is another key factor affecting model quality and overall latency. To some extent, a larger  $top\_k$  contributes to better model performance (Dai et al. 2024). However, a larger  $top\_k$  corresponds to a heavier communication overhead, leading to lower efficiency for training and inference. Taking this into account, the existing MoE models generally select a relatively small  $top\_k$ . Considering the high efficiency of BigMac in both computation and communication, BigMac is able to withstand a much larger  $top\_k$  to enhance the performance. Hence, BigMac provides a more flexible choice for practitioners.

| Metrics | GPT-Fine-Grained                                | GPT-BigMac                                                         |
|---------|-------------------------------------------------|--------------------------------------------------------------------|
| #Param  | $(4h^2 + 8h + (2rh^2 + 2rh)e)l + (v + e + 2)h$  | $(4h^2 + 8h + (2rh^2 + 2rh)e)l + (v + e + 2)h + 2rlh^2$            |
| #FLOPs  | $12bslh^2(2+\frac{s}{h}+\frac{v}{2lh}+rtop\_k)$ | $12bslh^2(2+\frac{s}{h}+\frac{v}{2lh}+rtop\_k)+\mathbf{12rbslh^2}$ |
| #A2A    | $8bslhtop\_k\frac{ep-1}{ep}$                    | $8bslhtop\_k\frac{ep-1}{ep}\mathbf{r}$                             |

Table 3: Statistics of two MoE models. #Param refers to the number of parameters, #FLOPs refers to the number of floating-point operations of an iteration for different MoE structures, and #A2A refers to the transfer size of All-to-All communication.

| Metrics | <b>GPT-Fine-Grained</b> | GPT-BigMac          |
|---------|-------------------------|---------------------|
| #Param  | 3.73B                   | 3.78B (+1.35%)      |
| #FLOPs  | 3,490.67 T              | 3,649.00 T (+4.54%) |
| #A2A    | 1,488.00 GB             | 372.00 GB (-75.00%) |

Table 4: Statistics for two MoE models with BF16 precision and an expert parallelism degree of 32 on 32 devices.

| Hyper-Params                 | Values           |
|------------------------------|------------------|
| #Layers l                    | 24               |
| #Heads a                     | 16               |
| Hidden Dimension h           | 2,048            |
| Sequence Length s            | 2,048            |
| Vocabulary Size v            | 50,257           |
| Global Batch Size b          | 0.5 M            |
| Dropout Rate                 | 0.1              |
| Expert Capacity Factor f     | 1.2              |
| Load Balance Type            | aux_loss         |
| Balance Coefficient $\alpha$ | 0.001            |
| Optimizer                    | Adam             |
| $Adam \epsilon, \beta$       | 1e-8, (0.9,0.95) |
| Weight Decay                 | 0.1              |
| Learning Rate                | 3.0e-4           |
| Minimum Learning Rate        | 3.0e-5           |
| Learning Decay Steps         | 5,200            |
| Learning Rate Decay Style    | cosine           |
| Warmup Steps                 | 1,200            |
| Gradient Clipping            | 1.0              |
| Random Seed                  | 1,234            |

Table 5: Hyper-parameters of pre-training to compare the validation perplexity curves in Figure 1.

## **Analysis of Different MoE Structures**

To understand how BigMac differs from the existing fine-grained MoE structure, we analyze the parameter size and the number of FLOPs as well as the communication overhead for different MoE structures in Table 3. It indicates that the two additional projection matrices in BigMac can significantly reduce the All-to-All transmission size by a ratio of (1-r) while involving negligible overhead. For a more intuitive elaboration, we show the concrete numbers with GPT3-XL as the base model in Table 4, where we activate 8 experts out of 64 experts and set the downscaling factor r as 0.25, considering a similar setting of DeepSeek-V2,

i.e., scaling down from 5,120 to 1,536. Table 4 shows that the additional scaling matrices introduce only 4.54% FLOPs while reducing up to 75% communication overhead.

