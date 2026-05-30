# Prediction Is All MoE Needs: Expert Load Distribution Goes from Fluctuating to Stabilizing

Peizhuang Cong, Aomufei Yuan, Shimao Chen, Yuxuan Tian, Bowen Ye, Tong Yang Peking University

#### Abstract

MoE facilitates the development of large models by making the computational complexity of the model no longer scale linearly with increasing parameters. The learning sparse gating network selects a set of experts for each token to be processed; however, this may lead to differences in the number of tokens processed by each expert over several successive iterations, i.e., the expert load fluctuations, which reduces computational parallelization and resource utilization.

To this end, we traced and analyzed loads of each expert in the training iterations for several large language models in this work, and defined the transient state with "obvious load fluctuation" and the stable state with "temporal locality". Moreover, given the characteristics of these two states and the computational overhead, we deployed three classical prediction algorithms that achieve accurate expert load prediction results. For the GPT3 350M model, the average error rates for predicting the expert load proportion over the next 1,000 and 2,000 steps are approximately 1.3% and 1.8%, respectively. This work can provide valuable guidance for expert placement or resource allocation for MoE model training. Based on this work, we will propose an expert placement scheme for transient and stable states in our coming work.

#### Index Terms

MoE, expert load, prediction, model training, load balancing

#### I. INTRODUCTION

In recent years, the magnitude of deep neural networks has scaled up sharply, especially with the emergence of Large Language Models (LLMs), whose parameter quantities have exceeded the trillions level. Although the expansion of parameters enhances the model's ability to handle complex tasks, it also results in enormous computational resources consumption. To alleviate this issue, some LLMs employed the sparse Mixture of Experts (MoE) technique, an architecture that assigns input to some expert networks and integrates their outputs as final output. By employing gating networks to select a subset of experts, sparse MoE LLMs can achieve comparable performance to the dense models while activating fewer parameters, thereby being able to scale model size without proportionally scaling up computational requirements.

Unfortunately, due to the dynamic nature of expert activation requires holding all experts' parameters, MoE LLMs suffer from high memory usage. When allocating GPU resources equally to all experts, it can result in unnecessary wastage of resources due to varying loads of experts (i.e., the volume of processing tokens). Then, some studies tried to add a regularization term to the loss function to balance the load distribution of experts. However, some datasets may be inherently biased towards expert activations, and thus, the accuracy of the model may be affected if the gate networks are over-interfered. From another perspective, the ideal way is to allocate GPU resources to experts based on their individual activity level. The more loaded experts occupy more resources and vice versa, which can guarantee model training efficiency with minimal resources. Nevertheless, it is not easy to make certain of the load of experts during the model training.

In existing researches, some works leverage simple prediction to adjust resources for experts, such as the moving average algorithm (i.e., using the average of historical data as the prediction result directly) [\[1\]](#page-7-0), [\[2\]](#page-7-1). In fact, such an approach is not always practical. According to our observation and analysis of the expert load data traced from model training, the load of experts has the following characteristics: (a) the load distribution of experts in a MoE layer tends to stabilize gradually as the training iterates, but there are prominent fluctuations at the early stage of training; (b) the load fluctuations of experts in different layers are different, and the fluctuation of load proportion of experts in the shallow layer is more noticeable than deep layers. As aforementioned, predictions can hardly provide valid indications for expert placement or resource allocation in the fluctuation phase. Therefore, it becomes a critical knot when the load proportion of experts starts to convert from fluctuating to relatively stable during the training process.

To better understand and analyze the expert load during model training, in this work, we conducted extensive training of MoE models under various scenarios (including different cases in terms of model architecture, parameters scale, hyper-parameters, routing strategy, load-balancing loss, dataset, etc.), and traced the load distributions of the experts in all these cases. Based on the in-depth analyses of traced data, we provided insights into the characteristics of the experts load and define transient state and stable state for model training. Moreover, given the computational overhead and the features of both states, we deployed three classical prediction algorithms that can achieve high-precision prediction of expert load distribution.

![](_page_1_Figure_0.jpeg)

<span id="page-1-0"></span>Fig. 1. Load proportions of experts in the MoE layer

In summary, the main contributions of this work are as follows:

- We reveal both the transient and stable states of the expert load during the training process of MoE models.
- We conduct extensive experiments to analyze the changing characteristics of the expert load and verify the defined transient and stable states.
- We achieve high-precision expert load prediction by using three classical prediction algorithms given the features of transient and stable states.

#### II. PRELIMINARY AND RELATED WORK

#### A. Mixture of experts

The Mixture of Experts (MoE) model is initially introduced by [3] in 1991, which integrates different networks through supervised learning, where each network is responsible for processing a specific subset of the training examples. In 2017, Google achieved the first sparsely gated MoE architecture by adding MoE between LSTM layers [4]. Given the advantages of sparse MoE architecture over traditional dense networks in terms of capacity and computational efficiency, MoE has been widely employed in various models, especially in transformer-based large language models [5]–[8].

For the MoE architecture, the Gating Network plays a key role, which calculates the weights of each expert based on the input data received from the previous layer, and then distributes the input data to the selected experts, i.e., the activated experts, based on the weights and the assigned selecting rules (e.g., Top-K). The results processed by all activated experts will be combined according to the corresponding weights by weighted summation or other forms of integration to generate the final output.

#### B. Balancing expert load

The learnability of the gating network makes the probabilities of tokens distributed to experts uncontrollable. In other words, during the training process, it is possible that some experts may receive an excessive number of tokens, while others receive too few. This imbalance in expert load distribution does not align with the intentions of the MoE architecture and can adversely affect computational efficiency. To alleviate this issue, several studies have proposed load-balancing strategies for experts during the model training process.

Load balancing loss: In order to make each set of tokens be distributed to the experts evenly, it is possible to add an auxiliary load balancing loss that usually defined as the sum of the activation entropy of all experts, which ensures that all experts can be trained sufficiently [4], [5].

Capacity factor: To further constrain the number of tokens processed by each expert, it is possible to limit the expert capacity, i.e., to stipulate each expert can only process a fixed number of tokens,  $CF * (\frac{Num_{tokens}}{Num_{experts}})$ . The expert capacity can be adjusted by setting different CF values, and when the experts receive more tokens than the set capacity, these extra tokens will be passed directly to the next layer through the residual connection [6].

Expert-based routing: In contrast to tokens choose experts, [9] proposed the way that experts choose tokens. Based on the token-expert affinity scores matrix, which is produced by the dot product of the token embedding and the expert embedding, each expert chooses its corresponding Top-K tokens.

Hash-based routing: As different from all of the above, [10] and [11] replaced the learning gated network with the hash-function, all tokens choose the corresponding experts according to the hash calculation, which can avoid the issue of unevenly balanced load of experts during the training process.

However, these methods may suffer from the problem that some tokens may be neglected for training the model.

#### *C. Load-based expert placement*

Despite the implementation of various expert load-balancing strategies, imbalances in load distribution still occur in practical scenarios. To ensure the training efficiency of the MoE models, it is possible to adjust the resource allocation for experts or to strategically reallocate experts more effectively. Based on the load of experts, FlexMoE employs a simple but effective heuristic algorithm to dynamically optimize resource placement during model training, enhancing the model training performance [\[1\]](#page-7-0). Prophet leverages the temporal locality of expert loads to schedule resource allocation operations using a layer-wise, fine-grained strategy [\[2\]](#page-7-1).

### III. MOTIVATION

We traced and investigated the activation frequency of each expert by tokens in each iteration during the training of GPT-3 125M and GPT-3 350M models, which revealed two distinct states of expert load. The first state appears in the early iterations of the training, featuring non-regular variations of the load of each expert in successive iterations. We refer to this period as the transient state. With several iterations of training, the second state emerges in which the loads of each expert are similar in adjacent iterations, i.e., showing temporal locality. This phase is referred to as the stable state.

In the transient state, the intrinsic fluctuation of the loads of the expert makes it hard to get an accurate load proportion by predicting, while it is contrary in the stable state. Therefore, accurately distinguishing state transition is crucial for model training. This is because load prediction can be leveraged to guide the resource allocation for experts during the stable state. In contrast, during the transient state, it is essential to reserve sufficient resources for each expert to cope with load bursts so as to ensure the model training efficiency.

It is important for resource-efficient training in MoE models to allocate resources flexibly and dynamically according to the load of experts. In order to better understand the load state of experts during training process, it is necessary to conduct further investigations and purposive experiments. Intuitively, the factors that may affect the transition of the model training include model structure, model parameter scale, model hyper-parameter setting, expert selecting strategy, load balancing loss function, dataset distribution, etc. In this context, we conducted extensive experiments to investigate how these factors influence the changes in expert load states. The experiment results indicate that during the model training, the load of the experts transitions from the transient state to the stable state, where the expert load fluctuates initially but gradually stabilizes, exhibiting locality characteristics. On basis of that, we conduct three classic algorithms for expert load predicting.

#### IV. METHODOLOGY

#### <span id="page-2-0"></span>*A. Preliminary statistic analyses*

TABLE I EXPERIMENT SETUP DESCRIPTION

|                       | Experiment setup 1 | Experiment setup 2 |
|-----------------------|--------------------|--------------------|
| Model                 | GPT-3 Small        | GPT-3 Medium       |
| Parameters            | 125M               | 350M               |
| Layers (MoE)          | 12 (6)             | 24 (12)            |
| Num of experts        | 16 per layer       | 128 per layer      |
| Hidden size           | 768                | 1024               |
| Num of attention head | 12                 | 16                 |
| Global batch size     | 256                | 256                |
| GPU                   | 4*4090             | 4*A800             |

We initially conducted a statistical analysis of the load distribution of experts in each MoE layer during the model training process under two experimental settings, as outlined in Table [I.](#page-2-0) Taking the results from *Experiment 1* as an example, as illustrated in Fig. [1,](#page-1-0) it is evident that the load characteristics of the experts exhibit the following features:

- In the temporal dimension, the load distribution of experts fluctuates in the early training iterations, and gradually exhibits locality, with minor variations in load distribution between adjacent training iterations;
- In the spatial dimension, the first MoE layer (Layer-2) experiences significant fluctuations, while the load distribution of other MoE layers is relatively stable;
- As training iterates, the expert load of each MoE layer will convert from transient to stable, i.e., the expert load stabilized gradually.

To investigate the state of expert load distribution more intuitively, we quantified the magnitude of changes of each expert load in successive iterations, i.e., we calculate the *variance* ( 1 Sizewin P(x<sup>i</sup> −xˆ)) and *range* (xmax −xmin) of each expert load under different sizes of sliding window.

Firstly, in the *Experimental setup 1* scenario, we set the sliding window sizes to 10 and 100, and respectively calculate the *variance* of the load proportion of each expert in sliding windows. The results are shown in Fig. [2.](#page-3-0)

![](_page_3_Figure_0.jpeg)

<span id="page-3-1"></span><span id="page-3-0"></span>Fig. 3. Variance values of experts load proportion of GPT-3 125M (*w=100*)

Then, in this setup with sliding window size of 100, the results of the *variance* in the sliding window of the expert load proportion for all MoE layers are shown in Fig. [3.](#page-3-1) Moreover, we calculated the *range* values of the each expert in the sliding window with the above setup. The results are shown in Fig. [4.](#page-4-0)

Similarly, we calculate the *variance* and *range* values of each expert load proportion in sliding windows for *Experiment Setup 2*. The detail results are shown in the Appendix part.

Both the statistical results of *Experiment Setup 1* and *2* demonstrate that each expert successively experiences the transient state and stable state during the model training, i.e., the load proportions of the experts in each MoE layer fluctuates from the beginning of training and stabilizes gradually with iterations of training. Taking into account the inherent characteristics of expert load and the computational efficiency, LSTM-based, ARIMA-based, and sliding window average-based algorithms were selected for the prediction of expert load in this work. The details of these methodologies will be delineated in the next subsection.

![](_page_4_Figure_0.jpeg)

<span id="page-4-0"></span>Fig. 4. Range values of experts load proportion of GPT-3 125M (w=100)

<span id="page-4-1"></span>![](_page_4_Figure_2.jpeg)

<span id="page-4-3"></span><span id="page-4-2"></span>Fig. 5. Prediction accuracy for GPT-3 125M

#### B. Prediction algorithms

Based on the statistical characteristics of the expert load distribution, we tried Long Short-Term Memory (LSTM)-based, Auto Regressive Integrated Moving Average (ARIMA)-based, and Sliding Window Average (SW\_Avg)-based algorithms to predict expert load in this work. In other words, these algorithms are intended to predict the future load distribution of each expert in the training process in accordance with historical data. Given that the total number of tokens  $N_{token}$  processed in an iteration by each MoE layer is fixed, it suffices to predict the load proportion of each expert within the MoE layers.

- **LSTM-based**: The input of the model is  $[n_1^1, n_2^1, ..., n_1^2, n_2^2, ..., n_e^m]$ , wherein  $n_j^i$  represents the historical data of the load proportion of the j-th expert in the i-th MoE layer, and the output of the model is the load proportion value of all experts in the next k iterations.
- ARIMA-based: The ARIMA model is a classic method for time series analysis and forecasting. The ARIMA model is typically denoted as ARIMA(p,d,q), where p is the order of the auto regressive part, d is the degree of first difference involved, and q is the order of the moving average part. The general form of the ARIMA model can be written as:  $(1 \sum_{i=1}^{p} \phi_i L^i)(1 L)^d X_t = (1 + \sum_{j=1}^{q} \theta_j L^j)\epsilon_t$ , where  $X_t$  represents the time series data,  $\phi_i$  is the parameter of the auto regressive part of the model,  $\theta_j$  is the parameter of the moving average part, L is the lag operator, such that  $L^k X_t = X_{t-k}$ ,  $\epsilon_t$  is the error terms, assumed to be white noise, d is the number of non-seasonal differences needed for stationarity.

The components of the ARIMA model are explained as follows: AUTO REGRESSIVE (AR) part,  $\phi(L) = 1 - \sum_{i=1}^{p} \phi_i L^i$ ; INTEGRATED (I),  $(1-L)^d X_t$ ; MOVING AVERAGE (MA) part,  $\theta(L) = 1 + \sum_{j=1}^{q} \theta_j L^j$ . The goal of the ARIMA model is to find a model that best fits the time series data by minimizing forecast errors, allowing for accurate future predictions.

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

<span id="page-5-2"></span>Fig. 6. Accuracy of LSTM-based prediction for GPT-3 350M

![](_page_5_Figure_2.jpeg)

<span id="page-5-3"></span>Fig. 7. Accuracy of ARIMA-based prediction for GPT-3 350M

In this prediction algorithm, the tests based on expert load proportion history data for stationarity and seasonality are performed to select appropriate p, d, and q parameters.

• SW Avg-based: A straightforward way, taking the arithmetic mean of the data of the load proportion in the historical multiple iterations as the predicted value for the next iteration, and predicting the load of the expert in the future through k rounds of calculation by the means of sliding. This manner exhibits extremely high performance in calculation efficiency, and is also hardware-friendly. Moreover, the experimental results also indicate that this method is simple but effective, and the specific results will be presented in [§V.](#page-5-0)

## V. EVALUATION

#### <span id="page-5-0"></span>*A. Experiment setup 1: GPT-3 125M*

In *Experimental Setup 1*, the prediction algorithms are intended to predict the load proportions of all experts in each MoE layer for the next 1,000 iterations. Compared to the real data, we calculate the mean value of the error ratio for each individual MoE layer prediction result in sliding way.

- LSTM-based prediction. We use the expert load data obtained from two independent training as the training set and test set respectively, and the results are shown in Fig. [5\(a\).](#page-4-1) The transient and stable states of model training can be observed from the changing trend of prediction error, i.e., the prediction accuracy in the transient state (at the early stage of model training) gradually improves with the training iterations, and drops to less than 1% and stays relatively stable after reaching the stable state (after training about 5,000
- iterations indicated by the experimental result). • ARIMA-based prediction. The experimental parameters are ARIMA(5, 1, 5), which achieves a lower error rate than the above LSTM-based algorithm. As shown in the Fig. [5\(b\),](#page-4-2) the prediction results can be close to 0.5% error rate at the stable state for each MoE layer.

![](_page_6_Figure_0.jpeg)

<span id="page-6-0"></span>Fig. 8. Accuracy of SW Avg-based prediction for next 1,000 iterations

<span id="page-6-1"></span>![](_page_6_Figure_2.jpeg)

Fig. 9. Accuracy of SW Avg-based prediction for next 2,000 iterations

• SW Avg-based prediction. Although this method is computationally simple, it performs best among the three algorithms. In the stable state (i.e., after 5,000 iterations training), the algorithm predicts the expert load for each MoE with an error rate of about 0.25%, as demonstrated in the Fig. [5\(c\),](#page-4-3) which can provide more valuable guidance for resource allocation.

#### *B. Experiment setup 2: GPT-3 350M*

- *1) LSTM-based prediction:* Similarly, in *Experimental Setup 2*, the prediction error rate results of the LSTM-based prediction algorithm are shown in Fig. [6\(a\).](#page-5-1) Moreover, to exhibit the guidance value of the prediction method for resource allocation more directly, we give the discrete prediction results with the granularity of per 1,000 iterations instead of the sliding calculating way, as shown in the Fig. [6\(b\),](#page-5-2) which can be observed that the prediction error rates approximate less than 10% at every 1,000 iterations from the second prediction.
- *2) ARIMA-based prediction:* Similarly, we conducted the same experiments and evaluations for the ARIMA-based forecasting method, which demonstrates better performance in terms of prediction accuracy and stability compared to the LSTM-based prediction. Once the model training reaches the stable state, as shown in the Fig. [7\(b\),](#page-5-3) the error rate of this algorithm stabilizes at approximately 1.4%.
- *3) SW Avg-based prediction:* Consistent with *Experimental Setup 1*, SW Avg prediction performs better than the other two algorithms, as exhibited in the Fig. [8,](#page-6-0) whose prediction error rate stabilizes at about 1.3% in the stable state. In addition, we also conduct experiments to evaluate the prediction algorithm with the step size of 2,000 iterations. As presented in the Fig. [9\(a\),](#page-6-1) the accuracy trend of prediction results is consistent with aforementioned results, which stabilizes at about 1.7% in the stable state.

## VI. CONCLUSION

The emergence of MoE makes the computational complexity no longer scale up linearly with the volume of model parameters, which facilitates the development of LLM. The key idea of MoE is to use a gate network to assign tokens to selected experts for processing, which may result in a non-equilibrium load on processing tokens among the experts in each MoE layer during the model training. To this end, in this paper, we analyze the load variation characteristics of experts by conducting extensive experiments and correspondingly define the transient and stable states of model training. Moreover, we deploy three classical prediction algorithms based on the expert load states and achieve high-precision expert load prediction, which can provide valuable guidance on resource allocation for model training.

*In progress*: We are investigating the transient and stable states prediction algorithms based on this paper, and designing sensible and subtle resource allocation schemes to optimize the MoE architecture-based large-scale model training.

#### REFERENCES

- <span id="page-7-0"></span>[1] X. Nie, X. Miao, Z. Wang, Z. Yang, J. Xue, L. Ma, G. Cao, and B. Cui, "Flexmoe: Scaling large-scale sparse pre-trained model training via dynamic device placement," *Proceedings of the ACM on Management of Data*, vol. 1, no. 1, pp. 1–19, 2023.
- <span id="page-7-1"></span>[2] W. Wang, Z. Lai, S. Li, W. Liu, K. Ge, Y. Liu, A. Shen, and D. Li, "Prophet: Fine-grained load balancing for parallel training of large-scale moe models," in *2023 IEEE International Conference on Cluster Computing (CLUSTER)*. IEEE, 2023, pp. 82–94.
- <span id="page-7-2"></span>[3] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, "Adaptive mixtures of local experts," *Neural computation*, vol. 3, no. 1, pp. 79–87, 1991.
- <span id="page-7-3"></span>[4] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, "Outrageously large neural networks: The sparsely-gated mixture-ofexperts layer," *arXiv preprint arXiv:1701.06538*, 2017.
- <span id="page-7-4"></span>[5] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "Gshard: Scaling giant models with conditional computation and automatic sharding," *arXiv preprint arXiv:2006.16668*, 2020.
- <span id="page-7-6"></span>[6] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," *Journal of Machine Learning Research*, vol. 23, no. 120, pp. 1–39, 2022.
- [7] C. Riquelme, J. Puigcerver, B. Mustafa, M. Neumann, R. Jenatton, A. Susano Pinto, D. Keysers, and N. Houlsby, "Scaling vision with sparse mixture of experts," *Advances in Neural Information Processing Systems*, vol. 34, pp. 8583–8595, 2021.
- <span id="page-7-5"></span>[8] B. Mustafa, C. Riquelme, J. Puigcerver, R. Jenatton, and N. Houlsby, "Multimodal contrastive learning with limoe: the language-image mixture of experts," *Advances in Neural Information Processing Systems*, vol. 35, pp. 9564–9576, 2022.
- <span id="page-7-7"></span>[9] Y. Zhou, T. Lei, H. Liu, N. Du, Y. Huang, V. Zhao, A. M. Dai, Q. V. Le, J. Laudon *et al.*, "Mixture-of-experts with expert choice routing," *Advances in Neural Information Processing Systems*, vol. 35, pp. 7103–7114, 2022.
- <span id="page-7-8"></span>[10] S. Roller, S. Sukhbaatar, J. Weston *et al.*, "Hash layers for large sparse models," *Advances in Neural Information Processing Systems*, vol. 34, pp. 17 555–17 566, 2021.
- <span id="page-7-9"></span>[11] Y. Zhou, N. Du, Y. Huang, D. Peng, C. Lan, D. Huang, S. Shakeri, D. So, A. M. Dai, Y. Lu *et al.*, "Brainformers: Trading simplicity for efficiency," in *International Conference on Machine Learning*. PMLR, 2023, pp. 42 531–42 542.

#### APPENDIX

We analyzed the expert loads of the GPT3 350M model during training, and the *variance* and *range* of the each expert load proportion in each MoE layer at a sliding window of 100 are shown in Fig. [10](#page-8-0) and Fig. [11.](#page-9-0) In general, the variation of expert load in Experimental Setup 2 also satisfies the defined transient and stable states.

<span id="page-8-0"></span>![](_page_8_Figure_2.jpeg)

![](_page_9_Figure_0.jpeg)

<span id="page-9-0"></span>Fig. 11. Range values of experts load proportion of GPT-3 350M (*w=200*)