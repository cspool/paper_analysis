# 4 RELATED WORK

This section presents a review of related work, framed within the lens of System-1 and System-2 inference. Building on the previously introduced notation, we begin with a straightforward definition of System-1 and System-2 inference.

<span id="page-8-0"></span>

| Training Stages      | P             | re-Training                                | Fine-Tuning                                                                |                 |  |  |
|----------------------|---------------|--------------------------------------------|----------------------------------------------------------------------------|-----------------|--|--|
|                      | Training data | Learning method                            | Training data                                                              | Learning method |  |  |
| System-1<br>System-2 | (Q) $(Q, A)$  | Self-supervised learning<br>RL + Self-Play | $\begin{array}{ c c }\hline (Q,A) \\ (Q,\ldots,S^j,\ldots,A)^4\end{array}$ | SFT<br>RFT      |  |  |

Table 3: System-1 v.s. System-2: relied training data and used learning method in the pre-training and fine-tuning stages

- System-1 inference: This involves directly inferring the answer  $A_i$  from the question  $Q_i$ , represented as  $p(A_i|Q_i)$ . It operates in a straightforward, single-step manner.
- System-2 inference: System 2 inference, i.e., reasoning, involves multiple intermediate inference steps before deriving an answer. Specifically, it first infers the reasoning steps  $\{S_i^1, \ldots, S_i^j, \ldots, S_i^m\}$  from the question  $Q_i$ , and then infer the final answer  $A_i$ . Formally, this can be expressed as:

$$p(A_i|Q_i) = \sum_{S_i^j} p(\{S_i^1, \dots, S_i^j, \dots, S_i^m\}|Q_i) \cdot p(A_i|Q_i, \{S_i^1, \dots, S_i^j, \dots, S_i^m\})$$

