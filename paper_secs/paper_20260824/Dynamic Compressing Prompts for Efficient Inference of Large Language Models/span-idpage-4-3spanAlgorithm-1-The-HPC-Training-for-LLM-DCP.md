# <span id="page-4-3"></span>Algorithm 1 The HPC Training for LLM-DCP

**Input:** The prompt for compression dataset  $\mathcal{D}$ , the DCP-Agent  $\pi_{\theta}$ , the critic  $V_{\phi}$  the reply buffer  $\mathcal{B}$ , the maximum trajectory number of the buffer M, the iteration number of training m, the number of curriculum learning stages P and the coefficients  $c_s$  and  $c_l$ . 1: Initialize buffer  $\mathcal{B}$ , actor parameters  $\theta$  and critic parameters  $\phi$ .

```
while Not convergence do
 3:
         for P_i in P do
 4:
             Calculate c_s and c_l via Eq.(8).
 5:
             for x_i in \mathcal{D} do
                 Collect a trajectory \tau = \{s_t, a_t, r_t, v_t, A^{\pi_{\theta_{old}}}(s_t, a_t)\}
 6:
                with old \pi_{\theta_{old}} and V_{\phi_{old}}. Put \tau into {\mathcal B}.
 7:
                 if length(\mathcal{B}) == M then
 8:
 9.
                    for iteration = 1, 2, ..., M do
10:
                        Uniformly sample \tau \in \mathcal{B}.
11:
                        Calculate \mathcal{J}(\theta) via Eq.(7).
                        Update \theta to maximize \mathcal{J}(\theta).
12:
13:
                        Calculate TD error via Eq. (9).
                        Update \phi to minimize TD error \delta_t.
14:
15:
                    end for
                    Empty the replay buffer \mathcal{B}.
16:
                    Update \theta_{old} \leftarrow \theta.
17:
                    Update \phi_{old} \leftarrow \phi.
18:
19:
20:
             end for
21:
         end for
22: end while
```

<span id="page-4-0"></span>send the features to a linear classification layer. Specifically, at time step t, the state  $s_t = \widetilde{x}_{t-1} = \{x_i\}_{i=1}^{\widetilde{L}_{t-1}}$  contains  $\widetilde{L}_{t-1}$  tokens, which can be formalized as:

$$\boldsymbol{h} = f_{\theta}(\widetilde{x}_{t-1}),\tag{5}$$

$$p(x_i, \theta) = \operatorname{softmax}(Wh_i + b),$$
 (6)

where  $\mathbf{h} = \{h_i\}_{i=1}^{L_{t-1}}$  is feature vectors for all tokens,  $p(x_i, \theta) \in \mathbb{R}^2$  denotes the probability distribution of label  $\{0, 1\}$  for the i-th token  $x_i$ . Here we use xlm-roberta-large [57] as Transformer encoder  $f_{\theta}$ . In the off-policy algorithm, the old policy  $\pi_{\theta_{old}}$  with old parameters  $\theta_{old}$  is used to collect trajectories with the environment, while the policy  $\pi_{\theta}$  is updated using trajectories collected by  $\pi_{\theta_{old}}$ .

**Critic.** The critic  $V_{\phi}(s)$  is used to estimate the expected return of the state  $s_t$  and calculate the advantage, which can aid the actor in learning more efficiently and stably. Similar to the actor, the critic is composed of a pre-trained *xlm-robertalarge* [57] as an encoder, and with two Linear layers. Besides, the old critic  $V_{\phi_{old}}(s)$  is used to collect trajectories, and the new critic  $V_{\phi_{new}}(s)$  is updated using the collected trajectories.

**Learning Objectives.** The goal of the learning is to maximize the expected long-term return  $\mathcal{J}(\theta)$ :

<span id="page-4-2"></span>
$$\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}(\tau)}[G(\tau)]$$

$$= \mathbb{E}_{\tau \sim \pi_{\theta_{old}}(\tau)}[\min(\delta A^{\pi_{\theta_{old}}}(s_t, a_t), \operatorname{clip}(\delta, 1 - \epsilon, 1 + \epsilon) A^{\pi_{\theta_{old}}}(s_t, a_t))], \tag{7}$$

where  $G(\tau)$  is the total return of the trajectory  $\tau = \{s_t, a_t, r_t, v_t, A^{\pi_{\theta_{old}}}(s_t, a_t)\}$  obtained by  $\pi_{\theta_{old}}$  and  $V_{\phi_{old}}$ ,  $\delta = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$  is the ratio of the probability of action  $a_t$  given by  $\pi_{\theta}$  and  $\pi_{\theta_{old}}$  for state  $s_t$ , and  $\epsilon$  is a hyperparameter, we

<span id="page-5-5"></span>Method Pub.'Year BLEU ↑ BLEURT ↑ Rouge-1 ↑ Rouge-2 ↑ Rouge-L ↑ BS F1 ↑ Tokens 1  $1/\rho \uparrow$ ShareGPT Selective-Context [20] EMNLP'2023 38.35 3.3x 38.53 -0.2151.27 43.51 78.30 183 LLMLingua[18] EMNLP'2023 38.71 -0.2151.43 38.62 43.57 78.27 186 3.2x LLMLingua-2-small [12] ACL'2024 56.79 0.37 76.09 58.47 63.56 89.54 191 3.1x LLMLingua-2 [12] ACL'2024 61.97 0.47 78.64 63.07 67.50 90.87 184 3.3xLLM-DCP (Ours) 64.93 0.54 80.24 65.54 69.89 91.80 175 3.4x Arxiv-March23 933 11.8x Selective-Context [20] EMNLP'2023 8.83 -0.6143.43 13.46 18.92 73.75 LLMLingua[18] EMNLP'2023 5.70 -0.7432.29 8.78 15.17 69.60 1276 8.7xLLMLingua-2-small [12] -0.451017 10.9x ACL'2024 8.56 45.52 15.47 21.09 75.49 LLMLingua-2 [12] ACL'2024 10.84 -0.5748.49 14.62 19.95 75.15 920 12.0x LLM-DCP (Ours) -0.55 48.81 15.94 75.91 855 12.9x 10.10 21.63

TABLE I
PERFORMANCE OF DIFFERENT METHODS ON THE CONVERSATION (SHAREGPT) AND SUMMARIZATION (ARXIV-MARCH23) TASKS.

set to 0.15 in this paper. The operation  $\operatorname{clip}(\delta, 1 - \epsilon, 1 + \epsilon)$  constrains  $\delta$  to the range  $[1 - \epsilon, 1 + \epsilon]$ , and  $A^{\pi_{\theta_{old}}}(s_t, a_t) = r_t - V_{\phi_{old}}(s_t)$  is the advantage at t.

**HPC Training.** The overview of the optimization process of the HPC training strategy is presented in Algorithm 1. Specifically, given a prompt for compression dataset  $\mathcal{D}$ , we use  $\pi_{\theta_{old}}$  and  $V_{\phi_{old}}(s)$  to interact with the environment to collect the trajectory  $\tau = \{s_t, a_t, r_t, v_t\}$ , and compute the advantage  $A^{\pi_{\theta_{old}}}(s_t, a_t)$ . During the collection of trajectories, the HPC training strategy increases the compression difficulty and guides the learning of the DCP-Agent incrementally by gradually decreasing the compression rate range  $[c_s, c_l]$  (see Eq. 4) and the maximum trajectory length  $T_{max}$  in stage  $(P_i)$ . The  $c_s$  and  $c_l$  are adjusted as follows:

<span id="page-5-2"></span>
$$\begin{cases} c_s = 0.6 - (P_i + \frac{t}{T_{max}})\psi \\ c_l = 1.0 - (P_i + \frac{t}{T_{max}})\psi \end{cases}, \tag{8}$$

where  $\psi$  set to 0.1,  $P_i$  denotes the  $i^{th}$  stage, with i starting at 1 and  $P_1=1$ . Notably the learning stage size is set to 3, and  $T_{max}=2$  except for the third stage where  $T_{max}=1$ . This easy to difficult curriculum learning strategy effectively improves the performance of prompt compression. We then put  $\tau$  into the reply buffer  $\mathcal{B}$ . When a certain number of trajectories (such as M) have been collected, they are used to train the actor and critic. In particular, we begin by uniformly sampling sequences from the replay buffer  $\mathcal{B}$ , then compute the expected long-term return  $\mathcal{J}(\theta)$  to optimize the policy parameters  $\pi_{\theta}$ . Additionally, the Temporal Difference (TD) error  $\delta_t$  is calculated to refine the critic's parameters  $V_{\phi}$ :

<span id="page-5-3"></span>
$$\delta_t = G_t - V_{\phi}(s_t), \tag{9}$$

where  $G_t$  represents the total expected return starting from time step t. After conducting a certain number of training iterations using the samples from the existing replay buffer  $\mathcal{B}$ , we clear the buffer and update the parameters of the old policy  $\pi_{\theta_{old}}$  and critic  $V_{\phi_{old}}$ . This process is then repeated until convergence is achieved.

