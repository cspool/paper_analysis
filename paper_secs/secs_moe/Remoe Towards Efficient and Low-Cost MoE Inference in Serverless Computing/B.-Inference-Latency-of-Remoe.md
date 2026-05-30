# B. Inference Latency of Remoe

Since pre-processing and post-processing only involve fixed components and their overhead is typically negligible, we omit these stages and focus on the **Prefilling** and **Decoding** phases.

1) **Prefilling**: The total prefilling time can be presented as:

$$PT = \sum_{l=1}^{L} (PT_{l}^{f} + PT_{l}^{e})$$
 (1)

where  $PT_l^f = \tau_l^f(N^{in})$  is the prefilling time of non-expert module  $\mathcal{F}_l$ . Here,  $\tau_l^f(n)$  is the time for  $\mathcal{F}_l$  to process n tokens and  $N^{in}$  is the number of input tokens.  $PT_l^e$  is the prefilling time of the expert module  $\mathcal{E}_l$ , which can be expressed as:

$$\begin{split} PT_{l}^{e} &= \max[\sum_{k=1}^{K_{l}}(1-x_{l,k})PT_{l,k}^{loc}, \max_{j\leqslant z_{l}}\{ZT_{l,j}\}] + 2\tau^{sw}(N^{in}) \ \ (2) \\ \text{where} \sum_{k=1}^{K_{l}}(1-x_{l,k})PT_{l,k}^{loc} \ \text{and} \ \max_{j\leqslant z_{l}}\{ZT_{l,j}\} \ \text{are the end-} \end{split}$$

where  $\sum_{k=1}^{K_l} (1-x_{l,k}) PT_{l,k}^{loc}$  and  $\max_{j \leqslant z_l} \{ZT_{l,j}\}$  are the end-to-end latency of local experts and remote experts, respectively. We will describe these two parts later. Since expert modules are deployed on CPU, the data need to transfer between GPUs and CPUs twice.  $\tau^{sw}(N^{in})$  is used to denote the migration time of  $N^{in}$  tokens.

the migration time of  $N^{in}$  tokens. **Local Experts Latency**.  $PT_{l,k}^{loc} = \sum_{v=1}^{V} w_v \tau_{l,k,v}^c(N_{l,k}^{pre})$  denotes the prefilling time when  $e_{l,k}$  is local, where  $w_v$  is the memory specifications allocated to the main model's container, and  $\tau_{l,k,v}^c(N_{l,k}^{pre})$  represents the computation time for expert  $e_{l,k}$  to process  $N_{l,k}^{pre}$  tokens under memory specification v. The term  $N_{l,k}^{pre}$  is the total number of tokens routed to  $e_{l,k}$  during prefilling, calculated as the sum  $N_{l,k}^{pre} = \sum_{i=1}^{N^{in}} s_{l,k,i}$ .  $s_{l,k,i} = 1$  indicates the i-th input token is processed by  $e_{l,k}$ .

**Remote Experts Latency**. With  $x_{l,k}$ , the remote expert set can be denoted as  $\mathcal{R}_l = \{e_{l,k}|x_{l,k} = 1\}$ . Since we utilize function replicas to accelerate the remote expert inference, we split the remote expert set  $\mathcal{R}_l$  into  $\mathcal{R}_{l,1}, \mathcal{R}_{l,2}, \dots, \mathcal{R}_{l,z_l}$  and

each replica undertakes the computation of one subset. Different replicas execute simultaneously, so the end-to-end latency of the remote experts is  $\max_{j \leq z_l} \{ZT_{l,j}\}$ .  $ZT_{l,j}$  represents the

<span id="page-3-10"></span>

latency for the j-th replica. It is calculated as:  $ZT_{l,j} = \sum_{\substack{e_{l,k} \in \mathcal{R}_{l,j} \\ v \in \mathcal{N}_{l,k}}} (PT_{l,k}^{rem} + 2N_{l,k}^{pre}D/B) + t_l^{rem} \qquad (3)$  where  $PT_{l,k}^{rem} = \sum_{v=1}^{V-1} y_{l,v} \tau_{l,k,v}^c(N_{l,k}^{pre})$  is the prefilling time when  $e_{l,k}$  is remote.  $V^e$  is the total number of memory specifications for remote experts ( $V^e < V$ ). D is the size of a single token embedding and B is the network transfer rate. The term  $t_1^{rem}$  denotes the additional overhead introduced by the serverless invocation for remote experts of layer l (under warm-start conditions), which is a random variable dependent on the vCPU scheduling policy and resource contention.

2) **Decoding**: After the first token is generated, the model enters the Decoding stage. Let the total number of generated tokens be  $N^{out} + 1$  (including the first token). Decoding time can be expressed as:  $N^{in}+N$ 

where 
$$t_l^f$$
 is the single token's decoding time of  $\mathcal{F}_l$ .  $GT_{l,i}^e$  (4)

the decoding time of  $\mathcal{E}_l$  for token i; it can be calculated as:

<span id="page-3-8"></span>
$$GT_{l,i}^{e} = 2\tau^{sw}(N^{topk}) + \max\left[\sum_{k=1}^{K_{l}} (1 - x_{l,k}) s_{l,k,i} GT_{l,k}^{loc}, \sum_{k=1}^{K_{l}} x_{l,k} s_{l,k,i} (GT_{l,k}^{rem} + 2D/B + t_{l}^{rem})\right]$$
(5)

where  $N^{topk}$  is the number of experts each token is routed to.  $GT_{l,k}^{loc}$  and  $GT_{l,k}^{rem}$  are the decoding times of  $e_{l,k}$  when it is local and remote, respectively. The former is denoted as  $GT_{l,k}^{loc} = \sum_{v=1}^{V} w_v t_{l,k,v}^c$  where  $t_{l,k,v}^c$  is the time for expert  $e_{l,k}$  to process a single token under memory specification v.

Similarly,  $GT_{l,k}^{rem}$  is calculated as  $GT_{l,k}^{rem} = \sum_{v=1}^{V^e} y_{l,v} t_{l,k,v}^c$ .

3) **TTFT and TPOT**: For LLMs, SLOs are typically measured by Time-to-First-Token (TTFT) and Time-per-Output-Token (TPOT). In our model,  $T^{cold}$  is the cold start time, and TTFT can be expressed as  $T^{ttft} = PT + T^{cold}$ . Besides, TPOT is denoted as  $T^{tpot} = GT/N^{out}$ .

