# <span id="page-7-0"></span>3.5 Fine-Grained Scaling under SLOs

We design a fine-grained, SLO-aware resource scaling scheme that jointly selects attention-side and MoE-side resources. Let  $n_a$  and  $n_e$  denote the numbers of active attention and MoE instances, respectively, where each instance runs on one GPU. Given a workload demand  $\lambda$  and a TPOT SLO, JANUS searches for a configuration  $(n_a, n_e)$  that can sustain  $\lambda$  while keeping the predicted TPOT within the SLO. In disaggregated MoE inference, scaling becomes a two-dimensional optimization problem rather than instance-level scaling of the full model. Among all SLO-feasible configurations, JANUS chooses the one with the smallest GPU count  $n_a + n_e$ , which equivalently maximizes the throughput per GPU.

As demand changes, JANUS re-runs this optimization and applies the new configuration incrementally.

**Performance model.** For a candidate configuration  $(n_a, n_e)$ , JANUS estimates TPOT with a layer-wise latency model. On the attention side, requests are evenly dispatched across the  $n_a$  data-parallel attention instances. Let B denote the in-flight decode batch size,  $b = B/n_a$  denote the per-instance local batch size,  $S_{\rm ctx}$  denote the average context length, and L denote the number of layers. Following prior LLM serving systems [1,22,42], JANUS models TPOT as the sum of attention, MoE, and communication costs across layers:

TPOT = 
$$\sum_{\ell=1}^{L} \left[ T_{\text{attn}}^{(\ell)} + T_{\text{moe}}^{(\ell)} + T_{\text{comm}}^{(\ell)} \right],$$
 (1a)

$$T_{\text{attn}}^{(\ell)} = \max\left(c_a^{(\ell)}, \ \alpha^{(\ell)}b + c_{kv}^{(\ell)}bS_{\text{ctx}}\right), \tag{1b}$$

$$T_{\text{moe}}^{(\ell)} = \beta^{(\ell)} \cdot a_{\text{max}}^{(\ell)}(n_e, B) + c_e^{(\ell)}.$$
 (1c)

Here  $T_{\rm attn}^{(\ell)}$ ,  $T_{\rm moe}^{(\ell)}$ , and  $T_{\rm comm}^{(\ell)}$  denote the attention, MoE, and communication latencies of layer  $\ell$ , respectively. The attention term  $T_{\rm attn}^{(\ell)}$  follows the roofline model [34]:  $c_a^{(\ell)}$  captures the memory-bound latency plateau that dominates at small workloads, while  $\alpha^{(\ell)}b+c_{kv}^{(\ell)}bS_{\rm ctx}$  captures the cost of computation and KV-cache access. The MoE term  $T_{\rm moe}^{(\ell)}$  follows the linear dependence on  $a_{\rm max}^{(\ell)}(n_e,B)$ , the maximum number of distinct activated experts across MoE instances under the candidate MoE size and AEBS strategy. The communication term  $T_{\rm comm}^{(\ell)}$  is obtained from the profiled cost model of the adaptive two-phase communication scheme (§3.3). All hardware-dependent coefficients, including  $\alpha^{(\ell)}$ ,  $\beta^{(\ell)}$ ,  $c_a^{(\ell)}$ ,  $c_{kv}^{(\ell)}$ ,  $c_e^{(\ell)}$ , are obtained through a one-time offline profiling.

**Problem formulation.** The in-flight batch size is not an independent decision variable. Under the steady-state decode serving, it is determined by Little's Law [11]:

<span id="page-7-2"></span>
$$B^* = \lambda \cdot \text{TPOT}(B^*, n_a, n_e, S_{\text{ctx}}). \tag{2}$$

Thus, changing the resource configuration  $(n_a, n_e)$  changes the TPOT curve and also the steady-state batch size  $B^*$ .

Each candidate configuration must satisfy the per-GPU memory constraints. Let M be the memory budget of a GPU, and let  $b^* = B^*/n_a$  denote the steady-state local batch size on each attention instance. We use  $\mathcal{M}_a(b^*, S_{\text{ctx}})$  to denote the memory usage of an attention instance, including the attention weights, KV cache, and activation buffers. On the MoE side, memory usage is dominated by the pinned expert weights: each MoE instance pins at most C expert replicas, which makes the per-GPU memory constraint easy to enforce during placement. The resource scaling problem is formulated as:

<span id="page-7-1"></span>
$$\min_{n_a, n_e, B^*} \quad n_a + n_e 
s.t. \quad \text{TPOT}(B^*, n_a, n_e, S_{\text{ctx}}) \leq \text{SLO}, 
\mathcal{M}_a(b^*, S_{\text{ctx}}) \leq M, 
n_e \cdot C \geq E, 
n_a, n_e \in \mathbb{Z}^+.$$
(3)

The first constraint enforces the TPOT SLO, and the second constraint enforces the attention-side memory feasibility. The third constraint ensures that the MoE sub-cluster has enough expert slots to host all expert replicas.

**Scaling solution.** Solving Eq. (3) has two challenges. First, the TPOT model depends on the maximum number of distinct activated experts  $a_{\max}^{(\ell)}(n_e, B)$ , which is workload- and scheduling-dependent and thus difficult to capture with a static closed-form model. Second, for each candidate resource configuration  $(n_a, n_e)$ , the steady-state batch size  $B^*$  is unknown in advance; it must be solved from the fixed-point equation in Eq. (2) before checking SLO and memory feasibility.

JANUS uses recent activation statistics to build a Monte Carlo estimator  $\widehat{a}_{\max}^{(\ell)}(n_e,B)$  of  $a_{\max}^{(\ell)}(n_e,B)$ . We formulate the top-K routing as a balls-into-bins problem [25] and derive a theoretical upper bound on  $a_{\max}$  in Appendix A (Eq. 5). Building on the analysis, JANUS uses a Monte Carlo approach for the  $a_{\max}$  estimation and scaling decisions. For each candidate  $(n_e,B)$  and each MoE layer  $\ell$ , it samples B tokens from the recent activation trace, applies the current scheduling strategy, and records the resulting estimate  $\widehat{a}_{\max}^{(\ell)}(n_e,B)$ . The resulting lookup table  $\widehat{a}_{\max}^{(\ell)}(n_e,B)$  is rebuilt periodically, ensuring that the model is aligned with the current workload.

To solve Eq. (2), JANUS performs a bounded one-dimensional search for the steady-state batch size  $B^*$  over  $[1, B_{\text{max}}]$ , where  $B_{\text{max}}$  is the maximum batch size allowed by the GPU memory budget. For a fixed configuration  $(n_a, n_e)$ , we define the residual  $f(B) = B - \lambda \cdot \text{TPOT}(B, n_a, n_e, S_{\text{ctx}})$ . In our profiled operating range, the residual is monotonic and thus JANUS solves it with a bounded binary search [2, 23]. JANUS handles two boundary cases explicitly. If  $f(1) \ge 0$ , the workload is too light to form a larger steady-state batch, so JANUS sets  $B^* = 1$ . If  $f(B_{\text{max}}) < 0$ , even the largest memory-

