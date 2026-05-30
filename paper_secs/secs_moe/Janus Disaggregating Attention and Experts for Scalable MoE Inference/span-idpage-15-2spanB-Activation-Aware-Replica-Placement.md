# <span id="page-15-2"></span>**B** Activation-Aware Replica Placement

This appendix provides the details of JANUS's expert placement in §3.5.

**Replica count.** Given  $n_e$  MoE instances each with C expert slots, the  $S=n_e\cdot C$  total slots first seat one replica of each of the E logical experts; the remaining S-E slots provide redundancy. Janus assigns these redundant slots iteratively: using activation counts c(e) over a sliding window, it repeatedly picks the expert with the largest per-replica load l(e)=c(e)/R(e) and grants it one more replica, until all S-E extra slots are exhausted. Hot experts accumulate more replicas, cold experts remain singleton, and per-replica activation pressure is equalized.

**Placement optimization.** Given replica assignments  $\{R(e)\}_{e=1}^{E}$  and per-instance capacity C, let  $x_{e,g} \in \{0,1\}$  indicate whether a replica of logical expert e is placed on instance g, and let a(e,e') denote the co-activation frequency between logical experts e and e' estimated from recent traces. We define the co-activation load on instance g as:

$$I(g) = \sum_{\substack{e, e' \in P(g) \\ g \in g'}} a(e, e')$$
 (6)

Colocating experts with high a(e,e') raises concurrent activations on that instance and thus MoE latency. JANUS solves the min–max assignment:

<span id="page-16-0"></span>
$$\min_{\{x_{e,g}\}} \max_{g \in \{1, \dots, n_e\}} I(g)$$
s.t. 
$$\sum_{e=1}^{E} x_{e,g} \le C,$$

$$\sum_{g=1}^{n_e} x_{e,g} = R(e),$$

$$x_{e,g} \in \{0, 1\}.$$
(7)

Eq. (7) reduces to unrelated-machines scheduling and is NP-hard [9]. JANUS uses the greedy heuristic in Algorithm 3. It first initializes per-instance placement sets, remaining slots, and a bitmap recording whether an instance already hosts a replica of a given logical expert (lines 1–3). It then iterates over replicas in descending order of load: if there exists an instance with free capacity that does not yet host that expert, the replica is placed on the instance that adds the least co-activation penalty (lines 5–10). Otherwise, a bounded swap between two instances is performed to create a feasible placement with minimal incremental co-activation cost (lines 11–18). This heuristic closely approximates the minmax objective while remaining efficient enough for periodic online reconfiguration.

