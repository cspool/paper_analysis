# <span id="page-16-1"></span>C Optimality Proof for Algorithm 1

*Proof.* The proof is divided into two main parts:

- If Algorithm 1 returns INVALID, no feasible solution exists.
- 2. If a feasible solution exists, the solution returned by Algorithm 1 is optimal.

Preliminaries and Notation:

- For each request  $r_i$ , we have a token tree  $T_{inf}(r_i)$ .
- Each node v in  $T_{inf}(r_i)$  is associated with a path probability f(v).
- The goal for each request  $r_i$  is to achieve a target path probability  $A(r_i)$  (the SLO).
- We have a total budget B, which is the maximum number of tokens (nodes) that can be selected across all requests.
- We define  $N_i$  as the minimal number of tokens needed to be selected from  $T_{inf}(r_i)$  to achieve  $A(r_i)$ .

<span id="page-16-2"></span>**Lemma C.1** (Minimality in Threshold Attainment). Given a token tree and a threshold  $\tau$ , consider a greedy algorithm that repeatedly selects the node with the highest f(v) not yet chosen, until the sum of f(v) of the chosen nodes meets or exceeds  $\tau$ . Suppose this process stops after selecting n nodes. Then there is no subset of fewer than n nodes from the tree whose sum of f(v) is at least  $\tau$ .

Proof of Lemma C.1: By construction, after selecting n-1 nodes, the greedy algorithm did not meet the threshold  $\tau$ . Therefore, any subset of size less than n cannot meet or exceed  $\tau$ , since the greedy set of n-1 nodes is by definition a best possible subset of that size in terms of cumulative f(v) (no other subset of n-1 nodes can have a greater sum than the greedily chosen n-1). Thus, n is the minimal number of nodes required to surpass the threshold.

Part 1: If Algorithm 1 returns INVALID, no feasible solution exists. Consider running Algorithm 1. For each request  $r_i$ :

- 1. The algorithm attempts to meet  $A(r_i)$  by repeatedly choosing the highest f(v) node from  $T_{inf}(r_i)$  not yet chosen by any request, until  $A(r_i)$  is reached or the budget B is exhausted.
- 2. If at some step i, the algorithm cannot find enough tokens to achieve  $A(r_i)$  (i.e., it runs out of budget before  $A(r_i)$  is met), it returns INVALID.

By Lemma C.1, the minimal tokens needed to achieve  $A(r_i)$  is  $N_i$ . If the algorithm fails at request i, it means it has already allocated tokens to previous requests  $r_1, \ldots, r_{i-1}$  optimally (since it picks the highest probability nodes first). Thus, by the time it considers  $r_i$ , it has spent at least  $N_1 + N_2 + \cdots + N_{i-1}$  tokens. If it cannot fulfill  $A(r_i)$ , it implies  $N_1 + \cdots + N_i > B$ . Therefore, there is no way to allocate B

tokens to meet all  $A(r_1), \ldots, A(r_i)$  simultaneously. Since this reasoning applies for the request where the algorithm fails, if Algorithm 1 returns INVALID, no feasible solution exists.

**Part 2:** If a feasible solution exists, the returned solution is optimal. Now suppose Algorithm 1 completes successfully. It produces a solution S that satisfies  $A(r_i)$  for all i within the budget B. We need to show that if there is any other feasible solution S' that also meets all SLOs, then S is at least as good as S' (i.e., S is optimal).

To prove this, we rely on another lemma about the greedy selection of nodes under a fixed budget.

<span id="page-17-0"></span>**Lemma C.2** (Maximality Under a Fixed Budget). Given a token tree and a budget b, let a greedy algorithm select the top b nodes in terms of f(v) from that tree. This selection maximizes the sum of f(v) over all subsets of size b.

Proof of Lemma C.2: Suppose for contradiction that there is a subset V' of size b whose total sum of f(v) is greater than that of the subset V chosen by the greedy algorithm. Since the greedy algorithm picks the top b nodes, every node in  $V \setminus V'$  must have f(v) greater than or equal to that of any node in  $V' \setminus V$ . By swapping the lower-probability nodes in V' with the higher-probability nodes from V, we form a new subset that has a sum at least as large as V'. But this new subset is precisely V, contradicting the assumption that V' has a strictly greater sum. Thus, V is optimal.

Establishing optimality of the returned solution *S*:

- 1. Define  $N_i$  as the minimal tokens required to achieve  $A(r_i)$  for each request  $r_i$ . Note that  $M_i(S) \ge N_i$  for the solution S returned by the algorithm, where  $M_i(S)$  is the number of tokens allocated to  $r_i$  in S. The same holds for any other feasible solution  $S': M_i(S') \ge N_i$ .
- 2. Suppose there exists a valid solution S' that is better than S. Being "better" might mean it uses fewer tokens or achieves a higher sum of f(v) for the given budget. Consider how S' distributes tokens among requests: there must be some difference in the number of tokens allocated to at least one request, otherwise they are identical solutions.
- 3. Fix a particular distribution of the budget across the requests. For any single token tree  $T_{inf}(r_i)$  and a token count  $M_i$ , by Lemma C.2, the greedy choice of  $M_i$  nodes yields the maximum possible sum of f(v) for that budget on  $r_i$ . Thus, if S' differs from S, but assigns the same number of tokens  $M_i(S')$  to request  $r_i$  as S does, then to improve upon S's solution, S' must choose nodes with a strictly greater total sum of f(v) than S under the same budget  $M_i(S)$ . This is impossible due to Lemma C.2, since S is constructed by a greedy procedure.
- 4. Hence, any improvement in one request's allocation in *S'* would require changing the budget distribution

among requests. However, after ensuring the minimal quotas  $N_i$  for each request (which both S and any feasible S' must respect), the second step of the algorithm in S distributes the remaining tokens globally in a greedy manner. This global greedy step ensures that no other distribution of these "extra" tokens can yield a strictly better sum, since that would contradict the global maximality of the greedy choice.

In other words, if S' tries to reallocate tokens among requests (while still meeting all SLOs), any purported improvement can be dismantled by applying Lemma C.2 within each token tree. Ultimately, this shows that no S' better than S can exist.

#### Conclusion:

- 1. If Algorithm 1 returns INVALID, no feasible solution can exist, since the minimal required tokens to meet the SLOs of the first *i* requests already exceed *B*.
- 2. If a feasible solution exists, the solution returned by Algorithm 1 must be optimal. Any other solution that meets all SLOs cannot be strictly better, due to the maximality properties of the greedy selections both per-request and globally.

Thus, Algorithm 1 is correct and optimal.

