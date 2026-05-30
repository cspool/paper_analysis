# <span id="page-8-1"></span>Algorithm 2 Fine-Grained, SLO-Aware Resource Scaling Input:

```
-n_{\text{max}}: upper bound of instance sizes
-n_a^{\text{min}}: lower bound of MoE instance sizes, i.e., [E/C]
-B_{\text{max}}: upper bound of batch sizes according to GPU memory budget
Output:
-(n_a^*, n_e^*, B^*): optimal configuration if feasible
  1: opt \leftarrow \bot; J^* \leftarrow \infty
  2: for (n_a, n_e) \in \{1, ..., n_{\text{max}}\} \times \{n_e^{\text{min}}, ..., n_{\text{max}}\} do
           B^* \leftarrow \text{batch size in } [1, B_{\text{max}}] \text{ satisfying Eq. (2)}
  3:
           if B^* = \bot then
  4:
                 continue
  5:
           T \leftarrow \text{TPOT}(B^*, n_a, n_e, S_{\text{ctx}})
  6:
           if T > \text{SLO or } \neg \text{MEMORYFEASIBLE}(B^*, n_a, n_e) then
  7:
                 continue
  8:
           if n_a + n_e < J^* then
  9:
                 opt \leftarrow (n_a, n_e, B^*); J^* \leftarrow n_a + n_e
 10:
 11: return opt
```

feasible batch cannot sustain the demand, so JANUS discards the current candidate configuration  $(n_a, n_e)$ .

JANUS then solves Eq. (3) by enumerating the candidate configurations over a bounded search space. Configurations that are clearly infeasible, such as  $n_e < n_e^{\min}$ , are pruned before evaluation. Algorithm 2 gives this scaling procedure. For each remaining candidate configuration  $(n_a, n_e)$ , JANUS first solves Eq. (2) to obtain  $B^*$  (line 3), evaluates TPOT, checks memory feasibility, and selects the feasible configuration with the smallest GPU count (lines 6–10). This computation incurs negligible runtime overhead: each TPOT evaluation only requires the constant-time lookups of  $\widehat{a}_{\max}^{(\ell)}$ , and the search space over  $(n_a, n_e)$  is bounded by the cluster size. The selected configuration  $(n_a^*, n_e^*)$  is then applied incrementally by adding or removing attention and MoE instances.

Expert placement at the MoE side. After determining the optimal resource configuration, JANUS allocates and places expert replicas to support the activated-expert-balanced scheduling. The key goal is to avoid collocating experts that are frequently activated together, which increases the number of distinct activated experts on the same instance and increases MoE latency. Accordingly, JANUS processes replicas in descending load order and places each replica on the instance that incurs the smallest additional co-activation pressure while respecting the per-instance capacity constraints. We provide the formal optimization and full algorithm in Appendix B.

#### 4 Implementation

We implement JANUS on top of SGLang [26] with about 4K lines of Python and 300 lines of CUDA/C++ code, extending SGLang to support disaggregated MoE inference. On the attention side, JANUS reuses SGLang's request batching, dispatching, and KV-cache management. For cross-sub-cluster

communication, JANUS implements the adaptive two-phase mechanism with NVSHMEM [20] and GPUDirect RDMA, while intra-node collectives over NVLink are implemented using NCCL. Specifically, JANUS uses NVSHMEM's onesided putmem\_signal/signal\_wait primitives to directly write payloads into receiver GPU memory and signal completion. We pack lightweight metadata, including layer index and token count, into the same signal value to avoid separate metadata transfers; CPU-side metadata unpacking is performed only at the first MoE layer and then reused for subsequent layers. We also tune NVSHMEM parameters, including IBGDA transport, request-batching threshold, and per-peer RC queue count, for our communication pattern. We place the shared expert on the attention side and execute it while each attention instance transfers intermediate data to the MoE side and waits for the results, thereby overlapping communication with computation. On the MoE side, each MoE instance runs AEBS (Algorithm 1) as a GPU kernel.

