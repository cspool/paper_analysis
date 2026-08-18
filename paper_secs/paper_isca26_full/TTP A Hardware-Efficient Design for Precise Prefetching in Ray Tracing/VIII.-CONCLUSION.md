# VIII. CONCLUSION

In this work, we propose a low-overhead, accurate hardware prefetcher, tree traversal prefetecher (TTP), for GPU accelerated ray tracing. TTP leverages the readily available traversal stack in hardware to accurately prefetch BVH tree nodes in a timely manner. For DFS based traversal, tree nodes are prefetched when consecutive stack pop operations happen, which indicate upward traversal. We show that such upward traversals make up a large portion of all RT read misses. As the per-thread traversal stacks store addresses of nodes that will be accessed next, this scheme requires no speculation of addresses, since the nodes in the stack will eventually be popped and read. TTP also readily supports BFS-based traversal, which is preferred for certain types of 3D scenes as well as general-purpose graph workloads.

We evaluate TTP on Vulkan-sim 2.0, a cycle-level GPU architectural model, and show that it achieves up to 1.89x speedup with a geometric mean of 1.48x in path tracing workloads, while saving the overall energy by 8.70%. To estimate the hardware overhead of TTP, we synthesize the RTL for state machines, which consume negligible space compared to existing hardware. We also compare our TTP with the stateof-the-art Treelet prefetcher specially designed for ray tracing. We simulate both prefetchers at various resolutions and show that TTP outperforms the Treelet prefetcher with much less hardware and software overhead.

#### ACKNOWLEDGEMENTS

We thank the anonymous reviewers for their valuable comments. The work is funded in part by NSF grants PHY-2325080 (with a subcontract to NC State University from Duke University), and OMA-2120757 (with a subcontract to NC State University from the University of Maryland).

#### APPENDIX

