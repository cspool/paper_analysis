# *E. Connecting the ISM and Operational Model via Refinement*

The goal of the refinement proof is to demonstrate that any execution trace of the operational model (§VI) corresponds to an ISM execution trace that produces partial execution graphs.

What complicates this is the detail in the operational model: most transitions do not update the execution graph but still meaningfully impact the state for transitions that do. Figure 18 shows this disconnect in an execution snippet of Figure 4a's program. While the operational model performs four transitions (s1 → s5), only two of them (ending an OnWB (s1 → s2) and starting an OnMiss (s4 → s5)) update the execution graph. In contrast, when the L3 sends an OnMiss request to the Network (s2 → s3) and the Engine receives it (s3 → s4), the graph is not updated (as our MCM is at ISA level and does not model unnecessary hardware details).

However, these *internal steps* are still pivotal to correctness: when the Engine later starts an OnMiss (s4 → s5), the address for which it runs and the cache line it populates are

<sup>2</sup>Prefix-closure in the literature is typically defined with respect to a commitment order: our commitment order respects cbo, meaning we never add callback events to the graph in an order that violates cbo.

![](_page_14_Figure_0.jpeg)

Fig. 18: The two components of our soundness proof. (\$VII-E) The refinement proof (\$VII-E) proves that each operational model execution (e.g.,  $s1 \to s5$ ), abstracted at each state, maps to an intermediate state machine execution (e.g.,  $abs(s1) \to abs(s5)$ ). Operational model transitions either add nodes to the partial execution graph (e.g.,  $s1 \to s2$ ) or are internal steps that abstract up to NoOps (e.g.,  $s2 \to s3$ ). (\$VII-B).

determined by the message it receives from the Network, which in turn is determined by the previous internal steps.

Figure 18 shows how refinement reasons about correctness in the presence of internal steps. We first define an abstraction function abs that, for any s of the täkō operational model produces an equivalent state in the ISM. Then we show via induction that for any transition  $s \to s'$  of the operational model,  $abs(s) \to abs(s')$  is either a valid ISM transition (e.g.,  $s1 \to s2$ ), or makes no change to the graph (e.g.,  $s2 \to s3$ ).

As we prove refinement through induction, this involves proving an inductive invariant which shows that the added complexity of caches and Network communication does not change the underlying behavior of the system.

Our proof assumes certain basic properties about our coherence protocol to avoid having to prove coherence in addition to correspondence between our täkō implementation model and our ISA-level MCM. We do so because the coherence protocol we use is well-studied and is known to provide coherence [46, 48]. Additionally, the coherence protocol in a täkō system is tangential to täkō's novel features: after phantom data is received from the engine and populates an entry in the directory-level cache, the data is indistinguishable from data fetched from memory to the coherence protocol.

We explicitly model all protocol transient states and their interaction with täkō, thus verifying any coherence-consistency interface [40] issues that might arise. For example, we verify that the dirty bit at directory level is accurately preserved by the coherence protocol, to ensure that OnEvict and OnWB are invoked appropriately. Even assuming coherence, our proof still required adding 119 clauses to our inductive invariant and 61K LoC of proof annotations.

