# <span id="page-3-0"></span>3.4 Adaptive Token Mask Cache with Earley Parsing

Prior works, such as XGrammar [9], use a token mask cache to accelerate mask generation by preprocessing the majority of tokens ahead of time. However, this design is tied to the state organization of pushdown automata. Under non-deterministic grammars, the number of PDA states can grow exponentially, which degrades both grammar compilation and runtime mask generation. To preserve the benefit of caching while improving efficiency on more complex

<span id="page-4-2"></span>**Algorithm 1** Canonical Hash of One FSM Given Referenced FSM Hashes

```
Input: Finite state machine \mathcal{A} = (S, E, F, s_0), where S, E, F, and
s<sub>0</sub> denote the state set, edge set, final-state set, and initial state
Input: For every rule-reference edge e \in E, the hash of the
referenced FSM h(e.ref) is already available
Output: Canonical structural hash h of \mathcal{A}
Hash function: Let \mathcal{H} be an order-sensitive hash function over
sequences
Constants: NODE_TAG, RANGE_TAG, REF_TAG, EPS_TAG
Phase 1: Canonical state ordering
Sort the outgoing edges of each state in the following order:
  (1) character-range edges by (e.min, e.max)
  (2) rule-reference edges by (h(e.ref))
  (3) epsilon edges
Run BFS from s<sub>0</sub> using the sorted outgoing edges
Assign each state a canonical ID in discovery order
Phase 2: Hash in the canonical order
Let M be the map from states to their canonical IDs, and let
for each state s in increasing canonical ID order do
  h \leftarrow \mathcal{H}(h, NODE\_TAG, \mathbf{1}[s \in F])
  for each edge e in the sorted outgoing edges of s do
     if e is a character-range edge then
        h \leftarrow \mathcal{H}(h, RANGE\_TAG, e.min, e.max, M[e.target])
     else if e is a rule-reference edge then
        h \leftarrow \mathcal{H}(h, REF\_TAG, h(e.ref), M[e.target])
     else
        {e is an epsilon edge}
        h \leftarrow \mathcal{H}(h, EPS\_TAG, M[e.target])
     end if
  end for
end for
```

grammars, we build a new adaptive cache mechanism on top of the Earley parser. This design inherits the cache-based acceleration strategy of prior work, while leveraging the stronger parsing efficiency of Earley parsing for complex context-free grammars.

return h

The Earley parser [10] maintains, at each input position, a set of partial parsing states. Each state records a production rule, a dot position within that rule, and the input position where the matching of this rule began. Together, these states define the current parsing frontier. This state organization provides a natural foundation for token-mask caching, while also requiring the cache to be defined over Earley parsing frontiers rather than the state representation used in PDA-based parsing.

Based on this observation, we design an adaptive token mask cache mechanism for the Earley parser. The key idea is to cache token validity only for the part of the parsing frontier that can directly affect the next decoding step. In Earley parsing, only scannable states, i.e., states whose next symbol is a terminal, can immediately determine whether a token may be accepted. We therefore construct caches only for these scannable states. Non-scannable states, whose next symbol is a non-terminal, are not considered in

caching; instead, they will be expanded through Earley's prediction and completion operations into scannable states.

Regarding the cache content, we adapt XGrammar's token mask categorization to the Earley parser, categorizing tokens into accepted, rejected, and context-dependent cases. The first two categories can be determined by the current partial Earley parser state, while the context-dependent tokens require the whole parsing state history to be determined. At runtime, to compute the full token mask, we first retrieve the mask cache with the current scannable states, and then check the context-dependent tokens against the full Earley context. This design reduces cache construction overhead, enables effective cache reuse, and ensures efficient mask generation for complex non-deterministic grammars.

