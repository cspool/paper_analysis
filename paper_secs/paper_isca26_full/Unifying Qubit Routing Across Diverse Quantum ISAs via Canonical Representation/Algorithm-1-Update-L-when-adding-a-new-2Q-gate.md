# **Algorithm 1:** Update L when adding a new 2Q gate

```
Input: G' (Routed DAG), \pi (current logic-to-physical
              mapping), L (last mapped layer), D (wire durations
             for each qubit), C (commutative pairs within L)
   Output: Updated G', L, D, C
   /\star g: resolved logical gate; g': routed gate \star/
 1 g' \leftarrow G'.PUSHBACK(g, \pi[g.q_0], \pi[g.q_1]); // <math>g'.q_i = \pi[g.q_i]
2 d \leftarrow \text{MAX}(D[g'.q_0], D[g'.q_1]) + \text{SYNTHCOST}(g);
3 D[g'.q_0] \leftarrow d; D[g'.q_1] \leftarrow d;
4 for pred \in G'.PREDECESSORS(g') do
        if IS2QGATE(pred) then
5
            if isCommutativeCanonicalPair(g', pred)
                 C[(\text{pred}.q_0, \text{pred}.q_1)] \leftarrow (g'.q_0, g'.q_1);
 7
                 L.POP((pred.q_0, pred.q_1), NONE);
                 C.POP((pred.q_0, pred.q_1), NONE);
10
11
             /* pred_pred must be None or a 2Q gate */
             \operatorname{pred\_pred} \leftarrow \operatorname{NEXT}(G'.\operatorname{PREDECESSORS}(\operatorname{pred}));
12
            if pred_pred \neq None then
13
                 L.POP((pred\_pred.q_0, pred\_pred.q_1), NONE);
14
                 C.POP((pred\_pred.q_0, pred\_pred.q_1), NONE);
15
16 L[(g'.q_0, g'.q_1)] \leftarrow g';
```

