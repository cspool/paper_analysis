# A Detailed Explanations for the Dependency Rules

#### <span id="page-14-1"></span>**Group B: Multi-head Attention** Node I Node J Case 1: Node J is dependent on Node I Head 1 Query Key Value Head n … … … … Node J Case 2: Node I is dependent on Node J Node I Node I Node J Case 3: Node K is not dependent on Node J Node K

Figure 6: Illustrations of the two dependency rules. All the cases are extracted from the multi-head attention module.

We provide a detailed explanation of the two dependency rules. It is important to note that these dependency rules do not pertain solely to the forward computation. Instead, they represent directional relationships that exist in both directions. For instance, removing a node in a subsequent layer may also result in the pruning of a node in the preceding layer. Recall the two dependency rules as follows:

<span id="page-14-2"></span>
$$N_j \in \operatorname{Out}(N_i) \wedge \operatorname{Deg}^-(N_j) = 1 \Rightarrow N_j \text{ is dependent on } N_i$$
 (7)

<span id="page-14-3"></span>
$$N_i \in \operatorname{In}(N_j) \wedge \operatorname{Deg}^+(N_i) = 1 \Rightarrow N_i \text{ is dependent on } N_j$$
 (8)

where N<sup>i</sup> and N<sup>j</sup> are two neurons. In(Ni) and Out(Ni) represents all the neurons that point towards or point from N<sup>i</sup> . Deg<sup>−</sup>(Ni) and Deg+(Ni) represents the in-degree and out-degree of neuron N<sup>i</sup> .

Figure [6](#page-14-1) serves as an illustration of the two dependency rules:

- In case 1, Node I and Node J satisfy the rule stated in Eq[.7.](#page-14-2) Consequently, Node J depends on Node I. When Node I is pruned, it is necessary to prune Node J as well.
- In case 2, Node I and Node J satisfy the rule Eq[.8.](#page-14-3) Thus, Node I is dependent on Node J. If Node J is pruned, it becomes imperative to prune Node I as well.
- In case 3, Node J and Node K do not meet the requirement of Eq[.7](#page-14-2) due to the mismatch in Deg<sup>−</sup>(Nk) ̸= 1. Thus, with Node J pruned, Node K would not be affected.

