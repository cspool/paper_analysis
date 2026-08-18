# *D. Proof Architecture: Dividing the Proof Obligation*

Figure 18 illustrates the two forms of reasoning in our endto-end proof. First, one must reason ( I ) about the ISM itself (§VII-B) to determine if it satisfies the axiom we are verifying using induction (§VII-A). Second, one must connect ( R ) the ISM with the operational model to ensure that the ISM is a faithful representation of the hardware.

Previous work has separated these two forms of reasoning by introducing an intermediate state machine that builds partial execution graphs [5, 25, 47, 64], and demonstrating that the axioms hold for each generated partial execution. Some of these works [5] performed a machine-checked proof of I by showing that the ISM produces exactly the set of all axiomatically consistent outcomes. However, the proof that the executions of the operational model correspond to those of the ISM ( R ) was done by hand, an error-prone approach that has historically missed bugs [28, 41, 65].

In contrast, we machine-check both parts, constructing an *end-to-end* machine-checked proof of axiomatic consistency. The feasibility of machine-checking the entire proof is facilitated by the decoupling of concerns that allows the two proofs to focus on different aspects of correctness in a modular way: the refinement proof (§VII-E) focuses solely on abstracting away the internals of data movement through layers of the caches and the engine running in the operational model, while the proof of axiomatic consistency only reasons about how adding events to the execution graph changes its structure.

