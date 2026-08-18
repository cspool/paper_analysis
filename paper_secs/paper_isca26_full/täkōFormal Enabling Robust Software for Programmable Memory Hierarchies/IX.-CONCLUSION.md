# IX. CONCLUSION

We develop a MCM for the täkō PMH, enabling programmers to reason about täkō programs without understanding täkō's implementation details. We also construct a microarchitectural täkō model that is parameterized over prefetching policies, cache replacement policies, and network-onchip specifics. This model enables architects to change these features in their täkō design to improve performance without compromising correctness. Finally, we prove our MCM sound against our microarchitectural model across all programs, thus verifying that our MCM accurately represents täkō.

Our formalization also discovers two more general insights. First, when creating an ISA-level MCM, ensuring prefix-closure for its axioms makes the MCM amenable to inductive correctness proofs. Second, microarchitectural models should (and *can*) serve the needs of both architects and formal methods experts, as our microarchitectural model of täkō does.

#### ACKNOWLEDGMENTS

We thank the anonymous reviewers and our shepherd for their helpful feedback. We thank Brian Schwedock and Nathan Beckmann for clarifying certain täkō implementation details. This work was supported in part by National Science Foundation grant CCF-2318954. GitHub Copilot was used for mundane code autocomplete and generation (e.g., find &

replace) in our Dafny proofs. (We handwrote the vast majority of our proofs.)

