# <span id="page-5-0"></span>*C. Cross-block Intermediate Representation and Optimization*

<span id="page-5-3"></span>*1) Static Single-assignment (SSA) Style Rotation Merging:* For the standard Clifford+T gate set {T, T† , S, S† , H, X, CNOT} [\[37\]](#page-13-35), the phase polynomial gate set {CNOT, Rz} is *not* universal. We therefore partition a general quantum circuit into blocks of phase polynomial subcircuits separated by non-R<sup>z</sup> single-qubit gates. In particular, H gates act as *block barriers*: two-qubit gates whose semantics depend on the post-H basis are excluded from the preceding phase polynomial block. Fig. [9\(](#page-5-2)a) illustrates a case with two such blocks.

<span id="page-5-2"></span>![](_page_5_Figure_13.jpeg)

Fig. 9: Cross-block optimization: (a) Two separate blocks before optimization (5 CNOTs, 3 T gates). (b) Across the H gate, we create a new qubit wire. In the new circuit, the output-parity before the H gate is set to q<sup>1</sup> ⊕ q2, the same as that before the H gate in the original circuit. Now we have a new phase-polynomial block, by optimizing this new block, we reduce the circuit to 4 CNOTs and 1 T.

To enable rotation merging across block boundaries, we introduce an SSA-style [\[38\]](#page-13-36) *qubit-state renaming and rotation merging*. Each input qubit state, and each state created after an H gate, is assigned a fresh SSA identifier. Every R<sup>z</sup> gate is then tagged with the SSA ID of the qubit state it acts on. By merging all phase terms associated with the same SSA ID, we achieve *whole-circuit* rotation merging rather than being restricted to a single block.

For example, in Fig. [9\(](#page-5-2)b), two T gates on q<sup>1</sup> share the same SSA ID and therefore merge, even though they appear in different blocks. Likewise, the blue wire q<sup>2</sup> ends at the SSA state {q1⊕q2} before the H gate; after the H gate we create a new SSA state q3, yielding an updated output-parity {q0⊕q3}.

Prior work on rotation merging [\[11\]](#page-13-10) typically operates within a single block, using *anchors* and *terminal points* to extend block-local cancellations; this increases implementation complexity and provides limited guarantees [\[39\]](#page-13-37). In contrast, SSA-based qubit-state renaming yields a simple correctness criterion—rotations merge *iff* they target the same SSA ID—thereby enabling global merging across block boundaries. Recent work by Amy and Lunderville [\[40\]](#page-13-38) formulates rotation merging via relational program analysis, discovering additional opportunities across control flow and non-linear relations (Toffoli gates) for rotation gate merging. Our approach is complementary: it also exposes parity relationships between two-qubit gates, constructing a Cross-block IR that reveals long-range parity reuse and reduces CNOTs across blocks.

Before constructing rotation merging, we perform preprocessing to reduce redundant block barriers: (i) Propagate X gates forward via Clifford conjugation so that H gates remain the only block barriers; (ii) Cancel adjacent H pairs; (iii) If a CNOT gate is bracketed by H on both wires, cancel the four H gates and switch the control-target accordingly; if exactly one wire is bracketed, conjugate that wire and insert two H gates on the other wire to preserve equivalence. We interleave this preprocessing pass with rotation merging twice.

*2) Cross-block Parity Matrix Intermediate Representation Design:* We merge adjacent single-block phase polynomial regions into a larger phase polynomial block using a *crossblock IR*, as shown in Fig. [9\(](#page-5-2)b). In the cross-block setting, post-H qubit states (new qubit rows) exist in the IR but remain *inactive* until their producer row (the original qubit wire before the H) is eliminated and the H gate is inserted; only then is the row activated and available for operations.

To eliminate a pre-H row and activate its successor (e.g.,  $q_3$  in Fig. 9), three conditions must hold:

- No pending phase terms on the row: the corresponding row in the phase-parity block is all zeros (no remaining phase terms depend on this state).
- 2) **Column isolation:** in the output-parity block, if the *i*-th row is to be removed, the *i*-th column must form a unit vector with its sole 1 located at row *i* (indicating that the target output state has been correctly prepared).
- 3) **Row isolation:** in the output-parity block, the *i*-th row must be a unit vector, containing a single 1 at its diagonal position in the output matrix.

When these are satisfied, this  $\operatorname{pre-}H$  row can be removed, which activates the  $\operatorname{post-}H$  row. The  $\operatorname{pre-}H$  row maintains the correct output-parity, i.e.,  $q_1 \oplus q_2$  in Fig. 9(b), before it retires. Elimination of the  $\operatorname{pre-}H$  row does not cause information loss: any correlations between the eliminated row and others have already been addressed or transferred. All subsequent transformations remain row operations induced by CNOTs.

<span id="page-6-0"></span>3) Linear Dependency Check for Correctness: Once conditions (1) and (3) are satisfied, condition (2) follows directly. In such a case, the joint parity matrix contains a row of the form  $v = [0 \cdots 0 \mid 0 \cdots 010 \cdots 0]$ . Because all other entries of v are zero, adding this row to any other row clears the 1's in the same column, achieving column isolation. Therefore, we first make sure that the CNOT network synthesis process can produce circuit states that satisfy condition (3) together with condition (1), producing the required diagonal 1 in the output block, which is then used to enforce condition (2). Making sure these two conditions are satisfiable, therefore, is equivalent to checking whether the target row v can be written as an XOR (over GF(2)) of a set of candidate rows from the overall parity matrix. Equivalently, we must check whether the desired unit vector v lies in the span of these rows.

We use a rank-based test for this purpose. Let M be the matrix formed by the candidate rows. Appending v as an additional row, elimination of the pre-H row is feasible iff  $\operatorname{rank}(M \cup \{v\}) = \operatorname{rank}(M)$ , otherwise v is linearly independent of M and the elimination is impossible.

Hence, as we update the coupled matrix [phase-parity | output-parity] using CNOT gates, we constantly check whether the resulting parity matrix passes the rank test. If a state fails this rank test, it is immediately pruned. This check ensures that the pre-*H* row(s) are removable in the end, and significantly improves efficiency.

- 4) Cross-Block Optimization Implementation: Using the above representation and rank check, we integrate it with the co-optimization of the phase-parity network and the output-parity network (Section III-A). Post-H SSA rows remain inactive and are excluded from row-pair selection until their pre-H producer row satisfies conditions (1), (2), and (3). Then, we insert the H gate, and activate its post-H row.
- a) Feasibility and witness set: We first run the linear dependency check (Section III-C3); if it is infeasible, we prune

the state. When feasible, solve  $M\alpha = v$  over  $\mathrm{GF}(2)$  and let  $S = \{i \mid \alpha_i = 1\}$  be a witness set whose XOR equals v. Write t for the index of the pre-H row to eliminate. If  $t \notin S$ , select any  $j \in S$  and apply  $\mathrm{CNOT}(j,t)$ . This replaces  $\mathrm{row}_j$  by  $\mathrm{row}_j \oplus \mathrm{row}_t$  without changing the span, yielding an equivalent witness set S' with  $t \in S'$ . Row isolation (condition (3)). For each  $i \in S' \setminus \{t\}$ , apply  $\mathrm{CNOT}(t,i)$  so that  $\mathrm{row}_t \leftarrow \bigoplus_{i \in S'} \mathrm{row}_i = v$ . Column isolation (condition (2)). For every  $k \neq t$  with a 1 in column t, apply  $\mathrm{CNOT}(k,t)$  to clear that entry. Eliminate and activate. Remove row t and insert t, which activates the post-t SSA row. Fig. 10 illustrates a case with  $t \notin S$ : we first bring t into the witness with one CNOT, then isolate the row, clear the column, and finally eliminate t and insert t to activate the post-t state.

<span id="page-6-1"></span>
$$\begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 1 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{CNOT}(q_1, q_0)} \begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 1 & 1 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{CNOT}(q_0, q_1)} \begin{bmatrix} \mathbf{1} & \mathbf{0} & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & 1 & 1 & 0 \\ \mathbf{0} & 0 & 1 & 0 \\ \mathbf{0} & 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{Eliminate } q_0} \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \xrightarrow{\text{Eliminate } q_0} \begin{bmatrix} 1 & 1 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

Fig. 10: Example of the  $t \notin S$  case, t=1 is the row to be eliminated, and S is the witness set which will XOR to the target row. (A) Initial cross-block intermediate representation, currently, the first row is expected to be eliminated, and the fourth row will be activated after that. (B) Bring t into the witness combination, yielding an equivalent witness set S' with  $t \in S'$ . (C) Isolate row/column t using CNOTs. (D) Eliminate t and insert H to activate the post-H state.

b) Scalability and robustness: The cross-block mechanism ensures correctness by locking inactive post-H rows and pruning infeasible states, but merging blocks increases complexity and may occasionally underperform the single-block optimization. To address this, we apply two strategies.

First, block merging is performed during a forward traversal of the block DAG obtained after applying the partition rules in Section III-C1. Each block corresponds to a node, with edges representing qubit dependencies. At each node, we evaluate merge opportunities with its immediate predecessor blocks.

To control complexity, we apply pruning: two blocks are merged only if they share multi-qubit interactions. The merging terminates when no further sharable interactions exist or when the merged block reaches the predefined size limit. For example, in Fig. 9, Block 1 acts on  $\{q_0, q_1, q_2\}$  and Block 2 on  $\{q_0, q_1, q_3\}$ . Since they share  $\{q_0, q_1\}$ , merging is allowed.

Second, we adopt an *Incremental Block Merging* strategy: we begin with single-block optimization as a stable baseline, then progressively apply cross-block merging gradually and interleave with local refinement. If a merge fails to improve upon the previous, we revert it and keep the previous state.

#### IV. EVALUATION

We implement the proposed techniques in *PhasePoly* and evaluate them through the following research questions:

Q1: How does *PhasePoly* compare with existing phase polynomial optimization methods?

Q2: Why is *PhasePoly* necessary in general optimization? Q3: Can *PhasePoly* capture long-range optimization opportunities at scale?

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

Fig. 11: Normalized total and two-qubit gate-count reductions across benchmark circuits, comparing *PhasePoly* against phase polynomial baselines. All values are normalized to the unoptimized circuits (1.0), with lower bars indicating greater reduction.

Q4: Do *PhasePoly*'s logical reductions translate to nearterm hardware execution improvements?

Q5: How does *PhasePoly* benefit fault-tolerant compilation pipelines, and how should it be integrated?

Q6: Is the cross-block optimization correct and robust? Q7: What are the compilation cost and parameter sensi-

tivity of *PhasePoly*?

