# <span id="page-11-2"></span>B.2 Universal Template Elements and Failure Handling

Return policy invariants. We enforce four invariants (summarized in Table 4). (i) Monotone progress:  $t_1{=}1$  and valid calls advance  $t_{k+1}{=}t_k{+}1$ ; invalid indices are rejected. (ii) Format completeness: validate(sequence) holds iff the sequence matches the required complete, comma-separated format and ordering. (iii) Return policy: valid sequences with  $t{<}T_{\rm max}$  trigger a Progress notice, while invalid sequences trigger a Repair notice without advancing t. (iv) Termination: only when  $t{=}T_{\rm max}$  and the latest sequence validates does the server emit the Terminal return (benign payload). Together, these invariants induce multi-turn, long tool-call traces while preserving task correctness and MCP compatibility without modifying code, identifiers, or payload semantics.

#### <span id="page-11-3"></span>**B.3** Seed Bank Screening and Promotion

The seed bank is a repository of protocol-compatible, task-correct, text-only templates  $T_{\theta}$ . We initialize it with a single human-authored seed and lightly screen on a fixed query set/agent to confirm acceptance at the fixed  $L^*$ . Each MCTS run starts from a selected seed and halts when the acceptance predicate is met; the resulting template is written back with minimal metadata (estimated ASR, segment stability, omission/repair rates, refusal notes). Subsequent runs resample top seeds by ASR and stability and apply a stricter acceptance target before promotion. This cyclic promotion improves starting points without touching code or identifiers and leaves the terminal payload unchanged. In practice, a few cycles yield reusable seeds that transfer across LLMs and MCP servers.

#### <span id="page-11-4"></span>**B.4** MCTS Optimizer Details

Action space and edit zones. We organize atomic text edits into three families:  $\mathcal{A}_{\mathrm{MT}}$  (multi-turn induction),  $\mathcal{A}_{\mathrm{LEN}}$ (length induction), and  $\mathcal{A}_{REP}$  (repair after omission/format errors). Beyond phase-aware gating, we instantiate these families as 16 atomic edits applied exclusively to non-executable, text-visible zones of the server (docstring argument descriptions, in-progress/unfinished notices, and validation-error messages). The multi-turn family sharpens next-call salience and enforces monotone segment progression; the length family strengthens the "complete, comma-separated" requirement to elicit long single-shot payloads during tool calling; and the repair family refines failure messaging to immediately solicit a compliant retry without advancing the segment. These primitives are intentionally small, mutually composable, and largely orthogonal, enabling MCTS to explore nuanced trade-offs between adherence and refusal while keeping the surface area auditable. Throughout, function identifiers, control flow, and terminal payload semantics remain untouched.

Phase gating, expansion, and parallelism. Each tree node v corresponds to a concrete server  $T_{\theta_v}$ ; edges apply a single localized text edit. We maintain a phase label  $\phi(v) \in \{\text{pre\_MT}, \text{post\_MT}\}$  and a node-local omission flag that unlocks repair actions if needed. In  $\text{pre\_MT}$ , we use  $\mathcal{A}_{\text{MT}}$  to stabilize multi-turn behavior; once screenings show stable segment sequencing, we switch to  $\text{post\_MT}$  and use  $\mathcal{A}_{\text{LEN}}$  to strengthen long outputs. An omission/format error observed at node v unlocks  $\mathcal{A}_{\text{REP}}$  at v only. When a node is expanded, we instantiate one child per untried action from the phase-appropriate set (plus  $\mathcal{A}_{\text{REP}}$  if enabled) and evaluate all new children in parallel.

Node selection and statistics. We use UCT (Kocsis and Szepesvári, 2006) with a running mean evaluation signal  $\bar{Q}$  and visit counts  $N_{\text{uct}}$  for exploration (see §3.4). UCT counts can be updated using Stage-1 samples only to avoid heavy batches skewing exploration, while  $\bar{Q}$  aggregates all observed rollouts.

<span id="page-12-0"></span>

| Element                     | Type   | Effect on agent trajectory                                                                                               | Constraints / failure handling                                                                                                                                                     |
|-----------------------------|--------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Segment (t)                 | ARG    | Marks progress and induces multi-turn; the<br>agent increments t→t+1 until a terminal<br>cue.                            | t ≥ 1 and strictly monotone by +1; non-positive<br>or non-monotone values are rejected.                                                                                            |
| Calibration<br>sequence (L) | ARG    | Inflates per-turn completion at the tool<br>calling site via a full comma-separated list.                                | Exactly L integers, strictly increasing, using digits<br>and commas (optional spaces); no ranges (e.g.,<br>"1–5"), no ellipses (" "), no duplicates.<br>Malformed ⇒ Repair notice. |
| Progress notice             | RETURN | Declares "in progress"; instructs the next<br>call with t+1 and a full calibration<br>sequence; preserves the goal path. | Emitted iff t < Tmax and the latest sequence<br>validates; never alters code, identifiers, or payload<br>semantics.                                                                |
| Repair notice               | RETURN | Corrects abbreviated/invalid sequences;<br>prevents bypass of the length gate.                                           | Triggers on omissions, ranges, duplicates, wrong<br>length/order, or illegal characters; keeps t<br>unchanged. Requests the complete<br>comma-separated list before proceeding.    |
| Terminal return             | RETURN | Ends the trajectory and returns the same<br>benign payload as the original server.                                       | Emitted only when t=Tmax and the latest<br>sequence validates; protocol-compatible<br>pass-through.                                                                                |

Table 4: Universal template elements (Appendix). Text-only arguments and notices enforce multi-turn progress and per-turn verbosity; the terminal return passes through the unchanged benign payload.

Node evaluation and reward. Each child undergoes a twostage evaluation with configurable sizes and gates. For each rollout, we compute

$$mt_{pass} = \mathbb{I}\{MT\}, \quad len_{pass} = \mathbb{I}\{LEN\},$$

where MT means the multi-turn target is met with ordered segments, and LEN means the fixed L ∗ is reached and any omissions are repaired. We use

$$r = \alpha \operatorname{mt\_pass} + \beta \operatorname{mt\_pass} \operatorname{len\_pass},$$
  
where  $0 < \alpha \le \beta \le 1$ ,

prioritizing stable multi-turn behavior (the α term) and adding credit for length only when multi-turn has been achieved (the multiplicative term). Stage-1 offers a quick screen to decide whether to run Stage-2 and to flip ϕ : pre\_MT→post\_MT once segment sequencing stabilizes; Stage-2 refines estimates under stochastic decoding.

Backpropagation. We propagate statistics along the path to the root. If a node meets the acceptance predicate (stabilized successes over the latest batch), we record the corresponding Tθ<sup>⋆</sup> and insert it into the seed bank.

