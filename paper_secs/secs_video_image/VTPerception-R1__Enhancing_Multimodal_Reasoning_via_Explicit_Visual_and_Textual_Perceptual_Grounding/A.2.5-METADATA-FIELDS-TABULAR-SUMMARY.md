# A.2.5 METADATA FIELDS (TABULAR SUMMARY)

Table 4: Key fields stored with each processed sample.

| Field                        | Description                                     |
|------------------------------|-------------------------------------------------|
| id                           | Unique sample identifier                        |
| formal<br>description        | Canonical, objective image description (string) |
| cot<br>thinking              | Reconstructed chain-of-thought (string)         |
| final<br>answer              | Extracted final answer (normalized string/type) |
| quality<br>metrics           | Per-dimension scores and overall<br>score       |
| passed<br>quality<br>check   | Boolean pass/fail                               |
| metadata.detected<br>objects | Object detector output (list)                   |
| metadata.ocr<br>text         | OCR-extracted text (string)                     |
| processing<br>log            | Warnings, retries, and module traces            |

Concluding remark. The pipeline converts noisy multimodal CoT data into a high-quality, structured dataset suitable for training and evaluation. The modular design allows swapping or upgrading detectors, OCR backends, and LLMs while preserving the single-source-of-truth principle: all reasoning must be grounded in the generated formal description.

#### A.3 DATA CONSTRUCTION FOR RL

Method Overview. Algorithm [2](#page-16-0) outlines our pipeline for constructing high-quality reinforcement learning (RL) data. It involves three main steps: (i) sampling diverse chain-of-thought (CoT) trajectories from multiple strong teacher models, (ii) verifying the quality of these trajectories using a

#### <span id="page-16-0"></span>Algorithm 2 Data construction for RL: Visual & Textual Key Info

**Require:** Dataset D of samples  $(\mathbf{x}, q, a^*)$ , teacher models  $\mathcal{M}$ , per-teacher sample count N, judge budget B, thresholds  $(\tau_{acc}, \tau_{coh}, \tau_{cons})$ **Ensure:** Distilled set  $\mathcal{P}$  with CoTs and key information 1:  $\mathcal{P} \leftarrow \emptyset$ 2: for all  $(\mathbf{x}, q, a^*) \in D$  do Generate N CoT samples per teacher  $\rightarrow \mathcal{T}$ 3. 4: Select top-B by log-probability 5: Judge each: compute correctness  $s_{acc}$ , coherence  $s_{coh}$ 6: Keep candidates where  $s_{\rm acc} \geq \tau_{\rm acc}$  and  $s_{\rm coh} \geq \tau_{\rm coh}$ 7: if  $a^*$  is missing then 8: Filter by self-consistency 9: 10: Remove duplicates by final answer 11: Select best trajectory  $\tilde{t}$  from verified candidates 12: 13: Extract visual key info  $\mathcal{V}$  and textual key info  $\mathcal{Z}$  from  $\hat{t}$ 14: Add  $(\mathbf{x}, q, \hat{a}, \tilde{t}, \{\mathcal{V}, \mathcal{Z}\})$  to  $\mathcal{P}$ 15: **end for** 16: return P

budgeted evaluation process, and (iii) extracting two types of supervision signals: visual key info and textual key info. Each final training instance is represented as a tuple  $(\mathbf{x}, q, \hat{a}, \tilde{t}, \{\mathcal{V}, \mathcal{Z}\})$ , where  $\hat{a}$  is a verified answer,  $\tilde{t}$  is a validated trajectory, and  $\mathcal{V}, \mathcal{Z}$  are the extracted key information components used in perception-aware RL.

**Teacher Ensemble and Diversity.** For a given input item  $s = (\mathbf{x}, q, a^*)$ , we query several 72B-scale teacher models using stochastic decoding to generate a diverse set of reasoning paths  $\mathcal{T}$ . These models produce different but plausible solutions—such as varied reasoning structures, subgoal decompositions, or interpretation strategies—while remaining anchored by accurate prior knowledge.

**Budgeted Verification.** To manage annotation cost efficiently, we first rank candidate trajectories by model log-likelihood (TOPKBYLOGPROB) and select the top B for further evaluation. Each selected trajectory t is scored based on: (i) *Correctness*  $s_{\rm acc}$ : measured by exact or normalized answer match, or by tolerance-based comparison if  $a^{\star}$  is numeric. (ii) *Coherence*  $s_{\rm coh}$ : assessed by a forward-backward consistency checker that examines step validity, use of evidence, and logical soundness. Only candidates passing both thresholds  $(\tau_{\rm acc}, \tau_{\rm coh})$  are retained in K. When no ground-truth answer is available, we apply a SelfConsistencyFilter to retain trajectories that reach similar conclusions. Otherwise, we de-duplicate based on final answers. We then select a single best trajectory  $\tilde{t}$  using a weighted score:  $w_1 s_{\rm acc} + w_2 s_{\rm coh}$ .

**Key Info Extraction.** From the verified trajectory t, we extract two complementary types of supervision:

- Visual key info  $\mathcal{V}$ : This includes atomic, image-grounded facts such as object attributes, visual measurements, geometric constraints (e.g., parallel lines, equal lengths), and spatial relationships. These are precise and verifiable visual elements crucial for grounded reasoning.
- **Textual key info**  $\mathcal{Z}$ : This includes (i) a structured parse of the question's textual facts—such as entities, quantities, conditions, and units—and (ii) an *application map* that links these facts to specific reasoning steps in  $\tilde{t}$ , explicitly showing how each fact is used during problem solving. This provides a fine-grained, executable understanding of the question content.

**Output and Usage in RL.** Each training item stores the verified answer  $\hat{a}$ , trajectory  $\tilde{t}$ , and extracted key information  $\mathcal{V}, \mathcal{Z}$ . These are used during RL to define perception-level rewards by checking how well the model's outputs align with the key information. These are then combined

with answer correctness and format-based rewards. The combination of budgeted verification and self-consistency ensures high-quality data while controlling annotation cost.

Differences from Prior Visual-Only Pipelines. Unlike previous pipelines that rely solely on visual supervision, our method: (i) unifies both *visual* and *textual* perception signals, (ii) introduces a budget-aware selection strategy that reduces unnecessary annotation by filtering weak candidates early, and (iii) formalizes textual key info not just as extracted facts, but as a structured fact-toreasoning mapping, making question constraints explicit and testable.