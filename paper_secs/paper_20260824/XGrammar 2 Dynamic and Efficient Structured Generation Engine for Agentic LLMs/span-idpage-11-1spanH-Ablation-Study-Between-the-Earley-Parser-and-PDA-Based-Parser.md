# <span id="page-11-1"></span>H Ablation Study Between the Earley Parser and PDA Based Parser

We also want to measure the advantages of the Earley Parser as an ablation study. Thus, we evaluate the efficiency of XGrammar-2, with PDA based parser and the Earley Parser, respectively, and both of them will compile the JSON schemas ahead of time. The dataset is JSONSchemaBench [\[11\]](#page-9-17), and the result in [Figure 12](#page-12-1) shows that the Earley Parser can significantly reduce the grammar compilation. Note that the long-tail is caused by the huge inputs, instead of the complexity of the algorithm.

<span id="page-12-1"></span>> **[图片提取文字 (无描述)]:**
> JSON Schema Compilation Time 1,000,000 Earley Parser Version PDA Version 100,000 10,000 1,000 Time (ms) 100 10 0.10 0.01 0.6 0.2 0.4 0.8 Cumulative Distribution Function
![](_page_12_Figure_2.jpeg)

Figure 12: Comparison between the Earley Parser and PDA on JSONSchemaBench.

#### <span id="page-12-0"></span>I Correctness and Task-level Effectiveness

By construction, constrained decoding guarantees that generated outputs conform to the target structure (e.g., JSON schema or toolcalling format). XGrammar-2 preserves the same constraint semantics as XGrammar, and thus both achieve 100% schema-valid tool-call arguments whenever a tool call is produced; the difference is efficiency (Section 4.3).

<span id="page-12-2"></span>

| Model Name      | Trmo           | Correct   | Correct     |  |
|-----------------|----------------|-----------|-------------|--|
| Wiodel Name     | Type           | Call Rate | Schema Rate |  |
| Llama-3.2-1B    | w/o XGrammar-2 | 6.07%     | 22.07%      |  |
| Liaina-3.2-1D   | w/ XGrammar-2  | 32.84%    | 100.00%     |  |
| Llama-3.2-3B    | w/o XGrammar-2 | 33.12%    | 40.70%      |  |
| Liailia-3.2-3D  | w/ XGrammar-2  | 77.75%    | 100.00%     |  |
| Llama-3.1-8B    | w/o XGrammar-2 | 59.48%    | 66.95%      |  |
| Liaina-3.1-0D   | w/ XGrammar-2  | 80.93%    | 100.00%     |  |
| Llama-3.1-70B   | w/o XGrammar-2 | 45.60%    | 51.94%      |  |
| Liailia-3.1-70B | w/ XGrammar-2  | 86.41%    | 100.00%     |  |

Table 6: The function calling accuracy rate and the JSON schema validity rate.

To validate end-to-end correctness and quantify task-level impact in realistic agent settings, we evaluate on BFCL-v3 [27]. As shown in Table 6, grammar-constrained decoding (XGrammar-2) substantially improves BFCL function-calling outcomes for most models, primarily by eliminating malformed tool calls (e.g., invalid JSON or schema violations) that would otherwise be unexecutable and scored as failures. Constraint enforcement can also narrow the gap between small and large models; for example, XGrammar-2 enables Llama-3.2-3B to outperform an unconstrained Llama-3.1-70B baseline on BFCL.

## J Formal Definitions of the Earley Parser and the Token Mask Generation with Cache

Table 7 shows the formal definition of the Earley Parser [23], and the formal definition of the token mask generation with cache. In the Table 7, Grammar Production represents a series of rules in the format of rule  $\rightarrow \gamma$ , where  $\gamma(\mu, \rho)$  is the sequence of the rule. A, B represents the non-terminal elements in the sequence, and a represents the terminal element.  $\mathcal V$  is the vocabulary of the tokenizer. For a token mask cache,  $\mathcal A$  means the set of accepted tokens,  $\mathcal U$  means the set of uncertain tokens, and  $\mathcal R$  means the set of rejected tokens.

<span id="page-13-0"></span>

|                | Earley Parser                                                                                                                    | Tok                           | en Mask Generation with Cache                                                                                                                                                                                            |
|----------------|----------------------------------------------------------------------------------------------------------------------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input          | String $x$ , Length $N$ , Start Symbol $S$ , Grammar Productions $\mathcal{P}$                                                   | Input                         | Vocabulary ${\mathcal V}$                                                                                                                                                                                                |
| Variables      | Indices $i,\ j,\ k\in[N],$ Non-terminals $A,\ B,$ Sequences $\mu,\ v,\ \rho,$ Terminal $a\in\Sigma,$ Augmented Start Symbol $S'$ | Token Mask Cache              | $[*,*,A \to \mu \bullet av] \to (\mathcal{A},\mathcal{R},\mathcal{U})$ $\mathcal{A}, \ \mathcal{R}, \ \mathcal{U} \subset \mathcal{V}, A \sqcup \mathcal{R} \sqcup \mathcal{U} = \mathcal{V}$                            |
| State          | $[i,j,A\to \mu \bullet \nu]$                                                                                                     | Accepted                      | $\mathcal{A} = \{v \in V: \exists [*,  v , *]$ derived from $[0, 0, A \to {}^\bullet a\mu]$ and input $v\}$                                                                                                              |
| Initialize     | $[0,0,S'\to \bullet S]$                                                                                                          | Uncertain                     | $\mathcal{U} = \{ v \in V \setminus \mathcal{A} : \exists i <  v , \ [0, i, A \to av \bullet]$ derived from $[0, 0, A \to \bullet av]$ and input $v_{0:i} \}$                                                            |
| Goal           | $[0,N,S'\to S^\bullet]$                                                                                                          | Rejected                      | $\mathcal{R} = V \setminus (\mathcal{A} \cup \mathcal{U})$                                                                                                                                                               |
|                | Rules                                                                                                                            | Ge                            | nerate Token Mask(At runtime):                                                                                                                                                                                           |
| ①Predict Rule  | $\frac{[i,j,A\to\mu\bullet B\ v] \qquad (B\to\rho)\in\mathcal{P}}{[j,j,B\to\bullet\rho]}$                                        | ①Retrieve Token Mask<br>Cache | $(\mathcal{A},\mathcal{R},\mathcal{U})$ from state $[i,j,A \rightarrow \mu \bullet av]$                                                                                                                                  |
| ©Scan Rule     | $\frac{[i, j, A \to \mu \bullet a v]  x_j = a}{[i, k, A \to \mu a \bullet v]}$                                                   | ©Check Uncertain              | $\mathcal{U}_{\mathcal{R}} = \{v \in \mathcal{U} : \exists [*, j +  v , *]$<br>derived from $[i, j, A \to \mu \cdot v]$ and input $v\}$<br>$\mathcal{U}_{\mathcal{R}} = \mathcal{U} \setminus \mathcal{U}_{\mathcal{R}}$ |
| ③Complete Rule | $\frac{[j,j,B\to\bullet\rho] [j,k,B\to\rho\bullet]}{[i,k,A\to\mu B\bullet\nu]}$                                                  | 3Output Token Mask            | $(\mathcal{A}', \mathcal{R}'),  \mathcal{A}' = \mathcal{A} \cup \mathcal{U}_{\mathcal{A}},  \mathcal{R}' = \mathcal{R} \cup \mathcal{U}_{\mathcal{R}}$                                                                   |

Table 7: Formal definitions of the Earley parser[23] and the token mask generation with cache.