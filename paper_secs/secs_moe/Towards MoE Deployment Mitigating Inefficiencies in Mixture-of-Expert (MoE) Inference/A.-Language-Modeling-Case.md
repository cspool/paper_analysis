# *A. Language Modeling Case*

For Language Modeling, we use the PILE dataset [\[8\]](#page-10-3) as the input, which is the validation set used in prior work [\[2\]](#page-10-0). We select three domains (Wikipedia, PubMed and Github) from the PILE dataset to study the effect of different input data on the expert activation patterns across time (*i.e.*, consecutive batches).

We visualize the results in Figure [6.](#page-5-1) Each row represents a batch and each column represents the load of a particular expert. A more intense color indicates the expert receiving a higher portion of all tokens in a batch. As shown in Figure [6\(](#page-5-1)a), load distribution across experts is highly imbalanced. There exists multiple hot experts that always get a large share of tokens (multiple lines of intense color), and the other experts consistently receive a small amount of tokens (lines of lighter colors).

In the most extreme cases, Figure [7](#page-5-2) indicates there exist experts that never get any tokens. Due to the static gating policy, these experts still receive and process empty token placeholders, introducing a huge waste of computational resources. As shown in Figures [6\(](#page-5-1)a) and [7,](#page-5-2) the set of hot experts and their hotness level varies across domains even though all domains consistently exhibit a high-degree of sparse expert activation.

