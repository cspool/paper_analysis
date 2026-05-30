# <span id="page-1-3"></span>2. Analysis

In this section, we begin with defining the latency optimization problem in semantic space with introducing the Consecutive Semantic Difference (CSD). Given that consecutive tokens show semantic localities, we aim to find a low-variance semantic embedding for each token. We first model the token embeddings as the combination of highlevel semantics and token-identity semantics. Then, we use attention scores to discover high-level semantics similarity and obtain semantic groups, and construct the Oracle Space with semantic group embeddings. Finally, we show that oracle-space-based routing yields significantly lower CSD compared to token-level routing, both theoretically and empirically.

