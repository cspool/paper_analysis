# <span id="page-2-0"></span>2 Problem Setup

Given an LRM M, it can generate an extended thinking trajectory comprising N reasoning steps, denoted as t = t (1) , t (2) , . . . , t (N) , and the corresponding solution s for a given question q. This process can be represented as:

$$(\mathbf{t}, \mathbf{s}) := \mathcal{M}(\mathbf{q}). \tag{1}$$

However, as the number of reasoning steps N increases, the model may frequently switch its thinking modes. Sampling a complete thinking trajectory can hinder convergence of the solution estimation, thereby increasing the inference cost.

Our objective is to identify a subset t ′ ⊆ t that preserves the LRM's reasoning performance to the greatest extent possible. A more compact thinking trajectory t ′ is expected to exhibit a higher knowledge density across various computational budgets.

