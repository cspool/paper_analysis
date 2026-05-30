# A Why Less Attention Produces Better Quality

The fact that Focus surpasses full attention—rather than merely approximating it—requires explanation. Three mechanisms contribute:

- 1. Softmax dilution. In full attention, softmax distributes probability mass across all n tokens, even when only a small subset is relevant. A pronoun at position 800 seeking its antecedent at position 200 must compete with hundreds of irrelevant distant tokens for attention weight. Focus restricts softmax to same-group tokens plus the local window, concentrating probability mass on a smaller, more relevant candidate set. The result is sharper, more informative attention distributions.
- 2. Noise removal. Irrelevant attention pairs do not merely waste computation—they actively degrade quality. Each irrelevant key–value pair contributes a small amount of noise to the attention output. Across 12 layers and 12 heads, this noise accumulates. Focus eliminates these pairs entirely: the model never computes attention over tokens it should ignore.
- 3. Implicit structural constraint. Full attention at 124M scale can memorize spurious long-range correlations in the training data. Restricting attention to semantically coherent groups acts as a structural prior—analogous to how L<sup>1</sup> penalties zero irrelevant features or dropout removes random connections. The restriction prevents the model from fitting noise in the attention pattern, without any explicit penalty term.

The key insight: full n <sup>2</sup> attention is not the performance ceiling—it is the unconstrained baseline. Learned sparsity improves upon it for the same reason that feature selection improves upon using all features: removing noise is not a cost, it is a benefit.

