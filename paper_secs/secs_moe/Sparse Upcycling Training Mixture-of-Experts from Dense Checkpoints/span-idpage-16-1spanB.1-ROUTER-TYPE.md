# <span id="page-16-1"></span>B.1 ROUTER TYPE

While our default upcycling recipe uses Expert Choice routing [Zhou et al.](#page-13-1) [\(2022\)](#page-13-1) (in the encoder), the same recipe can be applied to other routing mechanisms. Here, we compare with Top-K routing [Shazeer et al.](#page-12-4) [\(2017\)](#page-12-4), which is a very popular alternative. Table [2](#page-17-2) shows that, for vision, sparse upcycling with Top-K routing works comparably well to Expert Choice, on a per step basis, provided we also use Batch Priority Routing (BPR) [\(Riquelme et al., 2021\)](#page-11-6). BPR sorts tokens according to a model confidence proxy so that –when experts are full– high confidence tokens are given priority. We suspect this may be helpful right at the beginning, when applying the upcycling, to avoid discarding important tokens. Expert Choice avoids this problem by design, as experts are always balanced and select the most 'relevant' tokens.

<span id="page-17-2"></span>Table 2: Sparse Upcycling on L/32 vision models with Expert Choice and Top-K routing (also known as Top-K). K refers to the number of selected experts per token, while C refers to the capacity factor. Notice that with Expert Choice routing, each token chooses C experts on average. The initial dense checkpoint was trained for 7 epochs. Note that these comparison are on a per-step basis, and that Expert Choice upcycled models are actually slightly faster than Top-K models; see Figure 8.

| Model         | Capacity | From    | Extra Epochs | Val Prec@1 | ImageNet 10shot |
|---------------|----------|---------|--------------|------------|-----------------|
| Dense         | _        | Dense   | 7            | 49.60      | 73.59           |
| Expert Choice | C = 1    | Dense   | 7            | 51.91      | 74.04           |
| Top-K         | K = 1    | Dense   | 7            | 51.51      | 74.40           |
| Expert Choice | C = 2    | Dense   | 7            | 52.80      | 74.83           |
| Top-K         | K = 2    | Dense   | 7            | 52.88      | 74.91           |
| Expert Choice | C = 1    | Scratch | 7            | 50.42      | 72.95           |
| Expert Choice | C = 2    | Scratch | 7            | 51.28      | 74.01           |
| Expert Choice | C = 1    | Scratch | 14           | 54.84      | 75.02           |
| Expert Choice | C = 2    | Scratch | 14           | 55.46      | 75.75           |

<span id="page-17-3"></span>![](_page_17_Figure_3.jpeg)

Figure 8: Comparison of Expert Choice, Top-2 and Switch (Top-1) routing mechanisms for a Base upcycled language model.

For language, similar ablations (Figure 8) shows that Expert Choice routing outperforms both Top-2 routing (with BPR) and switch (Top-1) routing, on a train time basis.

