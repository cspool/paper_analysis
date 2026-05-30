# E.1 EFFECT OF COMPRESSION MODULE

To verify the effectiveness of the compression module, we compared the compression module in LLaVA-Mini with previous advanced token merging methods. To ensure a fair comparison of

<span id="page-20-0"></span>

| Table 10: Comparison of LLaVA-Mini with previous token merging methods. |
|-------------------------------------------------------------------------|
|-------------------------------------------------------------------------|

|            |                |       | Performance |      |
|------------|----------------|-------|-------------|------|
| Methods    | #Vision Tokens | VQAv2 | GQA         | MMB  |
| MQT-LLaVA  | 2              | 61.0  | 50.8        | 54.4 |
| MQT-LLaVA  | 36             | 73.7  | 58.8        | 63.4 |
| MQT-LLaVA  | 256            | 76.8  | 61.6        | 64.3 |
| PruMerge   | 32             | 72.0  | -           | 60.9 |
| PruMerge++ | 144            | 76.8  | -           | 64.9 |
| LLaVA-Mini | 1              | 72.4  | 54.2        | 57.7 |
| LLaVA-Mini | 16             | 74.1  | 55.4        | 59.2 |
| LLaVA-Mini | 64             | 75.3  | 56.7        | 62.1 |
| LLaVA-Mini | 144            | 76.9  | 58.9        | 64.9 |

token compression performance, we remove the modality pre-fusion module from LLaVA-Mini for the comparison with SOTA token merging methods, including PruMerge [\(Shang et al., 2024\)](#page-14-7), PruMerge++ [\(Shang et al., 2024\)](#page-14-7), and MQT-LLaVA [\(Hu et al., 2024\)](#page-12-0). Specifically, PruMerge applies the widely-used token merge (ToMe) technique [\(Bolya et al., 2023\)](#page-10-3) on ViT, PruMerge++ improves upon PruMerge by uniformly sampling additional vision tokens, and MQT-LLaVA employs Matryoshka representation learning to compress vision tokens.

As shown in Table [10,](#page-20-0) LLaVA-Mini's compression module outperforms PruMerge, PruMerge++, and MQT-LLaVA at the same compression rate, showing the advantages of query-based compression.

