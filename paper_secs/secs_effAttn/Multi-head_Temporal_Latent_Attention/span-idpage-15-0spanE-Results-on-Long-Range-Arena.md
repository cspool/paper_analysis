# <span id="page-15-0"></span>E Results on Long Range Arena

<span id="page-15-1"></span>Table 8: Experimental results on Long-Range Arena benchmark [\[41\]](#page-12-11). The published results other than MTLA are taken from [\[41\]](#page-12-11).

| Model                 | Listops(↑) | Text(↑) | Retrieval(↑) | Image(↑) | Pathfinder(↑) | Avg(↑) |
|-----------------------|------------|---------|--------------|----------|---------------|--------|
| Transformer           | 36.37      | 64.27   | 57.46        | 42.44    | 71.40         | 54.39  |
| Local Attention       | 15.82      | 52.98   | 53.39        | 41.46    | 66.63         | 46.06  |
| Sparse Transformers   | 17.07      | 63.58   | 59.59        | 44.24    | 71.71         | 51.24  |
| Longformer            | 35.63      | 62.85   | 56.89        | 42.22    | 69.71         | 53.46  |
| Linformer             | 35.70      | 53.94   | 52.27        | 38.56    | 76.34         | 51.36  |
| Reformer              | 37.27      | 56.10   | 53.40        | 38.07    | 68.50         | 50.67  |
| Sinkhorn Transformers | 33.67      | 61.20   | 53.83        | 41.23    | 67.45         | 51.39  |
| Synthesizer           | 36.99      | 61.68   | 54.67        | 41.61    | 69.45         | 52.88  |
| BigBird               | 36.05      | 64.02   | 59.29        | 40.83    | 74.87         | 55.01  |
| Linear Transformer    | 16.13      | 65.90   | 53.09        | 42.34    | 75.30         | 50.55  |
| Performer             | 18.01      | 65.40   | 53.82        | 42.77    | 77.05         | 51.41  |
| Proposed MTLA         | 40.47      | 66.99   | 59.88        | 48.10    | 67.55         | 56.60  |

This section further evaluates MTLA on the Long-Range Arena (LRA) benchmark [\[41\]](#page-12-11). The LRA benchmark was not included in the main text, as it is not primarily designed for decoderonly architectures and is therefore more suitable for evaluating encoder self-attention mechanisms. As shown in Table [8,](#page-15-1) MTLA achieves strong performance on the LRA benchmark compared to other attention mechanisms. These results complement the findings presented in the main text, demonstrating that MTLA consistently performs well across diverse modalities, including text, speech, and vision. The consistent performance across tasks indicates that MTLA effectively captures long-range dependencies while maintaining computational efficiency. It is worth noting that state space models (SSMs) are known to outperform Transformer-based attention mechanisms on the LRA benchmark due to their formulation as long convolutions with time-decay dynamics, which are particularly advantageous for tasks with strong positional dependencies. Nevertheless, we include Table [8](#page-15-1) because LRA remains a challenging and well-established benchmark for attention mechanisms, providing a valuable test of MTLA's capability under difficult long-context conditions.

