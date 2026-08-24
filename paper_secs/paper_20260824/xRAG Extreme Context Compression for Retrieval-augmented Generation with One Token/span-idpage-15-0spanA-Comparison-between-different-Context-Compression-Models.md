# <span id="page-15-0"></span>**A** Comparison between different Context Compression Models

In Table 5, we present a detailed comparison of various context compression models, emphasizing their real-world applicability. This comparison focuses on two key aspects: (1) Plug-and-play capability, which assesses whether dataset-specific tuning is necessary for new, unseen data; (2) Memory efficiency, which evaluates if additional memory space is required to store the compressed information, such as high-dimensional vectors typically used in soft prompting methods.

<span id="page-15-2"></span>

| Model                  | Specifically designed for RAG | Maximun<br>Compression Rate | Approach            | Plug-and-Play | Memory Efficient |
|------------------------|-------------------------------|-----------------------------|---------------------|---------------|------------------|
| AutoCompressor [14]    | Х                             | x15                         | Soft Prompting      | Х             | Х                |
| Gist [58]              | X                             | x26                         | Soft Prompting      | X             | X                |
| ICAE [19]              | X                             | x8                          | Soft Prompting      | ✓             | X                |
| LLMLingua [28]         | X                             | x20                         | Prompt Editing      | ✓             | ✓                |
| Selective Context [45] | X                             | x5                          | Prompt Editing      | ✓             | ✓                |
| Token Elimination [85] | ✓                             | x10                         | Attention Filtering | ✓             | ✓                |
| FilCo [74]             | ✓                             | x2                          | Prompt Editing      | X             | ✓                |
| RECOMP [79]            | ✓                             | x16.6                       | Prompt Editing      | ✓             | ✓                |
| xRAG                   | ✓                             | x178                        | Modality Fusion     | ✓             | ✓                |

Table 5: Comparison between different compression methods from their setting to design principle.

