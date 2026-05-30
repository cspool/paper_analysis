# <span id="page-12-1"></span>C.1. Dynamic Temporal Scope Retrieval (Sec. [4.4\)](#page-6-2)

To evaluate performance on dynamic temporal reasoning with WorldMM, we employ several approaches, including temporal grounding model, embedding-based retrieval models, hierarchical retrieval models, and keyframe selection method. For each method, we measure tIoU using ei-

<span id="page-13-3"></span><span id="page-13-1"></span>Table 7. Category-wise performance breakdown of WorldMM and baselines on EgoLifeQA, Ego-R1 Bench, HippoVlog, and LVBench.

| Model                |      |      | EgoL | ifeQA |      |      |      | E    | go-R1 | Beno | h    |      |             | H    | Iippo\ | Vlog  |      | LVBench |      |      |      |
|----------------------|------|------|------|-------|------|------|------|------|-------|------|------|------|-------------|------|--------|-------|------|---------|------|------|------|
| 1120401              | Ent. | EvR. | Hab. | Rel.  | Task | Avg. | Ent. | EvR. | Hab.  | Rel. | Task | Avg. | Aud.        | Vis. | A+V    | Summ. | Avg. | Short   | Med. | Long | Avg. |
| Base Models          |      |      |      |       |      |      |      |      |       |      |      |      |             |      |        |       |      |         |      |      |      |
| Qwen3-VL-8B [1]      | 35.2 | 30.2 | 39.3 | 46.4  | 46.0 | 38.6 | 31.8 | 41.5 | 38.5  | 42.1 | 44.7 | 35.7 | 73.6        | 74.0 | 69.2   | 80.8  | 74.4 | 48.8    | 44.4 | 53.4 | 48.3 |
| Gemini 2.5 Pro [4]   | 43.2 | 40.5 | 41.0 | 55.2  | 52.4 | 46.4 | 43.9 | 56.1 | 53.9  | 47.4 | 47.4 | 46.7 | 69.2        | 75.2 | 63.6   | 80.0  | 72.0 | 57.1    | 52.2 | 65.2 | 57.0 |
| GPT-5 [23]           | 47.2 | 42.1 | 47.5 | 53.6  | 55.6 | 48.6 | 41.8 | 58.5 | 53.9  | 52.6 | 50.0 | 46.3 | 73.6        | 75.6 | 69.2   | 84.4  | 75.7 | 59.1    | 59.1 | 69.1 | 60.4 |
| Long Video LLMs      |      |      |      |       |      |      |      |      |       |      |      |      |             |      |        |       |      |         |      |      |      |
| VideoChat-Flash [18] | 28.8 | 32.5 | 37.7 | 37.6  | 38.1 | 34.2 | 43.4 | 43.9 | 38.5  | 31.6 | 44.7 | 42.7 | 60.8        | 59.2 | 56.4   | 55.6  | 58.0 | 34.9    | 23.1 | 44.6 | 33.2 |
| Time-R1 [35]         | 39.2 | 50.8 | 65.6 | 48.8  | 47.6 | 48.8 | 49.2 | 48.8 | 46.2  | 42.1 | 44.7 | 48.0 | 58.2        | 58.2 | 49.4   | 52.4  | 54.6 | 32.1    | 23.6 | 40.2 | 31.1 |
| Video-RTS [36]       | 40.8 | 48.4 | 62.3 | 48.8  | 47.6 | 48.2 | 47.6 | 46.3 | 53.9  | 52.6 | 47.4 | 48.0 | 58.8        | 62.0 | 56.8   | 58.4  | 59.0 | 43.4    | 25.7 | 49.5 | 39.8 |
| RAG-based Video LL   | Ms   |      |      |       |      |      |      |      |       |      |      |      |             |      |        |       |      |         |      |      |      |
| LightRAG [10]        | 40.8 | 48.4 | 67.2 | 50.4  | 44.4 | 48.8 | 54.0 | 61.0 | 46.2  | 42.1 | 42.1 | 52.3 | 51.6        | 46.0 | 44.8   | 47.2  | 47.4 | 30.2    | 28.6 | 34.3 | 30.4 |
| HippoRAG [11]        | 48.8 | 60.3 | 70.5 | 60.8  | 66.7 | 59.6 | 54.5 | 65.9 | 69.2  | 52.6 | 50.0 | 56.0 | 72.4        | 53.2 | 54.0   | 73.2  | 63.2 | 54.9    | 47.5 | 62.3 | 54.0 |
| Video-RAG [21]       | 49.6 | 56.3 | 67.2 | 55.2  | 54.0 | 55.4 | 48.7 | 58.5 | 53.9  | 47.4 | 44.7 | 49.7 | 63.2        | 64.8 | 63.6   | 68.8  | 65.1 | 32.9    | 30.2 | 39.7 | 33.1 |
| Memory-based Video   | LLMs |      |      |       |      |      |      |      |       |      |      |      |             |      |        |       |      |         |      |      |      |
| EgoRAG [41]          | 40.0 | 56.3 | 62.3 | 54.4  | 52.4 | 52.0 | 46.6 | 56.1 | 46.2  | 47.4 | 55.3 | 49.0 | 64.8        | 53.2 | 47.6   | 64.4  | 57.5 | 32.4    | 32.0 | 31.9 | 32.2 |
| Ego-R1 [30]          | 51.2 | 53.2 | 63.9 | 50.4  | 50.8 | 53.0 | 50.8 | 63.4 | 38.5  | 36.8 | 57.9 | 52.0 | 57.2        | 58.8 | 52.0   | 67.2  | 58.8 | 32.5    | 36.5 | 37.3 | 34.1 |
| HippoMM [19]         | 45.6 | 53.2 | 70.5 | 55.2  | 58.7 | 54.6 | 51.9 | 56.1 | 46.2  | 52.6 | 57.9 | 53.0 | 68.8        | 77.6 | 59.2   | 82.0  | 71.9 | 40.7    | 33.3 | 35.8 | 38.2 |
| M3-Agent [20]        | 44.4 | 54.8 | 62.3 | 56.8  | 54.0 | 53.5 | 52.4 | 58.5 | 38.5  | 42.1 | 52.6 | 52.0 | 68.4        | 72.4 | 50.8   | 70.4  | 65.5 | 53.0    | 40.7 | 48.5 | 49.3 |
| WorldMM (Ours)       |      |      |      |       |      |      |      |      |       |      |      |      |             |      |        |       |      |         |      |      |      |
| WorldMM-8B           | 49.6 | 56.4 | 63.9 | 58.4  | 58.7 | 56.4 | 48.2 | 63.4 | 53.9  | 52.6 | 57.9 | 52.0 | 69.6        | 73.6 | 65.2   | 70.4  | 69.7 | 55.0    | 54.1 | 59.8 | 55.4 |
| WorldMM-GPT          | 62.4 | 64.3 | 75.4 | 62.4  | 71.4 | 65.6 | 64.6 | 70.7 | 76.9  | 57.9 | 63.2 | 65.3 | <b>75.6</b> | 81.6 | 73.2   | 82.8  | 78.3 | 58.3    | 65.4 | 72.1 | 61.9 |

ther the returned timestamps or the timestamps of the selected content. For the temporal grounding model, we use Time-R1 [35], with a slightly modified prompt that enables it to return both the evidence timestamps and the corresponding grounded responses. We sample videos at 0.5 fps and provide up to 768 frames. For embedding-based and hierarchical retrieval models, we follow the configurations described in Sec. B.1. Additionally, we include Qwen3 Emb., which applies the Qwen3-Embedding-4B [46] text encoder for caption retrieval, and InternVideo2, which encodes each segment using InternVideo2 [34] as an video encoder with uniform 16 frame averaging to enable segmentlevel retrieval. Both methods retrieve 30 second segments based on similarity search. For key frame selection, we apply AKS [29], which selects keyframes from the 0.5 fps sampled sequence. For tIoU evaluation, we interpret frames as representing their corresponding 30 second segments.

#### <span id="page-13-0"></span>C.2. Efficacy of Memory Modules (Sec. 4.7)

To assess the contribution of each component within WorldMM's multimodal memory system, we evaluate several ablated variants in Sec. 4.7. In this section, we detail each variant of WorldMM created by selectively disabling a specific component. For episodic memory variants, we first construct a **fixed timescale** variant by replacing hierarchical episodic memory with a single fixed timescale memory. Specifically, we use the episodic memory of the finest granularity timescale. We also experiment an **embedding retrieval** variant in which the model's graph-based episodic retrieval is replaced with an embedding-based similarity

search using Qwen-Embedding-4B. To examine the effect of semantic consolidation, we use a **w/o consolidation** version that bypasses the consolidation procedure to update the memory and instead store the raw extracted triplets without any update to existing memory. Finally, for visual memory, we ablate components of dual-retrieval mechanism by evaluating systems that rely exclusively on either **feature retrieval** through natural-language keyword search or **timestamp retrieval** based purely on temporal indices.

