# C.4 Evaluation on Complex Multi-hop Reasoning

To further assess EXIT's ability to handle complex reasoning tasks, we evaluated it on the MuSiQue dataset (Trivedi et al., 2022), which contains multihop questions requiring deeper reasoning (3–4)

<span id="page-18-1"></span>Table 11: Evaluation of EXIT and baselines on the MuSiQue dataset on Llama3-8B reader.

| Method        | EM  | F1   | Latency (s) |
|---------------|-----|------|-------------|
| Original Docs | 3.8 | 10.3 | 1.1         |
| RECOMP-Abst   | 4.5 | 11.8 | 2.5         |
| CompAct       | 4.1 | 11.0 | 8.5         |
| Refiner       | 4.5 | 11.3 | 7.6         |
| RECOMP-Extr   | 4.3 | 10.1 | 0.6         |
| LongLLMLingua | 4.0 | 10.7 | 0.9         |
| EXIT (Ours)   | 4.6 | 11.3 | 0.9         |

hops). While our main experiments focused on standard benchmarks like HQA and 2WIKI, MuSiQue provides a valuable test of generalization under more challenging conditions.

As shown in Table [11,](#page-18-1) EXIT achieves competitive performance on MuSiQue. Notably, it maintains low latency while matching or outperforming other extractive and abstractive baselines in EM and F1 scores. These results demonstrate EXIT's robustness in handling longer reasoning chains without incurring the high latency costs typically associated with abstractive methods.

These findings reinforce EXIT's ability to scale to more complex, multi-hop QA settings while maintaining an efficient trade-off between latency and performance.

