# Reasoning Path Compression: Compressing Generation Trajectories for Efficient LLM Reasoning

Jiwon Song <sup>1</sup> Dongwon Jo <sup>1</sup> Yulhwa Kim<sup>2</sup> <sup>∗</sup> Jae-Joon Kim1<sup>∗</sup> <sup>1</sup> Seoul National University <sup>2</sup> Sungkyunkwan University {jiwon.song, dongwonjo, kimjaejoon}@snu.ac.kr {yulhwakim}@skku.edu

### Abstract

Recent reasoning-focused language models achieve high accuracy by generating lengthy intermediate reasoning paths before producing final answers. While this approach is effective in solving problems that require logical thinking, long reasoning paths significantly increase memory usage and reduce throughput of token generation, limiting the practical deployment of such models. We propose Reasoning Path Compression (RPC), a training-free method that accelerates inference by leveraging the semantic sparsity of reasoning paths. RPC periodically compresses the KV cache by retaining cache entries that receive high importance score, which are computed using a selector window composed of recently generated queries. Experiments show that RPC improves generation throughput of QwQ-32B by up to 1.60× compared to the inference with full KV cache, with an accuracy drop of 1.2% on the AIME 2024 benchmark. Our findings demonstrate that semantic sparsity in reasoning traces can be effectively exploited for compression, offering a practical path toward efficient deployment of reasoning LLMs. Our code is available at <https://github.com/jiwonsong-dev/ReasoningPathCompression>.

