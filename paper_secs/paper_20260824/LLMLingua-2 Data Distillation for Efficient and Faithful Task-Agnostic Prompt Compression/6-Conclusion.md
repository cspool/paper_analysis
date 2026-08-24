# 6 Conclusion

This paper targets task-agnostic prompt compression for better generalizability and efficiency. In this paper, we identify the challenges encountered in existing methods and address them accordingly. We conduct extensive experiments and analysis on five benchmarks across different tasks and domains. Our model shows superiority over strong baselines in terms of performance and compression latency. We publicly release the dataset of text compression with no essential information loss in this paper.

## Limitations

Our text compression dataset was constructed using only training examples from MeetingBank, a dataset of summarization over meeting transcripts. This raises concerns about the generalization ability of our compressor. Here we discuss this question

from two perspectives.

Firstly, we have conducted extensive out-ofdomain evaluation on four benchmarks in the paper, including LongBench [\(Bai et al.,](#page-8-1) [2023\)](#page-8-1), Zero-SCROLLS [\(Shaham et al.,](#page-10-14) [2023\)](#page-10-14), GSM8K [\(Cobbe](#page-9-17) [et al.,](#page-9-17) [2021\)](#page-9-17), and Big Bench Hard (BBH) [\(bench au](#page-9-18)[thors,](#page-9-18) [2023\)](#page-9-18), which cover multiple tasks from document QA to math problems and in-context learning. The experimental results show that even our LLMLingua-2-small model that is of BERT-base size achieves superior performance than the two LLaMA-2-7B based baselines Selective-Context [\(Li et al.,](#page-9-6) [2023\)](#page-9-6) and LLMLingua [\(Jiang et al.,](#page-9-5) [2023a\)](#page-9-5). This demonstrates that our learned prompt compression model has good generalization ability to data from different domains.

Secondly, we expand the constructed text compression dataset using 50k examples from TriviaQA-wiki. Then train an LLMLingua-2 compressor with the expanded dataset to see whether there would be further performance gain. Table [6](#page-8-2) shows the results under the 2,000-token constraint. We can see that training the compressor with more data does bring further performance gain (LLMLingua-2‡ ). However, the improvement seems not that significant. We conjecture that this is because although the semantics of texts from different domains may vary a lot, their redundancy pattern might be similar. Such pattern or knowledge may be learned during in-domain training, and then act as an anchor that can transfer across different domains. We leave this for future work.

