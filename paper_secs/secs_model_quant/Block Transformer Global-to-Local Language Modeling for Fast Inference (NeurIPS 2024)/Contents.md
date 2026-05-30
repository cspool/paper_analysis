# Contents

| 1 |     | Introduction                                                       | 1  |
|---|-----|--------------------------------------------------------------------|----|
| 2 |     | Block Transformer                                                  | 3  |
|   | 2.1 | A primer on the key bottlenecks of autoregressive transformers     | 3  |
|   | 2.2 | Embedder                                                           | 4  |
|   | 2.3 | Block decoder                                                      | 4  |
|   | 2.4 | Token decoder                                                      | 4  |
| 3 |     | Experiments                                                        | 5  |
|   | 3.1 | Experimental setup<br>                                             | 5  |
|   | 3.2 | Main results                                                       | 5  |
|   | 3.3 | Analysis on parameter allocation ratio and block length            | 6  |
|   | 3.4 | Ablation on components of the Block Transformer<br>                | 7  |
|   | 3.5 | Analysis on global-to-local language modeling<br>                  | 8  |
|   | 3.6 | IsoFLOP analysis under inference throughput constraints            | 8  |
|   | 3.7 | Uptraining from vanilla transformers                               | 8  |
|   | 3.8 | Comparison to related works<br>                                    | 9  |
| 4 |     | Discussion                                                         | 9  |
|   | 4.1 | Contextual information encapsulated in context block embedding     | 9  |
|   | 4.2 | Techniques for further throughput improvement                      | 9  |
| 5 |     | Related work                                                       | 10 |
| 6 |     | Conclusion                                                         | 10 |
| A |     | Limitations                                                        | 20 |
| B |     | Discussion and future works                                        | 20 |
|   | B.1 | Optimizing hyperparameters for parameters or FLOPs                 | 20 |
|   | B.2 | Densification of the block decoder with longer block embedding<br> | 20 |
|   | B.3 | Relieving the locality of the token decoder for performance gains  | 20 |
|   | B.4 | Further scaling and advanced uptraining schemes<br>                | 21 |
|   | B.5 | Adaptive block lengths for dynamic compute allocation              | 21 |
| C |     | Broader impact                                                     | 21 |
| D |     | Extended related work                                              | 21 |
|   | D.1 | KV cache compression<br>                                           | 21 |
|   | D.2 | Architectural for optimizations of KV cache                        | 22 |
| E |     | Analysis on the inference efficiency of Block Transformer          | 22 |
|   | E.1 | Background: inference stages and principal bottlenecks             | 22 |
|   | E.2 | Inference-time advantages of block and token decoders<br>          | 22 |

| F |     | Architectural details                                                | 24 |
|---|-----|----------------------------------------------------------------------|----|
|   | F.1 | Embedder methods<br>                                                 | 24 |
|   | F.2 | Token decoder methods                                                | 24 |
| G |     | Experimental settings                                                | 24 |
|   | G.1 | Overall settings<br>                                                 | 24 |
|   | G.2 | Model sizes and hyperparameters                                      | 25 |
|   | G.3 | Settings for Section 3.2<br>                                         | 25 |
|   | G.4 | Settings for Section 3.3<br>                                         | 25 |
|   | G.5 | Settings for Section 3.4<br>                                         | 25 |
|   | G.6 | Settings for Section 3.5<br>                                         | 25 |
|   | G.7 | Settings for Section 3.6<br>                                         | 26 |
|   | G.8 | Settings for Section 3.7<br>                                         | 26 |
|   | G.9 | Settings for Section 3.8<br>                                         | 26 |
| H |     | Random length padding during pre-training                            | 26 |
| I |     | Throughput Comparison with FlashDecoding                             | 27 |
| J |     | Pareto frontiers at variable batch sizes and context lengths         | 28 |
| K |     | Position-wise loss by parameter allocation ratio                     | 29 |
| L |     | Loss trend by allocation ratio and block length                      | 29 |
|   |     | M Pareto frontier of throughput by allocation ratio and block length | 30 |
| N |     | Ablation studies on components of Block Transformer                  | 32 |
|   | N.1 | Embedder design<br>                                                  | 32 |
|   | N.2 | Token decoder design                                                 | 32 |
| O |     | Long-context modeling ability                                        | 33 |
| P |     | Uptraining strategy for training efficiency                          | 34 |
| Q |     | Performance comparison to MEGABYTE                                   | 35 |
| R |     | Visualization of attention scores in Block Transformer               | 35 |
| S |     | Analysis on the context block embedding                              | 37 |

## <span id="page-19-0"></span>**A** Limitations

The Block Transformer variants considered in our study require more parameters and FLOPs compared to their perplexity-equivalent vanilla models. Despite higher parameter and FLOP requirements, our Block Transformers achieve higher inference throughput, owing to low memory overhead and omission of prefill in the token decoder. However, this advantage is diminished during training—resulting in higher wall-time training costs compared to vanilla Transformers. The large parameter requirements also hinder the applicability of Block Transformers in situations with hard memory constraints such as on-device usage. We note that these are partially a result of our focus on inference throughput, rather than architectural limitations. There are many promising avenues to minimize parameter and FLOP (training cost) requirements, with minor adjustments to the architecture or hyperparameters. In the following section, we discuss several of these for future work.

#### <span id="page-19-1"></span>**B** Discussion and future works

#### <span id="page-19-2"></span>**B.1** Optimizing hyperparameters for parameters or FLOPs

We can optimize the hyperparameters of the Block Transformer architecture to minimize parameter or FLOP requirements, as opposed to inference throughput as in our main experiments. Frist, we can reduce the *block length* to enhance performance while maintaining the same parameter count. Our ablations on block length demonstrate that a shorter block length can significantly improve perplexity, while compromising inference throughput with increased FLOPs in the block decoder. Thus, to achieve comparable perplexity, we can utilize less parameters, which offsets the decreased throughput resulting from the shortened block length.

Secondly, we find that increasing the proportion of the block decoder can significantly reduce FLOP requirements with minor degradation in performance, due to the FLOP-intensive nature of the token decoder. However, this comes at the cost of increased inference wall-time due to the KV cache bottlenecks of the block decoder. Further experimentation is needed to precisely identify the tradeoffs associated with these hyperparameter choices with respect to various cost metrics.

## <span id="page-19-3"></span>B.2 Densification of the block decoder with longer block embedding

Another approach to improving the performance of Block Transformers without extra parameters would be through better utilization of those already in the block decoder, i.e., by passing more tokens through them. We could do this by representing a single block with a longer input block embedding, say  $L_B$ , instead of one. Let's call these *subblock tokens*. During a single decoding step,  $L_B$  input tokens would be projected into  $L_B$  subblock tokens. Then, these would be passed to the block decoder and forwarded in parallel.

This would effectively preserve the computational width [34] of the block decoder, i.e., the total embedding dimension of the inputs, to be equivalent to a vanilla Transformer of the same width and depth. The minor difference in perplexity between the vanilla Transformer and Block Transformer with  $L_B=1$  in Figure 4a suggests that Block Transformers could approach the performance of same-sized vanilla transformers when the computational width of the block decoder is the same.

While this would require the same FLOPs as a vanilla Transformer, we can expect roughly  $L_B$  times reduction in decoding wall-time due to parallel execution—since parameters and previous KV cache would only need to be fetched once per block, instead of once per input token. Note that total KV cache storage would be the same as vanilla Transformers since the number of input tokens and subblock tokens would be the same (this is why we expect  $L_B$  reduction in KV cache IO rather than  $L_B^2$  as in our original block decoder).

#### <span id="page-19-4"></span>B.3 Relieving the locality of the token decoder for performance gains

In our experiments, we bottleneck the global information passed to the token decoder into a single context embedding. This is done for simplicity and to highlight the viability of global-to-local modeling, where the local module has limited access to global context. However, we posit that the token decoder can benefit from performance gains with minimal extra costs by relieving this rather extreme limitation.

It is possible use additional context embeddings in the token decoder to facilitate the propagation of context information, as discussed in [Section 3.8.](#page-8-5) Instead of projecting only the last output block embedding to the token decoder, we could utilize a small window of previous output block embeddings. This could resolve the rise in perplexity in later positions in the token decoder due to insufficient context information, with only slight increase in FLOPs and KV cache overhead in the token decoder.

## <span id="page-20-1"></span>B.4 Further scaling and advanced uptraining schemes

The scale of experiments in our paper is relatively small compared to even previous-generation frontier models [\[14,](#page-11-9) [20\]](#page-11-10). While our experiments show that the inference throughput benefits of Block Transformers scale positively across two orders of magnitude, further experiments are required to verify this beyond 1 billion parameters.

We can consider uptraining as a cost-effective training approach for this analysis, which effectively utilizes existing pretrained vanilla transformers to minimize the training costs of Block Transformers. For example, we can consider a progressive adaptation approach where a vanilla transformer is first adapted to a Block Transformer with block length 1, to maximize compatability, and then progressively trained with larger block lengths. Moreover, instead of simply splitting the layers of a pretrained vanilla transformer to initialize the block and token decoders, exploring weight initialization methods like averaging the layers or identifying weights that produces similar activations could significantly enhance performance.

## <span id="page-20-2"></span>B.5 Adaptive block lengths for dynamic compute allocation

What if we can dynamically allocate computation to generate 'easy' tokens faster but ponder longer on 'hard' tokens? This has been the central question of several previous works on dynamic compute allocation [\[35,](#page-12-9) [68,](#page-15-10) [4,](#page-10-6) [65\]](#page-14-7). The multiscale nature of the Block Transformer architecture offers a novel avenue to achieving this in autoregressive language models–by dynamically setting the input and output length of blocks based on the 'difficulty' of its contents. For the embedder and token decoders, we can use our CLS-token and prefix token based designs respectively, and padding can be used to maintain static computation during training. A challenge remains in training the model to dynamically determine optimal input *and* output block lengths.

