# <span id="page-16-2"></span>B Additional Probing Details

#### <span id="page-16-1"></span>B.1 Activation Head Selection

For efficient context cutoff, our method does not use activations from all layers of the model, but rather selectively identifies the most informative attention heads in specific layers through probing. The activation selection process works as follows:

- We initially probe all attention heads across all layers of the model to identify which ones encode the strongest sufficiency signals.
- As shown in Figure [3](#page-3-1) for LLaMA3.2-1B and Figure [8](#page-17-0) for Qwen2.5-14B, we discovered that a subset of heads, primarily from middle layers, exhibit significantly higher predictive performance for context sufficiency. This aligns with findings in other interpretability work that middle layers often capture higher-level semantic information.
- After identifying these predictive heads, we select only the top-k heads with the highest F1 scores on the validation set (k = 5 in our implementation) to train our ensemble classifier.

As demonstrated in Table [9,](#page-20-1) we found that using just the top 5 attention heads yields the best performance, with minimal gains or even decreased performance when more heads are included. This confirms our hypothesis that context sufficiency signals are concentrated in specific architectural components rather than distributed throughout the entire model.

The specific layers used can vary across model architectures - we don't restrict our approach to predetermined layers, but rather let the probing results guide which heads (and consequently which layers) provide the most reliable sufficiency signals. This approach ensures our classifier focuses only on the most informative components of the model's internal representations while keeping computational overhead minimal.

#### <span id="page-16-3"></span>B.2 Additional Probing Results

Figure [8](#page-17-0) illustrates the probing results for the Qwen2.5 14B model, revealing that, similar to LLaMA models, the highest F1 scores are concentrated in the middle layers. However, the distribution of these high-performing heads differs between the two model families. While both models exhibit darker regions indicating stronger sufficiency signals in their intermediate layers, LLaMA3.2-1B shows a more dispersed pattern of high F1 scores across various heads within these layers. This suggests that although both LLaMA and Qwen models tend to encode context sufficiency signals primarily in their middle layers, the specific attention heads responsible and their activation patterns vary between architectures.

#### <span id="page-16-0"></span>B.3 Left-to-Right Context Processing

Our choice of left-to-right processing is motivated by two main factors. First, Transformer models are typically trained on left-to-right sequences, making this order naturally compatible with their internal representations. This avoids the need for significant architectural changes or retraining. Second, it enables efficient use of the key-value (KV) cache while preserving semantic consistency. As described in [§2.3,](#page-3-2) left-to-right processing allows us to reuse cached activations from previous

> **[图片提取文字 (无描述)]:**
> 0.84 -0.820.80 -0.78-0.76 -0.74-0.72Head (Sorted)
![](_page_17_Figure_0.jpeg)

<span id="page-17-0"></span>Figure 8: Probing results for the Qwen2.5 14B model. The heatmap shows the average F1 score for each head across all layers, which is different from LLaMA models.

chunks, maintaining contextual coherence across the sequence. As shown in Figure [5,](#page-7-2) the model's confidence in sufficiency predictions increases steadily as more context is processed, suggesting that meaningful information accumulates effectively under left-to-right processing.

Although alternative processing orders (e.g., reversed or random) or methods (e.g., RAG, which selects arbitrary subsets of context) are possible, our approach preserves semantic continuity between chunks. We leave the investigation of these alternatives for future work.

Note that when processing a chunk, we retain the KV cache from all preceding chunks. This means there is no computational difference between processing the context chunk-by-chunk from left to right and processing the entire context in a single pass. We do not alter the computation over the context; we simply segment it into chunks to determine when to stop. Each token receives exactly the same context as it would without chunking, due to reuse of the KV cache.

