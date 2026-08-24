# **Memory Footprint and Inference Time**

For models that focus on long texts, aside from training costs, another critical aspect is the memory footprint and inference time. In this section, we compare FocusLLM with several previous longcontext methods capable of retaining global information by preserving the cache of all context: Standard (PI/NTK), LongLlama, and CEPE. As for models like Activation Beacon and StreamingLLM, although they maintain a constant memory footprint by only retaining cache for a fixed window, they suffer significant information loss and struggle with the precise understanding of long texts as demonstrated in Section 4.2. Therefore, they are not the primary subjects of comparison.

The results are shown in Figure 3 and Figure 4. FocusLLM with or without parallel indicates whether we process each chunk either concurrently or sequentially. The findings indicate that: (1) When ample memory resources are available, parallel processing is more efficient for FocusLLM. (2) Although FocusLLM splits long texts into numerous chunks, resulting in a slightly longer inference time compared to the standard approach, it still holds a significant advantage over other longcontext methods.

#### **5.4 Chunk Size**

We conduct an investigation into the impact of different chunk sizes on performance. In theory, larger chunk sizes, as long as they do not exceed the model's default context length (e.g., 4K for LLaMA-2), are preferable because they allow for processing the memory with a smaller number of forward passes. However, smaller chunk sizes may enable more precise processing.

In experiments, we maintain a total sequence length of 8K, testing the perplexity using different chunk sizes on the same samples of PG19. We select {256, 512, 1024, 2048} as our test sizes. The results are shown in Figure 5. We observe that there is no consistent trend in perplexity as the chunk size increases; it remains relatively stable. This confirms our hypothesis that we can employ larger chunk sizes on models with longer default context lengths (e.g. LLaMA-2-32K). We will explore this direction in our future work.

#### 5.5 Ablation Studies

We employ both Continuation Loss and Reconstruction Loss for the training of FocusLLM. The motivation behind this is to equip the model with the natural language modeling capability while also enhancing its ability to recover information. Ablation

<span id="page-7-0"></span>

|                                                    |                      | LongBench      |              | ∞-Bench        |                |                  |
|----------------------------------------------------|----------------------|----------------|--------------|----------------|----------------|------------------|
|                                                    | Hyper Params.        | NarrativeQA    | TREC         | Math.Find      | En.MC          | Retrieve.PassKey |
| FocusLLM                                           | (2K, 2K)             | 18.53          | 65.5         | 13.43          | 31.00          | 99.32            |
| Continuation Loss only<br>Reconstruction Loss only | (2K, 2K)<br>(2K, 2K) | 17.36<br>17.05 | 60.5<br>62.0 | 13.71<br>12.86 | 27.95<br>26.64 | 1.69<br>91.19    |
| Local Context Size ↓                               | (1K, 2K)             | 17.87          | 63.0         | 8.86           | 29.69          | 99.32            |

Table 4: Investigations into the training loss and local context size of FocusLLM. We present the results for representative tasks from LongBench and ∞-Bench. For instance, NarrativeQA belongs to Single-Doc QA, while TREC relates to Few-shot learning. The Hyper Params is denoted as (local context size, chunk size).

studies as detailed in Table [4,](#page-7-0) reveal that relying solely on the Continuation Loss enables the model to manage some tasks effectively. Nonetheless, for tasks with substantial dependencies on the preceding context, like HotpotQA and Retrieve.PassKey, the model's efficacy deteriorates. Similarly, while employing the Reconstruction Loss ensures accurate restatement of the preceding context, the lack of generalizability of generating new tokens leads to a considerable decrease in performance. Therefore, the combined use of both loss functions is crucial for enhancing the performance and generalizability of FocusLLM.

We also investigate how the local context size influences performance in the last row of Table [4.](#page-7-0) As we reduce the local context size from 3.5K to 1K, the performance of most tasks experiences a slight decline. This suggests that candidate tokens cannot fully replace the information within the context.

## 6 Related Work

