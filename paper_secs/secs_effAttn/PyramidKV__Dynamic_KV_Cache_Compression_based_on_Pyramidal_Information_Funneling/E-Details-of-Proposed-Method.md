# **E Details of Proposed Method**

Based on the pyramidal information funneling observed across different layers, PyramidKV consists of two steps: (1) Dynamically allocating different KV cache sizes/budgets across different layers; and (2) Selecting important KV vectors in each attention head for caching as [Figure 7.](#page-15-0)

Our decision to use an arithmetic sequence is driven by three key factors:

- **Alignment with Pyramidal Information Funneling Pattern**: Empirical observations reveal a pyramidal information funneling pattern, where lower layers exhibit dispersed attention while higher layers concentrate on fewer tokens. Inspired by this, we adopt the arithmetic sequence design to align with this natural progression.
- **Superior Empirical Performance**: Through extensive experimentation across diverse datasets, we compared various methods, including the arithmetic sequence and adaptive approaches. Results consistently showed that the arithmetic sequence method outperformed others.
- **Computational Efficiency**: The arithmetic sequence method introduces minimal computational overhead compared to adaptive approaches, which require dynamically computing cache budgets across layers.

To perform KV cache eviction, we use torch.gather. Below, we outline the memory allocation and release process of torch.gather:

- **Index Selection**: Identify the positions of the elements to extract from the input tensor.
- **Memory Location Calculation**: Compute the specific memory locations of the elements to be extracted using the strides of the input tensor across each dimension.

- Output Tensor Creation: Allocate memory to create a new output tensor and copy the selected elements to their corresponding positions in the output tensor.
- **Memory Management**: Since torch gather is not an in-place operation, it creates a new tensor to store the results, while the memory of the original input tensor is released.

The speed-up offered by PyramidKV is complementary to that achieved through tensor parallelism and pipeline parallelism, as these approaches are not mutually exclusive. PyramidKV can be seamlessly integrated with both tensor parallelism and pipeline parallelism.

### <span id="page-16-0"></span>F Details of Evaluation

We use LongBench (Bai et al., 2023) to assess the performance of PyramidKV on tasks involving long-context inputs. LongBench is a meticulously designed benchmark suite that tests the capabilities of language models in handling extended documents and complex information sequences. This benchmark was created for multi-task evaluation of long context inputs.

We present the details of metrics, language and data for LongBench at Table 3. We run all the experiments on NVIDIA A100.

| Dataset                                                     | Source                                     | Avg len                  | Metric                          | Language                      | #data             |
|-------------------------------------------------------------|--------------------------------------------|--------------------------|---------------------------------|-------------------------------|-------------------|
| Single-Document QA NarrativeQA Qasper MultiFieldQA-en       | Literature, Film<br>Science<br>Multi-field | 18,409<br>3,619<br>4,559 | F1<br>F1<br>F1                  | English<br>English<br>English | 200<br>200<br>150 |
| Multi-Document QA<br>HotpotQA<br>2WikiMultihopQA<br>MuSiQue | Wikipedia<br>Wikipedia<br>Wikipedia        | 9,151<br>4,887<br>11,214 | F1<br>F1<br>F1                  | English<br>English<br>English | 200<br>200<br>200 |
| Summarization<br>GovReport<br>QMSum<br>MultiNews            | Government report<br>Meeting<br>News       | 8,734<br>10,614<br>2,113 | Rouge-L<br>Rouge-L<br>Rouge-L   | English<br>English<br>English | 200<br>200<br>200 |
| Few-shot Learning<br>TREC<br>TriviaQA<br>SAMSum             | Web question<br>Wikipedia, Web<br>Dialogue | 5,177<br>8,209<br>6,258  | Accuracy (CLS)<br>F1<br>Rouge-L | English<br>English<br>English | 200<br>200<br>200 |
| Synthetic Task<br>PassageCount<br>PassageRetrieval-en       | Wikipedia<br>Wikipedia                     | 11,141<br>9,289          | Accuracy (EM)<br>Accuracy (EM)  | English<br>English            | 200<br>200        |
| Code Completion<br>LCC<br>RepoBench-P                       | Github<br>Github repository                | 1,235<br>4,206           | Edit Sim<br>Edit Sim            | Python/C#/Java<br>Python/Java | 500<br>500        |

<span id="page-16-1"></span>Table 3: An overview of the dataset statistics in LongBench (Bai et al., 2023). 'Source' denotes the origin of the context. 'Accuracy (CLS)' refers to classification accuracy, while 'Accuracy (EM)' refers to exact match accuracy.

### G License

LongBench: MIT

### H Handle Rotary Embedding after Tokens are Removed in PyramidKV

We keep the rotary embedding unchanged after tokens are removed, so that LLMs can still capture the exact position information even if the tokens are removed. StreamingLLM (Xiao et al., 2023) shows that rolling ky cache with the correct relative position is crucial for maintaining performance. This is because StreamingLLM is designed to mainly handle unlimited context sizes, where contexts exceed the LLM's fixed context length. Without changing the rotary embedding after token removal, LLMs would receive rotary embedding of a non-monotonic position sequence. For example, after the first KV cache compression, LLMs might receive the input position embedding as  $[0, 1, 2, 3, 3096, 3097, \cdots, 4096]$ , and the position embedding of the generated sequences could be  $[1005, 1006, 1007, \cdots]$ . The position sequence of  $[0, 1, 2, 3, 3096, \dots, 4096, 1005, 1006, 1007, \dots]$  is a non-monotonic sequence, which may negatively hurts the performance. In contrast, our targeting settings will not process unlimited context size. For example, given a input sequence of 4012 length, after KV cache compression, the position sequence would be  $[0,4,6,16,\cdots,3927,3987,4012]$ , and the position sequence of the generated tokens would be [4013, 4014,  $\cdots$ ]. By keeping the rotary embedding unchanged after the tokens are removed, the LLM avoids non-monotonic position sequences, and the LLM can capture the exact position information even if the tokens are shifted. Our preliminary results show that rolling KV cache with the correct relative position will slightly decrease the performance.

