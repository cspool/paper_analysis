# <span id="page-21-1"></span>B Experimental details

This section provides additional details for the synthetic, JRT-Prompt and JRT-RNN experimental protocols. We use NVidia A100-80GB GPUs for all training runs.

## B.1 Additional details for set disjointness synthetic experiments

This section provides experimental details for Figure [2.](#page-3-0)

Dataset The procedure for generating training and evaluation data for our synthetic experiments is shown in Algorithm [1.](#page-21-0) We train on the following mixture of sequence lengths, where the tuple denotes (|A|, |B|) for sets A and B in the sequence:

```
(4, 16),(16, 4),(8, 32),(32, 8),(64, 16),(16, 64),(4, 128),(128, 4),(16, 256),(256, 16),(4, 256),(256, 4)
```

We evaluate on the following mixture of sequence lengths (requiring length extrapolation from training), where the tuple denotes (|A|, |B|) for sets A and B in the sequence:

```
(1, 32),(32, 1),(4, 32),(32, 4),(4, 128),(128, 4),(16, 256),(256, 16),(4, 256),(256, 4),(16, 512),
      (512, 16),(4, 512),(512, 4),(8, 768),(768, 8),(16, 768),(768, 16),(4, 768),(768, 4)
```

We include 20000 data points per tuple above during training and 1000 during evaluation. We use V = 2048 as the vocabulary size.

#### <span id="page-21-0"></span>Algorithm 1 Set Disjointness Synthetic Procedure

Require: Vocabulary V , Sequence lengths N<sup>A</sup> and N<sup>B</sup> for sets A and B, Special token IDs prefix\_token\_id, mask\_tok\_id, sep\_sets\_token\_id, sep\_answer\_tok\_id

#### Output: Synthetic sequence

- 1: Let the first half of V , VA, be prospective tokens for set A and the second half, VB, be prospective tokens for set B.
- 2: Randomly select N<sup>A</sup> tokens from V<sup>A</sup> for set A. Randomly select N<sup>B</sup> tokens from V<sup>B</sup> for set B.
- 3: Randomly select a token t from A as the intersecting token between sets. Replace a random token (at a random position) from B with t.
- 4: Construct the final input sequence as the concatenation:

$$[prefix\_token\_id], A, [sep\_sets\_token\_id], B, sep\_answer\_tok\_id], [t]$$

- 5: The label sequence contains a "-100" (i.e., a token to ignore computing the loss) at all positions except for the final position. We mask [t] (the final position) from the input sequence.
- 6: Output the synthetic input and label sequences.

Models We evaluate causal and non-causal variants of the Based recurrent model. Each model contains 4 layers alternating gated-convolutions (with a short filter of size 3) and linear attention with 2 query key and value heads. For the non-causal variant, we simply replace the causal cumulative sum in linear attention with a sum, and we use non-causal circular convolutions. For the linear attention feature map, we use a Taylor approximation to the softmax-exponential function as in [\[7\]](#page-12-6) (also defined in ??). Each layer has an MLP with GeLU activations. We do not use any explicit positional embeddings, instead finding the short-convolutions sufficient for positional information.

To sweep the state size, we vary the model width or dimension ∈ {36, 48, 64, 96, 128} and linear attention feature dimension ∈ {4, 8, 16, 24}.

**Training** We train using cross-entropy loss on the predicted vs. true intersection token t in Algorithm 1. For each point in Figure 2, we sweep learning rates  $\in \{0.0001, 0.0005, 0.0008\}$  (after identifying that this regime is most effective for the architectures) and report the maximum accuracy after 48 epochs of training. We use AdamW as the optimizer with 0.1 weight decay.

We build our synthetic experiments using the synthetics repository provided by prior work [23]: https://github.com/HazyResearch/zoology.

#### B.2 Additional details for JRT-PROMPT experiments

For Table 1 (JRT-PROMPT), we use the following publicly available models pretrained and released by the baseline works:

- Based [7] models are at https://huggingface.co/collections/hazyresearch/based-65d77fb76f9c813c8b94339c
- Gated Linear Attention [9] models are at https://huggingface.co/fla-hub.
- Mamba [1] and Mamba-2 [36] models are at https://huggingface.co/state-spaces

We integrate all tasks into the popular LM-Eval harness to run inference. We truncate long-documents (e.g., in NQ, FDA, SWDE) to length 1k tokens for the default prompting and length 2k tokens for JRT-PROMPT so that both methods receive the same information in-context. We note that these lengths are chosen because the listed pretrained models have 2048 context lengths. We ensure that the answer span is present in truncated documents. We do not use any task-specific prompt customization in this section, to highlight the effectiveness of JRT-PROMPT despite little effort.

#### B.3 Additional details for pre-training experiments

Additional details for JRT-RNN To facilitate comparisons to prior work, we start with the Based architecture [7] and replace its linear attention layers with JRT-RNN linear attention layers. Note that the Based architecture hybridizes gated convolution layers (kernel size 3), sliding window attention layers (window size 128), and linear attention layers (using a Taylor approximation to the exponential function as the feature map, with feature dimension 16). We maintain the exact same order and number of each layer type as the Based work. We reduce the number of gated convolution layers by 1 at 360M parameters to account for the increase in parameters due to the encoder projections.

Next we include a description of the linear attention feature map used in our trained models. Based uses a 2<sup>nd</sup>-order Taylor approximation to the softmax-exponential function as the feature map  $\phi : \mathbb{R}^d \to \mathbb{R}^{\tilde{d}}$  [76]. To approximate  $\exp(\mathbf{q}_i^{\top} \mathbf{k}_j / \sqrt{d})$ :

$$\exp(x) \approx 1 + x + \frac{x}{2!} \tag{6}$$

$$\phi(\mathbf{q}_i)^{\top}\phi(\mathbf{k}_j) = 1 + \mathbf{q}_i^{\top}\mathbf{k}_j + \frac{(\mathbf{q}_i^{\top}\mathbf{k}_j)^2}{2}$$
(7)

The second order term has large dimension 273 if  $\tilde{d} = 16$  as in [7]. As a result, a careful IO-aware implementation is key to efficiency.

Training protocol For Table 2, we use the code provided by the baseline works, which has been adapted from the FlashAttention code base: https://github.com/Dao-AILab/flash-attention/tree/main for our pretraining runs [12]. The Pile data is tokenized using the GPT2BPETokenizer and all models see the data in the same order. Here we provide details on the hyperaparamters and configurations used for training each architecture.

- **JRT-RNN** We provide hyperparameters and settings used for JRT-RNN in Table 15. We integrate JRT-RNN into the Based implementation released by the prior work.
- Based [7] We train using the specifications in Table 16 and the architecture implementation provided here: https://github.com/HazyResearch/based.

- Transformer++ [\[32\]](#page-14-0) We refer to the modern Llama architecture with Rotary encodings, RMSNorm and SwiGLU as Transformer++, following prior work [\[1,](#page-12-0) [9\]](#page-12-8). We train using the the specifications in Table [18](#page-55-0) using the Flash Attention training code provided here: [https://github.com/Dao-AILab/](https://github.com/Dao-AILab/flash-attention/tree/main) [flash-attention/tree/main](https://github.com/Dao-AILab/flash-attention/tree/main) [\[12\]](#page-12-11).
- Mamba [\[1\]](#page-12-0) We train using the specifications in Table [17,](#page-55-1) where the parameters are sourced from the Appendix of [\[1\]](#page-12-0). The architecture implementation is from the reference at [https://github.com/](https://github.com/state-spaces/mamba) [state-spaces/mamba](https://github.com/state-spaces/mamba).

We give all models the Transformer++ change (e.g., SwiGLU, Rotary) where relevant.

Inference protocol For JRT-RNN, we left-pad prefill when it is shorter than the encoder region and mask in the linear attention layer following Listing 3 Appendix [D.](#page-28-0) We apply no changes if the prefill exceeds the encoder region. For all results reported in this work, we use the parallel view of JRT-RNN to process the prefill and compute initial states following Section [4,](#page-5-1) then use the recurrent view to decode.

## B.4 Additional details for Pile perplexity slicing analysis

In Section [5.2,](#page-9-0) we analyze the perplexity of different models trained on the Pile, on the Pile test data. Here we provide additional details for the protocol.

We compute the training counts of bigrams across 10M Pile training documents, each of length 2048. We evaluate the models on 3, 200 sequences of length 2048 (6.6M total tokens), and measure perplexity on the last 1024 tokens per sequence (the causal, decoder region for JRT-RNN) (3.3M total tokens). We then evaluate perplexity on two slices of this test set:

- 1. Associative recall (AR) hits. Tokens in the final position of a bigram which previously occurred in context, and this bigram is infrequent during training. For instance, in the sequence "While lunching at the Maison Bergey bistro near his apartment: he had been musing about the ... (723 tokens) ... the young waitress's sigh at the Maison Bergey." the second "Bergey" would be included as an "AR hit" if "Maison Bergey" is a rare bigram during training. Intuitively, the model would need to rely on the context to predict the next token if the bigram were rare during training (i.e., was not memorized), testing the model's recall ability.
- 2. Other tokens. All other tokens. Intuitively, these tokens test the knowledge memorized in the model parameters.

In Figure [3,](#page-9-0) for the recall frequencies plot, we restrict to "AR hits" where the bigram and the reoccurrence of the bigram in context are separated by at least 1024 in distance within the context. In the recall gaps plot, we restrict to bigrams that are seen fewer than 1000 times during training and vary the distance between bigram occurrences in-context on the x axis.

## B.5 Evaluation datasets

Here we provide additional details on the recall-intensive benchmark suite used in this work. The tasks include:

- FDA FDA is an information extraction task where documents are FDA reports for pre-market medical devices and the model needs to extract attributes such as the device code, classification, and indications for use [\[7,](#page-12-6) [29\]](#page-13-11). These FDA reports are frequently analyzed by domain experts [\[56\]](#page-15-8). We use the dataset released at: <https://huggingface.co/datasets/hazyresearch/based-fda>, which is part of the LM-Eval Harness repository [\[77\]](#page-17-0).
- SWDE SWDE is an information extraction task where documents are HTML webpages spanning 14 different websites in the Movie and University topic domains (e.g., "IMDB.com", "RottenTomatoes", "USNews") and the model needs to extract attributes such as the Movie director / assistant director and University tuition [\[7,](#page-12-6) [29,](#page-13-11) [57,](#page-15-9) [78\]](#page-17-1). We use the dataset released at: [https://huggingface.co/datasets/](https://huggingface.co/datasets/hazyresearch/based-swde) [hazyresearch/based-swde](https://huggingface.co/datasets/hazyresearch/based-swde), which is part of the LM-Eval Harness repository [\[77\]](#page-17-0).

- SQUADv2 SQUADv2 is a document QA benchhmark where documents come from Wikipedia and answer to questions are a span of tokens in the document [\[7,](#page-12-6) [58\]](#page-15-10). We use the version of the dataset released at: <https://huggingface.co/datasets/hazyresearch/based-squad>, which is part of the LM-Eval Harness repository [\[77\]](#page-17-0).
- TriviaQA TriviaQA is a popular document QA benchmark where documents come from both Wikipedia and the general web and the question structure varies [\[60\]](#page-15-12). We use the dataset released at: [https:](https://huggingface.co/datasets/mandarjoshi/trivia_qa) [//huggingface.co/datasets/mandarjoshi/trivia\\_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- Natural Questions (NQ) Natural Questions is a popular document QA benchmark where documents come from Wikipedia and the questions are real queries issued to the Google search engine [\[59\]](#page-15-11). The answers are spans of text from the documents. We use the dataset released at: [https://huggingface.](https://huggingface.co/datasets/natural_questions) [co/datasets/natural\\_questions](https://huggingface.co/datasets/natural_questions).
- Drop DROP is a challenging document QA benchmark that requires discrete reasoning over paragraphs from Wikipedia articles [\[61\]](#page-15-13). The questions often require arithmetic operations, counting, or sorting of information found in the documents. We use the dataset released at: [https://huggingface.co/datasets/](https://huggingface.co/datasets/ucinlp/drop) [ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop).

Cloze Completion Formatting As the models in this work are not instruction fine-tuned and have been trained on next token prediction, they are more effective at producing relevant answers when the prompt format aligns with the pre-training task (next token prediction) as shown in prior work [\[79\]](#page-17-2). Therefore, we reformat the questions in these benchmarks to a cloze-completion format using Meta's Llama-3-70B model [\[69\]](#page-16-6).

Given the question and the answer, the prompt we use is, where we provide the original question and answer from the task example:

```
Converting to Cloze Format
Can you rewrite this question and answer as a statement . Ensure that the answer is the last part
     of the statement .
Question : { question }
Answer : { answers }
Rewrite :
```

As an example:

