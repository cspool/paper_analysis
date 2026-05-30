# <span id="page-3-0"></span>3.2. CuMo Architecture

Sparse MoE in MLP Connector The MLP connector converts visual tokens into word embedding space, aligning dimensions between visual and text tokens. An effective architecture for the vision-language connector is an MLP block [\[46\]](#page-10-2) that contains two linear layers. We start from a single MLP block and replace it with a Top-K sparse MoE block, incorporating a Top-K router and a set of experts for projecting visual tokens into word embedding space.

Sparse MoE in Vision Encoder Vision encoders extract image features as sequences of visual tokens for reasoning in LLMs. CLIP [\[57\]](#page-10-15) is one the most popular pre-trained vision encoders for multimodal LLM since it is pre-trained on large-scale image-text pairs, which makes it suitable for processing images for multimodal usage. The visual encoding part of CLIP is a ViT [\[15\]](#page-8-10) model, which has consecutive MLP blocks in the transformer encoder. We substitute each MLP block with a Top-K sparse MoE block, retaining skip connections alongside MoE block outputs.

Sparse MoE in LLM In terms of using MoE in LLM, we compare the co-upcycled LLM with pre-trained MoEbased LLM. We start from Mistral-7B and the upcycled Mistral-7B-MoE slightly outperforms Mistral-7B on certain benchmarks. However, considering the constrained knowledge base of upcycled experts from Mistral-7B, we compare it with the pre-trained Mixtral 8x7B with pre-trained experts of a diverse knowledge base. Experimental results reveal that pre-trained Mixtral 8x7B significantly outperforms Mistral-7B-MoE. As a result, LLM is not co-upcycled with CLIP and MLP connectors since it brings marginal improvements with great additional parameters.

## <span id="page-3-3"></span>3.3. Training Recipe

Co-Upcycling MoE blocks We start with training the added MoE blocks from scratch while the model is struggling to converge. Attempts to address this issue with lower learning rates perform worse compared to the baseline. As a result, we adopt a co-upcycling approach, initializing each module that integrates sparsely-gated MoE blocks with pretrained MLPs to replace corresponding MLP blocks, as shown in Figure [3.](#page-2-0) This strategy consistently improves training stability and model performance.

Three-Stage Training To further enhance training stability, we adopt a three-stage training strategy for CuMo models, as illustrated in Figure [4.](#page-3-1) In the first stage, we only pretrain the MLP connector, given that the vision encoder and LLM have already undergone pre-training on large-scale data. During the second pre-finetuning stage, we train all parameters using high-quality caption data to warm up the entire model before introducing MoE blocks in the subsequent stage. The third stage involves visual instruction finetuning, where the multimodal LLM is scaled up with upcycled MoE blocks and trained on visual instruction tuning

<span id="page-4-2"></span><span id="page-4-1"></span>

|                     |              |       | SQA  | Text | <b></b> | DODE |        | MI   |      | MM                | VQA  | LLaVA             | SEED | MMMU | Math              |
|---------------------|--------------|-------|------|------|---------|------|--------|------|------|-------------------|------|-------------------|------|------|-------------------|
| Method              | LLM          | Act.  | IMG  | VQA  | GQA     | POPE | MME    | EN   | CN   | Vet               | v2   | Wild              | IMG  | val  | Vista             |
| 7B to 13B N         | Models       |       |      |      |         |      |        |      |      |                   |      |                   |      |      |                   |
| InstructBLIP [13]   | Vicuna-7B    | 7.9B  | 60.5 | 50.1 | 49.2    | -    | -      | 36.0 | 23.7 | 26.2              | -    | 60.9              | 60.5 | -    | -                 |
| Qwen-VL-Chat [3]    | Qwen-7B      | -     | 68.2 | 61.5 | 57.5    | -    | 1487.5 | 60.6 | 56.7 | -                 | 78.2 | -                 | 58.2 | 35.9 | -                 |
| LLaVA-v1.5 [46]     | Vicuna-7B    | 7.1B  | 66.8 | 58.2 | 62.0    | 85.9 | 1510.7 | 64.3 | 58.3 | 30.5              | 78.5 | 63.4              | 66.1 | -    | -                 |
| LLaMA-VID [41]      | Vicuna-7B    | -     | 68.3 | -    | 64.3    | 86.0 | 1521.4 | 65.1 | -    | -                 | 79.3 | -                 | 59.9 | -    | -                 |
| VILA [44]           | Vicuna-7B    | 7.1B  | 68.2 | 64.4 | 62.3    | 85.5 | 1533.0 | 68.9 | 61.7 | 34.9              | 79.9 | 69.7              | 61.1 | -    | -                 |
| SPHINX-Intern2 [20] | InternLM2-7B | -     | 70.4 | 58.1 | 56.2    | 86.9 | 1260.4 | 57.9 | -    | 36.5              | 75.5 | 57.6              | 68.8 | -    | 35.5              |
| LLaVA-NeXT [48]     | Mistral-7B   | 7.6B  | 72.8 | 65.7 | 64.8    | 86.7 | 1498   | 68.7 | 61.2 | 47.3              | 82.2 | 83.2              | 72.2 | 35.3 | 37.7              |
| LLaVA-NeXT [48]     | Vicuna-7B    | 7.1B  | 70.1 | 64.9 | 64.2    | 86.5 | 1519   | 67.4 | 60.6 | 43.9              | 81.8 | 81.6              | 70.2 | 35.8 | 34.6              |
| LLaVA-LLaMA3 [12]   | LLaMA3-8B-IT | 8.4B  | 72.9 | 59.0 | 62.6    | 86.4 | 1469   | 72.3 | 66.4 | -                 | -    | -                 | 70.1 | 36.8 | -                 |
| Mini-Gemini [42]    | Vicuna-7B    | 7.3B  | 65.2 | -    | -       | -    | 1523   | 69.3 | -    | 40.8              | -    | -                 | -    | 36.1 | 31.4              |
| MM1 [54]            | MM1-7B       | -     | 72.6 | 72.8 | -       | 86.6 | 1529.3 | 79.0 | -    | 42.1              | 82.8 | 81.5              | 69.9 | 37.0 | 35.9              |
| InstructBLIP [13]   | Vicuna-13B   | 14.2B | 63.1 | 50.7 | 49.5    | 78.9 | 1212.8 | -    | -    | 25.6              | -    | 58.2              | 63.1 | -    | -                 |
| LLaVA-v1.5 [46]     | Vicuna-13B   | 13.4B | 71.6 | 61.3 | 63.3    | 85.9 | 1531.3 | 67.7 | 63.6 | 35.4              | 80.0 | 70.7              | 68.2 | 36.4 | 27.6              |
| VILA [44]           | Vicuna-13B   | 13.4B | 73.7 | 66.6 | 63.3    | 84.2 | 1570.1 | 70.3 | 64.3 | 38.8              | 80.8 | 73.0              | 62.8 | -    | -                 |
| LLaMA-VID [41]      | Vicuna-13B   | -     | 70.0 | -    | 65.0    | 86.0 | 1542.3 | 66.6 | -    | -                 | 80.0 | -                 | 62.3 | -    | -                 |
| SPHINX-Plus [20]    | LLaMA2-13B   | -     | 74.2 | 65.7 | -       | 89.1 | 1457.7 | 71.0 | -    | 47.9              | -    | 71.7              | 74.8 | -    | 36.8              |
| Mini-Gemini[42]     | Vicuna-13B   | 13.6B | 65.9 | -    | -       | -    | 1565   | 68.5 | -    | 46.0              | -    | -                 | -    | 38.1 | 37.0              |
| InternVL-Chat [10]  | Vicuna-13B   | 19B   | -    | 61.5 | 66.6    | 87.6 | 1586.4 | -    | -    | -                 | 81.2 | -                 | -    | -    | -                 |
| LLaVA-NeXT [48]     | Vicuna-13B   | 13.4B | 73.6 | 67.1 | 65.4    | 86.2 | 1575   | 70   | 64.4 | 48.4              | 82.8 | 87.3              | 71.9 | 36.2 | 35.3              |
| CuMo                | Mistral-7B   | 7.8B  | 73.9 | 67.0 | 64.9    | 86.7 | 1548.6 | 73.0 | 66.6 | 51.0 <sup>†</sup> | 82.2 | 85.7 <sup>†</sup> | 72.1 | 39.1 | 35.1 <sup>†</sup> |
| 7B MoE M            | lodels       |       |      |      |         |      |        |      |      |                   |      |                   |      |      |                   |
| SPHINX-MoE [20]     | Mixtral-8×7B | l -   | 74.5 | 68.0 | 63.8    | 89.6 | 1485.3 | 71.3 | _    | 40.9              | 81.1 | 70.2              | 73.0 | 31.1 | 42.7              |
| MM1 [54]            | MM1-7B-MoE   | _     | 75.3 | 72.8 | _       | 87.6 | 1629.0 | 79.7 | _    | 47.0              | 83.4 | 82.0              | 70.4 | 40.9 | 40.9              |
| Mini-Gemini [42]    | Mixtral-8×7B | 13.5B | -    | 69.2 | -       | -    | 1639   | 75.6 | -    | 45.8              | -    | -                 | -    | 41.8 | 41.8              |
| CuMo                | Mixtral-8×7B | 13.5B | 77.9 | 66.0 | 63.8    | 85.7 | 1639.5 | 75.3 | 68.0 | 48.7 <sup>†</sup> | 81.8 | 84.7 <sup>†</sup> | 73.2 | 45.0 | 38.2 <sup>†</sup> |
| Private Me          | odels        |       |      |      |         |      |        |      |      |                   |      |                   |      |      |                   |
| GPT4V [56]          | -            | -     | -    | 78.0 | _       | -    | -      | 77.0 | 74.4 | 60.2              | _    | -                 | -    | 56.8 | 49.9              |
| Gemini 1.5 Pro [58] | _            | -     | -    | 73.5 | -       | _    | -      | 73.6 | 74.3 | 64.3              | 73.2 | -                 | _    | 58.5 | 52.1              |
| Claude 3 Opus [2]   | _            | -     | -    | -    | -       | _    | -      | 63.3 | 59.2 | 58.1              | -    | -                 | _    | 59.4 | 50.5              |
| Qwen-VL-Max [64]    | -            | -     | -    | 79.5 | -       | -    | 1790.1 | 77.6 | 75.1 | 66.6              | -    | -                 | -    | 51.4 | 51.0              |

Table 1. Comparisons between CuMo and other state-of-the-art multimodal LLMs on competitive benchmarks. These models are grouped by the size of the base LLM. The benchmarks are double-rowed due to limited space: SQA-IMG [50]; TextVQA [62]; GQA [24]; POPE [40]; MME [19]; MMBench [49]; MMVet [71]; VQAv2 [21]; LLaVA-Wild [47]; SEED-IMG [37]; MMMU [72]; MathVista [51]. Act.: Activated Parameters. Numbers are averaged by three inference runs of querying GPT API.

data.

**Loss Function** To maintain a load balance between experts in each MoE block, we adopt auxiliary losses based on the language modeling cross-entropy loss. The auxiliary losses comprise loading balance loss and router z-loss [77]. Hence, the total loss is

$$L = L_{ce} + \alpha_b L_b + \alpha_z L_z \tag{5}$$

Here,  $L_{ce}$  represents the language modeling loss, which computes the cross-entropy of next-token predictions.  $\alpha_b$  and  $\alpha_z$  denote coefficients for loading balance loss  $L_b$  and router z-loss  $L_z$ , set to 0.1 and 0.01, respectively, across all experiments. These auxiliary losses, abbreviated as bzloss in Section 4, are individually applied to the MLP connector, vision encoder, and LLM for simplicity.

### <span id="page-4-0"></span>4. Experiments

We train the CuMo models on a mixture of open-sourced datasets, which are converted into the visual instruction tuning format. Then, we conduct comprehensive evaluations of the performance of CuMo models across various competitive VQA-based and instruction-following-based benchmarks. Additionally, we perform ablation studies on each module with upcycled MoE blocks with qualitative analysis of the results.

#### 4.1. Implementation Details

Training Datasets During pre-training, we only utilize LLaVA-558K [47] to train the MLP connector for better alignment. In the subsequent pre-finetuning stage, detailed image caption data from ALLaVA [7] is employed to warm up all parameters of the multimodal LLM. For the final visual instruction tuning stage, a mixture of datasets including LLaVA-665K [46], ShareGPT4V [8], LAION-GPT-V [16], DocVQA [66], ChartQA [52], AI2D [31], InfoVQA [53], SynDog-EN [32], ALLaVA [7], and LIMA [74] is utilized to train the CuMo models with upcycled MoE blocks. The total data size for visual instruction tuning is approximately 1.65 million, and all training data are publicly accessible.

<span id="page-5-4"></span><span id="page-5-0"></span>

|                   |            |      |            | SQA  | Text |      |      |        | MMI  | Bench | MM   | VQA  | LLaVA | SEED |
|-------------------|------------|------|------------|------|------|------|------|--------|------|-------|------|------|-------|------|
| Method            | LLM        | PT   | IT         | IMG  | VQA  | GQA  | POPE | MME    | EN   | CN    | Vet  | v2   | Wild  | IMG  |
| InstructBLIP [13] | Vicuna-7B  | 129M | 1.2M       | 60.5 | 50.1 | 49.2 | -    | -      | 36.0 | 23.7  | 26.2 | -    | 60.9  | 60.5 |
| InstructBLIP [13] | Vicuna-13B | 129M | 1.2M       | 63.1 | 50.7 | 49.5 | 78.9 | 1212.8 | -    | -     | 25.6 | -    | 58.2  | 63.1 |
| IDEFICS-9B [25]   | LLaMA-7B   | 353M | 1 <b>M</b> | -    | 25.9 | 38.4 | -    | -      | 48.2 | 25.2  | -    | 50.9 | -     | -    |
| IDEFICS-80B [25]  | LLaMA-65B  | 353M | 1 <b>M</b> | -    | 30.9 | 45.2 | -    | -      | 54.5 | 38.1  | -    | 60.0 | -     | -    |
| Qwen-VL [3]       | Qwen-7B    | 1.4B | 50M        | 67.1 | 63.8 | 59.3 | -    | -      | 38.2 | 7.4   | -    | 78.8 | -     | 56.3 |
| Qwen-VL-Chat [3]  | Qwen-7B    | 1.4B | 50M        | 68.2 | 61.5 | 57.5 | -    | 1487.5 | 60.6 | 56.7  | -    | 78.2 | -     | 58.2 |
| LLaVA-v1.5 [46]   | Vicuna-7B  | 558K | 665K       | 66.8 | 58.2 | 62.0 | 85.9 | 1510.7 | 64.3 | 58.3  | 30.5 | 78.5 | 63.4  | 66.1 |
| LLaVA-v1.5 [46]   | Vicuna-13B | 558K | 665K       | 71.6 | 61.3 | 63.3 | 85.9 | 1531.3 | 67.7 | 63.6  | 35.4 | 80.0 | 70.7  | 68.2 |
| CuMo              | Mistral-7B | 558K | 665K       | 71.7 | 59.3 | 63.2 | 87.1 | 1428.6 | 69.6 | 62.6  | 34.3 | 80.6 | 68.8  | 69.6 |

Table 2. Comparisons between CuMo Mistral-7B and other multimodal LMM models with limited training data.

<span id="page-5-1"></span>

| Method                 | SQA  | $VQA^T$ | MMVet | SEED |
|------------------------|------|---------|-------|------|
| Baseline on Mistral-7B | 72.8 | 57.6    | 32.1  | 66.4 |
| + Top 2-in-4 & Scratch | 68.1 | 55.6    | 29.3  | 65.1 |
|                        | 73.7 | 57.2    | 32.3  | 67.1 |
| + bzloss               | 73.5 | 57.4    | 33.1  | 67.4 |
| ≓ Top 2-in-8 & Upcycle | 73.4 | 57.6    | 32.4  | 67.2 |

Table 3. Ablation study on the MLP-MoE module. Each row represents a different configuration, with changes or additions marked using  $\rightleftharpoons$  and + symbols, respectively. Settings highlighted with a light blue background are those adapted for the MLP-MoE module in Table 1.

<span id="page-5-2"></span>

| Method                | SQA  | $VQA^T$ | MMVet | SEED |
|-----------------------|------|---------|-------|------|
| MLP-MoE               | 73.5 | 57.4    | 33.1  | 67.4 |
| + Unfreeze CLIP       | 72.0 | 58.9    | 34.7  | 69.0 |
| + Top 2-in-4 & bzloss | 72.8 | 59.7    | 35.4  | 69.8 |
| ⇒ Top 2-in-8 & bzloss | 71.0 | 59.0    | 33.6  | 69.2 |

Table 4. Ablation study on the CLIP-MoE module. All MoE blocks in CLIP are initialized with upcycling.

<span id="page-5-3"></span>

| Method                                     | SQA  | $VQA^T$ | MMVet | SEED |
|--------------------------------------------|------|---------|-------|------|
| MLP-MoE & CLIP-MoE                         | 71.7 | 59.3    | 34.3  | 69.6 |
| + Mistral 4×7B & Upcycle                   | 72.8 | 57.0    | 35.2  | 69.9 |
| <i>≅ Mistral 8×7B &amp; Upcycle</i>        | 73.2 | 56.4    | 35.7  | 70.5 |
| $\rightleftharpoons$ Mixtral $8 \times 7B$ | 74.2 | 60.6    | 40.0  | 72.6 |

Table 5. Ablation study on the LLM-MoE module. Mixtral  $8\times7B$  outperforms upcycled Mistral MoE models significantly.

The detailed breakdown of the training dataset is listed in Appendix A.

Evaluation Benchmarks Evaluation of CuMo models primarily focuses on academic VQA-based datasets such as VQAv2 [21], GQA [24], Science-QA [50], and TextVQA [62], as well as instruction-following-based LMM benchmarks including POPE [40], MME [19], MM-Bench [49], SEED-Bench [37], LLaVA-Wild [47], and MM-Vet [71]. Additionally, the challenging MMMU [72] and MathVista [51] datasets are evaluated to assess the vi-

sual reasoning abilities of the multimodal LLMs.

Training Settings We employ the pre-trained CLIP ViT-L [57] as the vision encoder, a two-layer MLP as the vision-language connector, and Mistral-7B [29] as the LLM to establish the baseline model following LLaVA v1.5 [46]. We only use LLaVA-558K [46] as pre-training data and LLaVA-665K [46] as visual instruction tuning data to train the baseline model and make ablation studies for comparisons. The learning rate is set to 1e-3 for pre-training the MLP connector and reduced to 2e-5 for visual instruction tuning of both the MLP connector and CLIP. To further stabilize the visual instruction tuning process after scaling up with additional data, the learning rate is lowered to 2e-6 for all parameters of the CuMo models in the final results. More hyperparameters of the training process is listed in Appendix B.

**Evaluation Settings** During evaluation, we adhere to the settings outlined in the LLaVA series [46], employing a greedy decoding strategy for all benchmarks. The data and questions are converted into visual instructions to prompt the multimodal LLMs. For benchmarks that utilize GPT API for evaluation, we adopt gpt-4-0613 for LLaVA-Wild [47] and gpt-3.5-turbo for MathVista [51].

#### 4.2. Main Results

Comparison with SoTA Multimodal LLMs In Table 1, we present a comparison of CuMo models with other state-of-the-art instruction-following-based multimodal LLMs. We categorize the models based on the size of the base LLMs, including 7B models, 13B models, and 7B MoE models. CuMo Mistral-7B outperforms other 7B-based state-of-the-art multimodal LLMs across multiple benchmarks. Moreover, the performance of the CuMo Mistral-7B model is comparable to many 13B-based multimodal LLMs. In the case of Mixtral-8×7B models, CuMo achieves results on par with SPHINX-MoE, MM1, and Mini-Gemini. LLaMA-based LLMs [11, 67] are not utilized in our experiments due to license constraints.

Comparison under limited training data To further evaluate the effectiveness of the co-upcycled MoE blocks, we

<span id="page-6-3"></span><span id="page-6-0"></span>

| $1 \times$                | $2\times$    | $3\times$    | SQA  | $\mathbf{V}\mathbf{Q}\mathbf{A}^T$ | MMVet | SEED |
|---------------------------|--------------|--------------|------|------------------------------------|-------|------|
| $\checkmark$              | -            | -            | 71.7 | 59.3                               | 34.3  | 69.6 |
| $\overline{\hspace{1em}}$ | <b>√</b>     | -            | 71.7 | 60.6                               | 35.0  | 69.7 |
| $\checkmark$              | -            | $\checkmark$ | 72.9 | 61.0                               | 37.0  | 69.7 |
| $\checkmark$              | $\checkmark$ | $\checkmark$ | 72.2 | 60.5                               | 36.9  | 70.1 |

Table 6. Ablation study on multi-resolution image features. The combination of  $3\times$  and  $1\times$  is adopted for the final models in Table 1.

<span id="page-6-1"></span>

| Method                      | SQA  | $VQA^T$ | MMVet | SEED |
|-----------------------------|------|---------|-------|------|
| No PFT                      | 71.7 | 59.3    | 34.3  | 69.6 |
| + ShareGPT4V                | 72.4 | 61.7    | 36.5  | 70.0 |
| $\rightleftharpoons$ ALLaVA | 73.0 | 62.8    | 37.2  | 70.9 |

Table 7. Ablation study on the pre-finetuning stage. ALLaVA is chosen for pre-finetuning due to its provision of high-quality image caption data.

train the vanilla CuMo mistral-7B under limited training data in Table 2. It shows that CuMo outperforms other 7B models and reaches comparable performance to LLaVA-v1.5 Vicuna-13B under the same training data.

### <span id="page-6-4"></span>4.3. Ablation Study

Upcycle MLP connector to MLP-MoE We initiate the ablation study by replacing the MLP connector with upcycled MLP-MoE, as depicted in Table 3. We start with a Top 2-in-4 router and train the MoE blocks from scratch, which leads to a clear performance drop on all benchmarks. Then, we adopt the upcycling strategy to initialize the MLP experts. We observe marginal improvements over the baseline, considering each expert comprises only two linear layers. Subsequently, the incorporation of bzloss to ensure a balanced loading of experts in the MLP-MoE yields noticeable enhancements on MMVet. However, employing a Top 2-in-8 router with upcycling and bzloss results in a slight performance decline, possibly due to the limited visual instruction tuning data to train robust and well-balanced eight experts. Empower CLIP with CLIP-MoE In Table 4, initially unfreezing CLIP based on MLP-MoE leads to noticeable improvements on TextVQA and MMVet benchmarks. However, training the added Top2-in-4 MoE blocks in CLIP from scratch proves unsuccessful, as the model fails to converge even with reduced learning rates. Consequently, adopting upcycled MoE blocks during the visual instruction tuning stage yields further enhancements on TextVQA, MMVet, and SEED benchmarks.

**Upcycle LLM vs Pre-trained LLM-MoE** Upon replacing all MLP blocks with sparsely-gated MoE blocks in the visual part, we further investigate the utilization of the MoE architecture in the LLM. Starting from the Mistral-

<span id="page-6-2"></span>![](_page_6_Figure_8.jpeg)

Figure 5. Expert distributions of MoE blocks in CLIP. We select layers from CLIP and summarize the activated experts during the feed-forward process on the MME test set.

7B model, we first lower the learning rate to 2e-6 to set the baseline and the following experiments since a learning rate of 2e-5 induces training instabilities. Then, we upcycle each MLP block with a sparsely-gated MoE block, initializing the weight of each expert from the pre-trained MLP block. As demonstrated in Table 5, the upcycled Mistral- $4 \times 7B$  and  $8 \times 7B$  outperform the Mistral-7B model slightly except for TextVQA. However, considering that the upcycled experts significantly increase parameters without introducing new knowledge, we replace the upcycled Mistral  $8 \times 7B$  with Mixtral  $8 \times 7B$  [30]. In Mixtral  $8 \times 7B$ , all expert layers are pre-trained on large-scale language data, providing superior initialization compared to upcycling. The results indicate that CuMo Mixtral-8x7B outperforms its upcycled counterparts significantly and is employed in the final models with bzloss to maintain a balanced loading of experts.

**Multi-Resolution Visual Features** Incorporating multi-resolution inputs is crucial for enhancing the understanding of image content in multimodal LLMs. Following the approach outlined in  $S^2$ [61], we introduce multi-resolution inputs to CLIP and concatenate the feature maps channelwise to maintain the total number of visual tokens consistent with low-resolution inputs. As illustrated in Table 6, an empirical combination of  $3\times$  and  $1\times$  yields optimal performance and we adopt this configuration for the final CuMo models.

**Pre-FineTuning Stage** Previous ablation studies were conducted directly after the pre-training of the MLP connector, leading to observed training instabilities during visual instruction tuning. To address this, we introduce a pre-finetuning stage using high-quality image caption data, wherein all parameters are unfrozen. In Table 7, we leverage caption data from ALLaVA for this stage. Results indicate that ALLaVA data proves to be a superior option,

<span id="page-7-0"></span>![](_page_7_Figure_0.jpeg)

prominently displayed, indicating the film's availability in theaters. The poster also highlights the film's availability in various formats, including 3D, which was a significant selling point at the time of its release.

![](_page_7_Figure_2.jpeg)

#### CuMo-Mistral-7B

The poster is for the movie "The Avengers," which is part of the Marvel Cinematic Universe. It features a team of superheroes, each with unique abilities and costumes, coming together to face a significant threat. The central figure, with the iconic red and gold suit, is likely the leader of the group. The presence of the Hulk and Thor suggests that the film includes a mix of science fiction and fantasy elements. The cityscape and the explosion in the background imply that the movie's plot involves a large-scale conflict that impacts the urban environment. The release date and the mention of "3D and IMAX 3D" indicate that the movie was available in various formats, catering to different viewing preferences. The poster is designed to convey the epic scale and the high stakes of the film, as well as the diverse team of heroes that viewers can expect to see on screen.

Figure 6. Dialogues between the user and multimodal LLMs on challenging images. We highlight the correct answers and hallucinations from the responses of the multimodal LLMs.

providing fewer but higher-quality captions for training, ultimately leading to improved performance.

can you introduce this movie based

on this poster

### 4.4. Qualitative Analysis

Expert Distribution As shown in Figure [5,](#page-6-2) we visualize the expert distributions in the MoE block from selected layers at CLIP-MoE. The dataset analyzed is the test set of the MME benchmark. The distribution indicates that the selected experts during inference are evenly spread across layers, providing further evidence of the effectiveness of the auxiliary losses in maintaining load balance.

Dialogue Comparisons Presented in Figure [6,](#page-7-0) we contrast the responses from CuMo-Mistral-7B, LLaVA-Yi-34B, and MiniGemini-Yi-34B. It demonstrates that CuMo-Mistral-7B can effectively follow instructions and predominantly provide correct answers to challenging questions derived from complex scenes. However, CuMo also exhibits instances of hallucinations, such as responding with "2 characters standing on the table," highlighting the need for further investigation to mitigate hallucinations in CuMo.

