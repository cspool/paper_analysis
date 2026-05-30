# 4 Experiments

## 4.1 Visual Question Answering Benchmarks

We conduct experiments on a diverse set of visual question answering benchmarks, including VQAv2 [\(Goyal et al.,](#page-19-6) [2016\)](#page-19-6), OK-VQA [\(Marino et al.,](#page-20-9) [2019\)](#page-20-9), GQA [\(Hudson & Manning,](#page-19-7) [2019\)](#page-19-7), VizWizQA [\(Bigham et al.,](#page-17-2) [2010\)](#page-17-2), and TextVQA [\(Singh et al.,](#page-20-10) [2019\)](#page-20-10). The VQAv2 dataset is currently the largest visual question answering dataset available. OK-VQA involves questions that require external knowledge beyond multimodal inputs. GQA is designed to validate the model's reasoning capabilities. VizWizQA is constructed from question-answer pairs sourced from visually impaired users. TextVQA focuses more on evaluating the model's ability to understand text in natural scenes. These datasets are strategically selected to thoroughly evaluate our model's ability to understand and reason across various visual contexts and knowledge domains. Table [3](#page-7-0) presents the comparison results between mPLUG-Owl3 and State-of-the-Art multimodal large language models, including CogVLM [\(Wang et al.,](#page-21-9) [2023\)](#page-21-9), EVLM-Chat [\(Chen et al.,](#page-18-3) [2024b\)](#page-18-3), flamingo [\(Alayrac et al.,](#page-17-0) [2022\)](#page-17-0), Qwen-VL-Chat [\(Bai et al.,](#page-17-3) [2023\)](#page-17-3), Idefics [\(Laurençon et al.,](#page-19-2) [2023\)](#page-19-2), InstructBLIP [\(Dai et al.,](#page-18-10) [2023\)](#page-18-10), mPLUG-Owl2 [\(Ye et al.,](#page-22-1) [2024\)](#page-22-1), LLaVA-1.5 [\(Liu et al.,](#page-20-1) [2024a\)](#page-20-1), LLaVA-Next [\(Liu et al.,](#page-20-11) [2024b\)](#page-20-11), VILA-1.5 [\(Lin et al.,](#page-20-12) [2023b\)](#page-20-12), Idefics2 [\(Laurençon et al.,](#page-19-3) [2024\)](#page-19-3), Mantis-SigLIP [\(Jiang et al.,](#page-19-0) [2024\)](#page-19-0).

<span id="page-7-2"></span><span id="page-7-0"></span>

| Model         | # Param | VQAv2 | OK-VQA | GQA               | VizWizQA | TextVQA     |
|---------------|---------|-------|--------|-------------------|----------|-------------|
| CogVLM        | 17B     | 82.3  | 64.8   | -                 | _        | 70.4        |
| EVLM-Chat     | 32B     | 81.9  | -      | 64.4              | 47.3     | 67.5        |
| Flamingo      | 80B     | 81.3  | 50.6   | -                 | 57.2     | 54.7        |
| 8B-level MLMM | Is      |       |        |                   |          |             |
| Qwen-VL-Chat  | 9B      | 78.2  | 56.6   | 57.5              | 38.9     | 63.8        |
| Idefics1      | 9B      | 68.8  | 50.4   | -                 | -        | 39.3        |
| Flamingo      | 9B      | 51.8  | 44.7   | -                 | -        | 46.3        |
| InstructBLIP  | 7B      | 75.2  | 45.2   | 49.2              | 34.5     | 33.6        |
| mPLUG-Owl2    | 8B      | 79.4  | 57.7   | 56.1              | 54.5     | 58.2        |
| LLAVA-1.5     | 8B      | 78.5  | -      | 62.0              | 50.0     | 58.2        |
| LLAVA-Next    | 8B      | 81.8  | -      | 64.2              | 57.6     | 64.9        |
| VILA-1.5      | 8B      | 80.9  | -      | $\overline{61.9}$ | 58.7     | 66.3        |
| Idefics2      | 8B      | 80.8  | 53.5   | -                 | -        | 70.4        |
| Mantis-SigLIP | 8B      | 74.9  | 55.4   | -                 | -        | 59.2        |
| mPLUG-Owl3    | 8B      | 82.1  | 60.1   | 65.0              | 63.5     | <u>69.0</u> |

Table 3: **Performance comparison on visual question answering.** The accuracy is reported. We use **bold** to mark the highest score and underline to mark the second highest of 8B-level MLLMs.

mPLUG-Owl3 outperforms 8B-level language models in VQAv2, OK-VQA, GQA, and VizWizQA. Furthermore, it surpasses the 32B-parameter EVLM³ in GQA and VizWizQA. In TextVQA, although mPLUG-Owl3's performance is slightly lower than that of Idefics2, it still exceeds that of other 8B models. It is noteworthy that, despite having 8B parameters, mPLUG-Owl3 exhibits superior inference speed and memory efficiency compared to models of the same scale, thanks to the introduction of Hyper Attention.

#### 4.2 General MLLM Benchmarks

We evaluate mPLUG-Owl3 on various single-image general multimodal large language model benchmarks including MMBench-EN/CN (Liu et al., 2023b), MM-Vet (Yu et al., 2023), POPE (Li et al., 2023d) and AI2D (Kembhavi et al., 2016). MMBench provides a comprehensive evaluation of a model's multimodal capabilities in both Chinese and English contexts. MM-Vet assesses the multimodal conversational abilities of a model using GPT-4 evaluation. POPE can evaluate the extent of multimodal hallucinations in a model. AI2D assesses a model's ability to understand science diagrams inputs.

Table 4 shows that mPLUG-Owl3 achieves state-of-the-art performance on MMBench-EN, MMBench-CN, MM-Vet and POPE across 8B-level models such as OpenFlamingo (Awadalla et al., 2023), Cambrian (Tong et al., 2024) and MiniCPM-Llama3-V2.5 (Yao et al., 2024). It also matches or surpasses the performance of larger models such as CogVLM (Wang et al., 2023) and EVLM-Chat (Chen et al., 2024b). mPLUG-Owl3 does not achieve state-of-the-art performance on the AI2D dataset. Due to limited training resources, we do not fine-tune the vision encoder, which restricts its performance in scenarios rich in text.

### 4.3 Multi-image and Video Benchmark

We also evaluate the performance of mPLUG-Owl3 on video and multi-image benchmarks, as it is capable of processing multiple images with an interleaved format. we include VideoChat2 (Li et al., 2023c), Video-LLaMA2 (Cheng et al., 2024), Video-ChatGPT (Maaz et al., 2023), ShareGPT4Video (Chen et al., 2024c), PLLaVA (Xu et al., 2024), Idefics2 (Laurenccon et al., 2024), Mantis-SigLIP (Jiang et al., 2024) and LLAVA-Interleave (Li et al., 2024a).

The results of video evaluation is shown in Table 5. The NextQA (Xiao et al., 2021) and MVBench (Li et al., 2023c) are short video benchmarks, with video durations all less than one

<span id="page-7-1"></span><sup>&</sup>lt;sup>3</sup>EVLM does not provide the number of parameters for its cross module. The parameter count in this table is estimated based on its model architecture.

<span id="page-8-2"></span><span id="page-8-0"></span>

| Model               | # Param | MMB-EN | MMB-CN | MM-Vet | POPE | AI2D        |
|---------------------|---------|--------|--------|--------|------|-------------|
| CogVLM              | 17B     | 65.8   | 69.8   | 52.8   | 88.0 | 63.3        |
| EVLM-Chat           | 32B     | 76.9   | 76.9   | -      | 89.7 | 76.0        |
| InstructBLIP        | 13B     | 38.3   | -      |        | 81.5 | -           |
| 8B-level MLMMs      |         |        |        |        |      |             |
| LLAVA-1.5           | 8B      | 64.3   | 58.3   | 31.1   | 85.9 | 55.5        |
| OpenFlamingo        | 9B      | 32.4   | 14.4   | 24.8   | -    | 31.7        |
| mPLUG-Owl2          | 8B      | 64.5   | -      | 36.2   | -    | 55.7        |
| LLAVA-Next          | 8B      | 67.4   | 60.6   | 43.9   | 86.5 | 66.6        |
| LLAVA-Interleave    | 8B      | -      | -      | -      | 86.8 | 73.9        |
| VILA1.5             | 8B      | 72.3   | 66.2   | 38.3   | 84.4 | -           |
| Idefics2            | 8B      | 75.7   | 68.6   | 34.0   | 86.2 | 72.3        |
| Cambrian            | 8B      | 74.6   | 67.9   | -      | -    | 74.6        |
| MiniCPM-Llama3-V2.5 | 8B      | 77.6   | 73.8   | -      | -    | <b>78.4</b> |
| Mantis-SigLIP       | 8B      | 68.7   | -      | -      | -    | -           |
| mPLUG-Owl3          | 8B      | 77.6   | 74.3   | 40.1   | 88.2 | 73.4        |

Table 4: **Zero-shot multi-modal evaluation on multi-modal benchmarks.** The overall scores are reported for evaluation. We use **bold** to mark the highest score and <u>underline</u> to mark the second highest of 8B-level MLLMs.

<span id="page-8-1"></span>

| Model            | # Param | NextQA      | MVBench | VideoMME w/o sub | LongVideoBench-val |
|------------------|---------|-------------|---------|------------------|--------------------|
| VideoChat2       | 8B      | 68.6        | 51.9    | 43.8             | 36.0               |
| Video-LLaMA2     | 8B      | _           | 54.6    | 47.9             | -                  |
| Video-ChatGPT    | 8B      | _           | 32.7    | -                | -                  |
| ShareGPT4Video   | 8B      | _           | -       | 39.9             | 39.7               |
| PLLaVA           | 8B      | _           | 46.6    | -                | 40.2               |
| Idefics2         | 8B      | _           | 29.7    | -                | 49.7               |
| Mantis-SigLIP    | 8B      | _           | 50.2    | -                | $\overline{47.0}$  |
| LLAVA-Interleave | 8B      | <u>78.2</u> | 53.1    | -                | -                  |
| mPLUG-Owl3       | 8B      | 78.6        | 54.5    | 53.5             | 52.1               |

Table 5: **Multi-modal evaluation on video understanding benchmarks.** The overall scores are reported for evaluation. We use **bold** to mark the highest score and <u>underline</u> to mark the second highest.

minute. mPLUG-Owl3 achieves performance comparable to state-of-the-art models. For benchmarks like VideoMME (Fu et al., 2024a) and LongVideoBench (Wu et al., 2024), with longer video durations up to one hour, mPLUG-Owl3 significantly outperforms existing models. It demonstrates that mPLUG-Owl3 is highly suitable for understanding videos with various durations.

Table 6 presents the the evaluation results on multi-image understanding. NLVR2 (Suhr et al., 2018) and Mantis-Eval (Jiang et al., 2024) test the model's ability to perform logical reasoning based on the content of multiple images. MathVerse-mv (Li et al., 2024a) and SciVerse-mv (Li et al., 2024a) evaluate the model's multi-image mathematical and scientific capabilities. We use the version released by llava-next-interleave for comparison with its reported results. BLINK (Fu et al., 2024b) and Q-Bench2 (Zhang et al., 2024d) test the model's multi-image question answering ability based on low-level visual perception. We compare mPLUG-Owl3 with models support image-text interleaved inputs such as Qwen-VL-Chat (Bai et al., 2023), InstructBLIP (Dai et al., 2023), CogVLM (Wang et al., 2023), VideoLLaVA (Lin et al., 2023a), VILA (Lin et al., 2023b), Idefics2 (Laurenccon et al., 2024), Mantis-SigLIP (Jiang et al., 2024) and LLAVA-Interleave (Li et al., 2024a).

mPLUG-Owl3 surpasses existing models in both NLVR2 and Mantis-Eval. On MathVerse-mv and SciVerse-mv, it can be observed that mPLUG-Owl3 significantly outperforms LLaVA-Interleave. However, on BLINK, mPLUG-Owl3 performs weaker than LLaVA-Interleave. We note that this dataset requires models to possess low-level visual perception capabilities for fine details in images, and mPLUG-Owl3's ability may be limited due to the vision encoder being frozen during training.

<span id="page-9-1"></span>On the Q-Bench2, which evaluates a model's ability to discern low-level visual differences across multiple images globally, mPLUG-Owl3 achieves performance comparable to the state-of-the-art.

<span id="page-9-0"></span>

| Model            | # Param | NLVR2 | Mantis-Eval | MathVerse-mv | SciVerse-mv | BLINK       | Q-Bench2 |
|------------------|---------|-------|-------------|--------------|-------------|-------------|----------|
| Qwen-VL-Chat     | 8B      | 58.7  | 39.2        | -            | -           | 31.2        | 45.9     |
| InstructBLIP     | 8B      | 60.3  | 45.6        | -            | _           | 42.2        | 44.3     |
| CogVLM           | 17B     | 58.6  | 45.2        | -            | -           | 41.5        | 53.2     |
| VideoLLaVA       | 8B      | 56.5  | 35.9        | -            | _           | 38.9        | 45.7     |
| VILA             | 8B      | 76.5  | 51.2        | -            | -           | 39.3        | 45.7     |
| Idefics2         | 8B      | 86.9  | 48.9        | -            | _           | 45.2        | 57.0     |
| Mantis-SigLIP    | 8B      | 87.4  | 59.5        | -            | _           | 46.4        | 69.9     |
| LLAVA-Interleave | 8B      | 88.8  | <u>62.7</u> | 32.8         | <u>31.6</u> | <b>52.6</b> | 74.2     |
| mPLUG-Owl3       | 8B      | 90.8  | 63.1        | 65.0         | 86.2        | 50.3        | 74.0     |

Table 6: **Multi-modal evaluation on multi-image understanding benchmarks.** The overall scores are reported for evaluation. We use **bold** to mark the highest score and <u>underline</u> to mark the second highest.

To more comprehensively investigate the fine-grained abilities of mPLUG-Owl3 in multi-image scenarios, we conduct experiments on MI-Bench (Liu et al., 2024c), a recently proposed multi-image benchmark. We exclude Fine-Grained Visual Recognition from evaluation because it consists of images from mini-ImageNet that may have been seen by existing models.

Table 7 shown that mPLUG-Owl3 achieves state-of-the-art performance on aspects of General Comparison, Subtle Difference, Temporal Reasoning, Logical Reasoning and Text-Rich Images across popular open-sourced MLLMs. It also outperform GPT-4V and GPT-40 on General Comparison. The results demonstrates that our model possesses robust capabilities in various multi-image input scenarios. The Hyper Attention structure of mPLUG-Owl3 better preserves the original visual features, enabling it to excel in single-image tasks as well. And this type of multimodal knowledge also assists it in more accurately completing multi-image tasks.

#### 4.4 Ablation Studies

We adopt the training methods of LLaVA-1.5 (Liu et al., 2024a) using the same datasets to conduct our ablation study. Additionally, we employ the Qwen1.5 7B as our language model. To validate the single-image understanding capabilities of our structures, we use datasets such as GQA and TextVQA (with OCR). Furthermore, we examine the generalization capabilities of our structures in multi-image understanding and video comprehension by conducting zero-shot evaluations on benchmarks including MvBench, VideoMME, NLVR2, and Mantis-Eval.

#### 4.4.1 Cross Attention Integration

There are two primary methods to integrate Cross-Attention into the transformer block: one method positions it prior to the self-attention (referred to as Pre-Cross-Attention), while the other places it subsequent to the self-attention (referred to as Post-Cross-Attention). We analyze both configurations and compare them to the concatenate-based method and our novel Hyper Attention in mPLUG-Owl3. Specifically, for Pre-Cross-Attention, it is positioned before the layer normalization at the input stage of the Transformer block. Conversely, for Post-Cross-Attention, it is positioned after the layer normalization that follows the self-attention stage. Both attention mechanisms employ a gating mechanism to fuse the multimodal representations effectively.

Table 8 shows that the concatenate-based model which directly embeds image features into the input sequence of the language model, has the best performance in single-image understanding. On the other hand, utilizing Post-Cross-Attention results in the worst performance. Comparatively, Pre-Cross-Attention performs better but still incurs some performance loss. Hyper Attention, however, achieves comparable performance with concatenate-based model.

In evaluations involving videos and multiple images, we observe that the concatenate-based model may not follow textual instructions as accurately, leading to a significant performance degradation in multi-image scenarios. This is attributed to the inadequate training of inter-image attention, which

<span id="page-10-3"></span><span id="page-10-0"></span>

| Model                                       | GC                           | SD                           | VR                           | TR                           | LR                           | TRI                          | VTK                          | TVK                          |  |  |
|---------------------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|--|--|
| Closed-source MLLMs                         |                              |                              |                              |                              |                              |                              |                              |                              |  |  |
| GPT-4o<br>GPT-4V                            | 80.7<br>72.8                 | 90.5<br>79.2                 | 46.8<br>45.8                 | 68.0<br>61.8                 | 69.8<br>66.3                 | 74.8<br>71.0                 | 54.7<br>52.0                 | 63.3<br>56.0                 |  |  |
| Open-source MLLMs                           |                              |                              |                              |                              |                              |                              |                              |                              |  |  |
| mPLUG-Owl2<br>MMICL<br>Idefics2-I<br>Mantis | 64.2<br>53.7<br>83.1<br>83.0 | 40.1<br>46.4<br>49.7<br>54.1 | 35.6<br>41.1<br>32.6<br>37.6 | 30.7<br>47.0<br>44.8<br>45.5 | 41.3<br>59.6<br>56.4<br>63.4 | 39.0<br>27.6<br>43.9<br>37.7 | 17.0<br>22.1<br>25.6<br>26.4 | 25.6<br>35.9<br>39.0<br>41.7 |  |  |
| mPLUG-Owl3                                  | 86.4                         | 70.1                         | 33.0                         | 46.8                         | 67.2                         | 50.1                         | 31.1                         | 48.8                         |  |  |

Table 7: **Multi-image evaluation on MI-Bench [\(Liu et al.,](#page-20-14) [2024c\)](#page-20-14)**. We use **bold** to mark the highest score of open-sourced multimodel large language models. The evaluation consists of the following tasks: General Comparison (GC), Subtle Difference (SD), Visual Referring (VR), Temporal Reasoning (TR), Logical Reasoning (LR), Text-Rich Images (TRI), and Vision-linked Textual Knowledge (VTK).

<span id="page-10-1"></span>

| Attention Structure  | GQA  | TextVQA | MvBench | VideoMME | NLVR2 | Mantis-Eval |
|----------------------|------|---------|---------|----------|-------|-------------|
| Concatenate          | 59.0 | 51.6    | 22.4    | 25.1     | 55.7  | 38.7        |
| Pre-Cross-Attention  | 53.8 | 45.2    | 43.0    | 38.9     | 55.3  | 44.7        |
| Post-Cross-Attention | 48.9 | 40.9    | 38.3    | 37.0     | 54.0  | 47.0        |
| Hyper Attention      | 57.6 | 50.0    | 42.8    | 39.4     | 59.5  | 51.6        |

Table 8: Comparison between different attention structure. Concatenate means direct concatenate visual and text feature sequences. We use **bold** to mark the highest score.

significantly disrupts the model's hidden states. Conversely, both single images and multiple images share the same paradigm when performing cross attention with text, which allows its multi-image capability to be better generalized from single-image training. the Hyper Attention design stands out as particularly effective in balancing the model's capabilities for handling both single and multiple images, showcasing superior generalizability

We also explore the integration position of the hyper attention. As shown in Table [9.](#page-10-2) The results indicate that even with just two layers of Hyper Attention, the model achieves impressive performance on single-image benchmarks, while also demonstrating generalization capabilities for videos and multiple images. However, when we apply a denser integration strategy by introducing eight layers of Hyper Attention, we find that it does not yield improved single-image performance at this scale of training data, and its zero-shot generalization is even worse. Therefore, we ultimately integrate only four layers into the entire model.

## 4.4.2 Design of Hyper Attention

To further investigate the impact of the structural design of Hyper Attention on model performance, we start with a basic hyper attention model and gradually introduce adaptive gating, shared layernorm, and MI-Rope. The Table [10](#page-11-0) shows that, when incorporate adaptive gating, the single-image

<span id="page-10-2"></span>

| Hyper Attention Layers Indices | GQA  | TextVQA | MvBench | VideoMME | NLVR2 | Mantis-Eval |
|--------------------------------|------|---------|---------|----------|-------|-------------|
| [9, 27]                        | 55.1 | 51.3    | 42.2    | 38.2     | 58.3  | 48.4        |
| [1, 5, 9, 13, 17, 21, 25, 29]  | 56.2 | 48.3    | 41.5    | 39.5     | 52.4  | 47.5        |
| [1, 9, 17, 25]                 | 57.6 | 50.0    | 42.8    | 39.4     | 59.5  | 51.6        |

Table 9: Comparison between different layers for integrating hyper attention structures. We use **bold** to mark the highest score.

<span id="page-11-1"></span><span id="page-11-0"></span>

| Adaptive Gating | Shared LayerNorm | MI-Rope | GQA  | TextVQA | MvBench | VideoMME | NLVR2 | Mantis |
|-----------------|------------------|---------|------|---------|---------|----------|-------|--------|
|                 |                  |         | 53.3 | 44.6    | 40.2    | 38.1     | 52.7  | 41.9   |
| ✓               |                  |         | 55.7 | 49.3    | 43.2    | 40.1     | 53.4  | 47.9   |
| ✓               | $\checkmark$     |         | 58.1 | 49.7    | 42.8    | 38.4     | 54.9  | 46.1   |
| $\checkmark$    | ✓                | ✓       | 57.6 | 50.0    | 42.8    | 39.4     | 59.5  | 51.6   |

Table 10: Ablation on the Adaptive Gating, Shared LayerNorm and MI-Rope.

understanding performance is significantly improved. And if we use a shared layernorm, performance is further improved. In video scenario, we notice that even without any inter-image position encoding, the performance of video understanding is also improved, suggesting the temporality inherent in visual content can also be implicitly modeled by the model with the help of adaptive gating. However, when evaluating models with multiple images, the contextual position of the images is crucial and cannot be implicitly modeled. Therefore, it can be observed that incorporating adaptive gating and shared layernorm does not lead to performance improvement on multi-image benchmarks. However, with the introduction of MI-Rope, the metrics for various multi-image benchmarks have demonstrated significant improvement.

#### 4.5 DISTRACTOR RESISTANCE IN LONG VISUAL CONTEXTS

Recent works adopt the multimodal needle in a haystack (Wang et al., 2024b) approach to evaluate the understanding of long sequences. However, we notice that multimodal models, when understanding multiple images, are susceptible to interference from surrounding images, leading to visual illusions. The multimodal needle in a haystack evaluation cannot detect such errors. Therefore, we develop a challenge evaluation method to assess the distractor resistance of multimodal models in long visual contexts.

Specifically, we take samples from the MMBench dev set. For each test sample, we randomly select N-1 images from the original MMBench dev set as distractor and construct the model input in the format of  $Image\ 1$ :  $<|image|>Image\ 2$ :  $<|image|>...\ Image\ N$ :  $<|image|>...\ Image\ N$ :  $<|image|>...\ Image\ X$ ,  $\{question\}$ , where N=1,5,10,20,50,100,200,400 and X denotes the index of the image corresponding to the question. We use the CircularEval to measure the accuracy scores. For each question, we construct test samples with different orders of options and varying distractor images. The model needs to answer all test samples for a given question correctly for it to be counted as correct. Consequently, as the number of distractor images increases, the evaluation becomes significantly more challenging.

We compare mPLUG-Owl3 with LLaVA-Next-Interleave 7B (Li et al., 2024a), Mantis-Idefics2 (Jiang et al., 2024), Qwen-VL (Bai et al., 2023) and mPLUG-Owl2 (Ye et al., 2024). LLaVA-Interleave-7B can handle approximately 20 images given 80GB of VRAM. By utilizing model parallelism, we extend its capacity for images to 50 images. However, LLaVA-Next-Interleave is unable to handle settings with more images. Mantis-Idefics2 can handle up to 100 images but costs 9 hours to finish the evaluation.

The results are shown in Figure 4. It can be observed that the introduction of distractor images results in a certain degree of performance loss for all the models. When the number of images reaches 20 and 50, the performance of LLaVA-Next-Interleave dramatically drops to 43.18% and 12.52%, respectively. We observe that when the number of images reaches 50, LLaVA struggles to consistently answer the questions accurately when different distractor images are present, resulting in a low accuracy rate. And when the number of images reaches 100, Mantis-Idefics2 fails to solve most of the problems correctly. In contrast, mPLUG-Owl3 only drops to a performance level of 43.09% when processing 50 images. As the number of images increases to 400, the performance of mPLUG-Owl3 decreases to 28.58%. Since our multi-image training data consists of only about 6-8 images, this also presents a challenge for our model. Nonetheless, mPLUG-Owl3 can serve as a baseline for future research.

#### 4.6 Qualitative Results

mPLUG-Owl3 can handle various number of images and videos as inputs. In this section, we further investigate the ability of mPLUG-Owl3 in real-world dialogue scenarios.

<span id="page-12-1"></span><span id="page-12-0"></span>![](_page_12_Figure_0.jpeg)

Figure 4: The performance of interference resistance with long visual context across LLaVA-Next-Interleave 7B [\(Li et al.,](#page-19-4) [2024a\)](#page-19-4), Mantis-Idefics2 [\(Jiang et al.,](#page-19-0) [2024\)](#page-19-0), Qwen-VL [\(Bai et al.,](#page-17-3) [2023\)](#page-17-3) mPLUG-Owl2 [\(Ye et al.,](#page-22-1) [2024\)](#page-22-1)) and mPLUG-Owl3.

