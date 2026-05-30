# <span id="page-4-0"></span>4 Experiments

#### 4.1 Baselines

Captioner-Reasoner We tested LLoVi [\[37\]](#page-10-7), which addresses video reasoning by tokenizing frames using VLMs such as BLIP-2 [\[20\]](#page-9-9). This efficient approach allowed LLoVi to achieve an accuracy of 67.7 on NExT-QA [\[10\]](#page-8-4). Given its success, LLoVi shows potential for handling more complex, long-range video QA tasks by effectively summarizing captions.

Large Multimodal Model Instruction Fine-tuning SEVILA [\[12\]](#page-9-8) introduces a two-stage pipeline that utilizes fine-tuned large multimodal models to localize keyframes and apply reasoning to selected frames, achieving an accuracy of 73.8 on NExT-QA with only 4 frames used for sparse sampling as inputs. Considering that TiM might require more input frames, we also incorporate LLaMA-VID [\[13\]](#page-9-1), which adopts a different strategy by projecting frames into two tokens to efficiently handle long-range video inputs.

Visual Programming ViperGPT [\[15\]](#page-9-3) leverages LLMs as a code generator that dynamically allocates VLMs and vision models, such as object detection, to progressively derive reasoning results. Although Viper may not always outperform LLoVi and SEVILA in terms of performance, it offers superior interpretability because the generated code illustrates how LLMs decompose tasks and perform stepwise reasoning.

Gemini 1.5 To assess the limits of machine learning models, we tested Gemini 1.5 [\[4\]](#page-8-1), a trillionscale model that significantly surpasses the size of previously mentioned models. This serves as a benchmark for future research.

#### 4.2 Proposed Method

In our initial approach to TiM, we enhanced Viper [\[15\]](#page-9-3) with two novel features designed to address Abstract Perception and Long-range Compositional Reasoning respectively.

Face-Enhanced Viper of Role Interactions (FEVoRI) Previous datasets have primarily focused on short, simple clips rather than movies featuring numerous characters with rich interactions. Consequently, the original Viper design lacked tools specifically aimed at role identification. FEVoRI augments Viper by providing a face detection tool with examples in the prompts. FEVoRI enhances the fine-grained understanding of the "human" object to address Abstract Perception[2](#page-4-2) .

<span id="page-4-2"></span>Implementation details in Appendix [B.1](#page-11-0)

<span id="page-5-2"></span>Table 2: State-of-the-art performance on TiM. everyshot: the model takes one frame per shot. SeViLA $^{\dagger}$ : SeViLA that uses the zero-shot localizer. 120 $\rightarrow$  16: SeViLA localizer selects 16 keyframes from 120 frames. 16(seViLA): Viper uses 16 frames selected by SeViLA localizer. FEVoRI $^{*}$ : evaluate on *Mainset*. Human: human evaluation result from [19]. we select Mainset as multi-modality setting for fair comparison

|                            |                          |                              |       |       |       | Category F1 |       |       |       |
|----------------------------|--------------------------|------------------------------|-------|-------|-------|-------------|-------|-------|-------|
| Modality                   | Method                   | # Frames                     | Pre.  | Rec.  | F1    | CT          | RI    | ST    | SL    |
|                            | Random                   | -                            | 12.24 | 48.48 | 19.54 | 19.23       | 19.99 | 17.37 | 23.37 |
|                            | LLoVi [37]               | everyshot                    | 20.47 | 17.67 | 18.97 | 13.46       | 16,67 | 15.22 | 25.58 |
|                            | SeViLA <sup>†</sup> [12] | $120 \rightarrow 16$         | 12.35 | 96.71 | 21.90 | 25.12       | 19.02 | 22.38 | 20.96 |
| V(Fullset)                 | SeViLA [12]              | $120 \rightarrow 16$         | 15.29 | 51.75 | 23.61 | 23.46       | 23.43 | 17.81 | 27.58 |
|                            | Viper [38]               | $16~(\text{SeViLA}^\dagger)$ | 13.26 | 67.33 | 22.15 | 21.58       | 22.63 | 19.92 | 24.60 |
|                            | Viper [38]               | 16 (SeViLA)                  | 14.09 | 68.70 | 23.39 | 21.41       | 24.62 | 20.90 | 26.85 |
|                            | FEVoRI*                  | 120                          | 27.07 | 32.32 | 29.42 | 12.36       | 22.75 | 35.62 | 48.78 |
|                            | Gemini 1.5 [4]           | 120                          | 38.37 | 34.42 | 40.74 | 40.45       | 38.79 | 38.55 | 45.11 |
|                            | Random                   | -                            | 14.14 | 50.08 | 22.06 | 20.26       | 21.24 | 19.50 | 23.92 |
|                            | LLoVi [37]               | everyshot                    | 31.35 | 17.21 | 18.78 | 20.20       | 24.40 | 35.95 | 40.63 |
|                            | SeViLA <sup>†</sup> [12] | $120 \rightarrow 16$         | 17.30 | 89.33 | 28.98 | 22.64       | 24.76 | 32.83 | 35.79 |
| V+D(Mainset <sup>4</sup> ) | SeViLA [12]              | $120 \rightarrow 16$         | 22.98 | 58.18 | 28.54 | 28.92       | 25.00 | 37.50 | 42.86 |
|                            | LLaMA-VID [3]            | 240                          | 15.56 | 90.12 | 26.53 | 25.72       | 24.60 | 28.31 | 38.15 |
|                            | Viper [38]               | $16~(\text{SeViLA}^\dagger)$ | 14.58 | 37.87 | 21.05 | 18.15       | 14.35 | 20.58 | 31.56 |
|                            | Viper [38]               | 16 (SeViLA)                  | 14.38 | 38.79 | 20.98 | 24.39       | 15.22 | 18.02 | 24.76 |
|                            | Viper [38]               | 120                          | 27.78 | 21.74 | 24.39 | 22.91       | 19.59 | 40.43 | 48.78 |
|                            | FEVoRI                   | 120                          | 27.88 | 39.80 | 32.79 | 30.52       | 29.55 | 42.42 | 49.67 |
|                            | FEVoRI+ConQueR           | 120                          | 32.11 | 51.28 | 39.64 | 42.80       | 34.48 | 39.78 | 55.17 |
| Synopses                   | Human [19]               | -                            | 65.77 | 63.98 | 64.87 | -           | -     | -     | -     |

Table 3: Ablation study on FEVoRI framework on TiM *Mainset*.

<span id="page-5-3"></span>

|        |          |           |        |                |       |       |                  | Category F1 |       |       |       |
|--------|----------|-----------|--------|----------------|-------|-------|------------------|-------------|-------|-------|-------|
|        | Modality | # Frames  | VLM    | Coder          | Pre.  | Rec.  | F1               | CT          | RI    | ST    | SL    |
| FEVoRI | V+D      | 120       | BLIP-2 | GPT-4          | 27.88 | 39.80 | 32.79            | 30.52       | 29.55 | 42.42 | 49.67 |
|        | V        | 120       | BLIP-2 | GPT-4          | 27.07 | 32.23 | 29.42 ((-3.374)) | 12.36       | 22.75 | 35.62 | 48.00 |
|        | V+D      | everyshot | BLIP-2 | GPT-4          | 27.27 | 46.15 | 34.29 ((+1.50†)) | 33.30       | 30.12 | 44.68 | 50.00 |
|        | V+D      | 16        | BLIP-2 | GPT-4          | 25.71 | 40.72 | 31.52 ((-1.271)) | 23.74       | 25.56 | 38.83 | 47.54 |
|        | V+D      | 120       | Gemini | GPT-4          | 29.37 | 51.15 | 37.31 ((+4.52†)) | 28.71       | 29.49 | 47.17 | 53.23 |
|        | V+D      | 120       | BLIP-2 | <b>GPT-3.5</b> | 30.16 | 35.52 | 32.62 ((-0.171)) | 27.18       | 30.34 | 39.56 | 38.65 |

**Context Query Reduction (ConQueR)** Viper [15] processes NExT-QA [10] by temporally locating frames or objects and querying the VLM about them. This approach struggles with TiM due to the intricate narratives of movies and the complex definitions of tropes. ConQueR addresses the Long-range Compositional Reasoning challenge by progressively decomposing the narrative context and trope query. It systematically checks if the extracted context matches each dimension of a trope through the generated program<sup>3</sup>.

#### 4.3 Setup

Most models in our experiments are training-free, so the entire TiM dataset is used for testing. Additionally, we fine-tuned SeViLa on TiM in a supervised setting to evaluate its performance. For these experiments, we employed five-fold cross-validation and reported the average performance. For LLoVi [9], we employ the standard prompt with BLIP-2 [20] to generate captions for each frame in every shot of TiM. This is followed by a multi-round summarizing process to create a summary for each movie. These summaries, coupled with binary classification queries, are then inputted into an LLM to generate answers. In the multi-modality version, we enhance the summarization process by integrating captions with subtitles. For SeViLA [12], we use NExT-QA setting for both zero-shot and fine-tuned scenarios, enabling the Localizer to select 16 frames from a set of 120. These selected frames are used to address binary classification queries, with an enhancement in the multi-modality version where visual features are concatenated with subtitles before being processed by the LLM

<span id="page-5-1"></span><sup>&</sup>lt;sup>3</sup>Implementation details in Appendix B.2

<span id="page-5-0"></span><sup>&</sup>lt;sup>4</sup>Performance difference between Mainset and VDset in Appendix A

during both the localizer and answerer stages. For LLaMA-VID [\[3\]](#page-8-6), we use long-video-tuning model, which was tuned with QA pair from MovieNet [\[36\]](#page-10-6), to inference on binary classification query on TiM. For Viper and our proposed method, we have adapted the NExT-QA prompt on TiM and use GPT-4 as the code generator. For FEVoRI, we have integrated additional face identification tools into the Viper API, specifically employing DeepFace [\[39\]](#page-10-10) for face recognition.

#### 4.4 Existing LLM-based state-of-the-arts cannot reason on TiM

As shown in Table [2,](#page-5-2) all LLM-based baselines struggle with reasoning on TiM, achieving only random-level performance (first row of each block). This underscores that despite their significant achievements on various video reasoning benchmarks, state-of-the-art models are unable to overcome the challenges posed by TiM. Access to dialogues results in an F1 score improvement of 2-4 points. Captioner-Reasoner (LLoVi [\[9\]](#page-8-3)) records lower F1 scores, indicating that the loss of information or the abstraction gap during video captioning may lead to subpar performance on TiM. LLoVi also achieves relatively better performance in the Storyline (SL) category, which focuses on the overall plot rather than on fine-grained details. LMM-IF methods, including SeViLa [\[12\]](#page-9-8) which achieves significant performance on various benchmarks, and LLaMA-VID [\[13\]](#page-9-1) designed for long videos, often resort to blindly guessing "yes." This approach typically results in high recall but poor precision. Fine-tuning SeViLa enhances performance through supervised learning. Viper [\[15\]](#page-9-3) achieves decent performance without resorting to blindly guessing "yes," and maintains superior precision compared to SeViLa and LLaMA-VID. Gemini [\[4\]](#page-8-1) achieves a 41 F1 score, surpassing all previously mentioned methods due to its larger scale of parameters and training data. However, it still significantly trails human performance [\[19\]](#page-9-7), scoring 41 compared to 65 F1. Comprehensive experiments show that SOTA LLMs still struggle to address challenges in TiM.

#### 4.5 FEVoRI Analysis

FEVoRI significantly boosts the F1 score by 8.5. Comparing Viper and FEVoRI in the second block of Table [2,](#page-5-2) our augmentation allows the VP LLM to understand character interactions, leading to substantial performance improvements, particularly in the CT (Character Traits) and RI (Role Interaction) categories, where fine-grained role interactions are crucial. Remarkably, even the visualonly FEVoRI outperforms the supervised SeViLA [\[12\]](#page-9-8), demonstrating the superior design of our methodology. As the performance gain is primarily from an 18.00 improvement in recall, while precision improves by only 0.1, we hypothesize that FEVoRI improves by effectively identifying more relevant cases.

ConQueR further increases the F1 score by 6.9. In the second block of Table [2](#page-5-2) , comparing FEVoRI and FEVoRI+ConQueR, the modified ConQueR demonstrates how progressively decomposing the trope query and movie narrative context enhances understanding. ConQueR also effectively filters key signals to extract crucial information from long-range videos. The performance improvement highlights promising directions for future work in addressing Long-range Compositional Reasoning.

A higher frame rate consistently outperforms sparse sampling. Several tropes depend on fleeting, fine-grained details or a comprehensive understanding of the entire plot. We compared the density of frame sampling by evaluating every shot (approximately 1,000 frames) and 120 frames per video, alongside a sparse sampling method that uses only 16 frames per video, which is commonly used in many approaches. As shown in Table [3,](#page-5-3) a higher frame rate leads to marginal yet consistent improvements, with every-shot sampling boosting the F1 score by 2.8 points across all categories. This indicates that while sparse sampling is efficient, it may compromise performance.

Enhancing VLM Abstract Perception improves performance by 4.5. A core challenge of TiM is Abstract Perception, which involves tokenizing visual signals into coherent concepts. Table [3](#page-5-3) shows that replacing BLIP-2 [\[20\]](#page-9-9) with more advanced Gemini [\[4\]](#page-8-1), the F1 score is boosted by 4.5 as Gemini is capable to tackle more abstract queries.

GPT-4 shows a slight improvement over GPT-3.5 in program generation. When replacing GPT-4 with GPT-3.5, the F1 score drops by 0.2, as shown in Table [3,](#page-5-3) demonstrating that GPT-3.5 is capable to generate programs without ConQueR.

<span id="page-7-2"></span>Table 4: We propose an AST Based Code Dignosis (ABCD) to assess the levels of Abstract Perception and Long-range Compositional Reasoning in a dataset, using code generated by VP. A higher number indicates greater complexity and challenge. (Section [5\)](#page-7-0)

|                   |           | Abstract Perception | Long-range Compositional Reasoning |           |  |  |  |
|-------------------|-----------|---------------------|------------------------------------|-----------|--|--|--|
| Dataset           | VLM Calls | VLM Tokens          | AST Nodes                          | AST Edges |  |  |  |
| NExT-QA [10]      | 1.60      | 11.15               | 102.09                             | 146.32    |  |  |  |
| GQA [17]          | 1.34      | 12.69               | 42.16                              | 55.63     |  |  |  |
| OKVQA [18]        | 1.66      | 13.75               | 42.50                              | 58.46     |  |  |  |
| TiM (w/o ConQueR) | 1.77      | 14.11               | 123.19                             | 178.01    |  |  |  |
| TiM (w/ ConQueR)  | 1.97      | 20.67               | 141.81                             | 205.06    |  |  |  |

