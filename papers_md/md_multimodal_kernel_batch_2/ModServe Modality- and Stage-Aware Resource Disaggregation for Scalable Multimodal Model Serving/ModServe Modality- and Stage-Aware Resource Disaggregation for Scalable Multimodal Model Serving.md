# ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving

Haoran Qiu Microsoft Azure Research, USA haoran.qiu@microsoft.com

Jayashree Mohan Microsoft Research India, India jamohan@microsoft.com

Íñigo Goiri Microsoft Azure Research, USA inigog@microsoft.com

Chetan Bansal Microsoft M365 Research, USA chetanb@microsoft.com

Anish Biswas Microsoft Research India, India t-anibiswas@microsoft.com

Alind Khare Microsoft M365 Research, India alindkhare@microsoft.com

Zeyu Zhang University of Virginia, USA qxc4fh@virginia.edu

Ramachandran Ramjee Microsoft Research India, India ramjee@microsoft.com

Zihan Zhao University of Virginia, USA rxy6cc@virginia.edu

Esha Choukse Microsoft Azure Research, USA esha.choukse@microsoft.com

Haiying Shen University of Virginia, USA hs6ms@virginia.edu

Rodrigo Fonseca Microsoft Azure Research, USA fonseca.rodrigo@microsoft.com

# Abstract

Large multimodal models (LMMs) demonstrate impressive capabilities in understanding images, videos, and audio beyond text. However, efficiently serving LMMs in production environments poses significant challenges due to their complex model architectures and heterogeneous characteristics across their multi-stage inference pipelines and modalities.

We present the first comprehensive systems analysis of two prominent LMM architectures, decoder-only and cross-attention, across six representative open-source models, revealing key systems design implications. We also present an in-depth analysis of production LMM inference traces, uncovering unique multimodal workload characteristics, including variable, heavy-tailed request distributions and bursty traffic patterns.

Based on these insights, we propose ModServe, a modular LMM serving system that decouples model stages for independent optimization and adaptive scaling. ModServe dynamically reconfigures stages and handles bursty traffic with modality-aware scheduling and autoscaling to meet tail latency SLOs while minimizing costs. ModServe achieves 3.3–5.5× higher throughput (leading to 25– 41.3% cost saving) while meeting SLOs on a 128-GPU cluster with production multimodal traces.

# 1 Introduction

The rapid advancement in generative AI has led to the development of large multimodal models (LMMs) capable of processing inputs across various modalities such as text, image, video, and audio. These models have demonstrated remarkable capabilities in tasks like image captioning [\[6,](#page-11-0) [21,](#page-12-0) [40\]](#page-12-1), visual question answering [\[51,](#page-12-2) [52\]](#page-12-3), multimodal dialogue systems [\[9,](#page-12-4) [30,](#page-12-5) [58\]](#page-13-0), and voice assistant [\[18\]](#page-12-6). This has led to a rapid adoption of LMMs in production services, including online user-facing applications where latency servicelevel objectives (SLOs) are critical.

Unlike traditional large language models (LLMs) that process purely textual inputs using a single component, a decoder-based

<span id="page-0-0"></span>![](_page_0_Figure_21.jpeg)

Figure 1: Impact of image/video workload on LMM inference TTFT for state-of-the-art implementation of Llama3.2-11B on vLLM vs. ModServe with an 8-A100 GPU server. The "Monolith" setup deploys the full model using 8 GPUs while the "Decoupled" setup deploys the LLM backend on 4 GPUs and four image encoders on the other 4 GPUs.

transformer architecture [\[61\]](#page-13-1), LMMs handle fundamentally different types of inputs, each requiring distinct processing approaches. This heterogeneity introduces unique serving complexities that demand novel analysis and serving strategies. For Image-Text-to-Text models [\[25\]](#page-12-7), the inference pipeline consists of multiple specialized stages: image preprocessing to transform raw images into tensor representations, image encoding to convert these tensors into image tokens, and a language model backend that combines text prompts with image tokens to generate text outputs. Currently, these stages are typically served as a monolithic system [\[5,](#page-11-1) [27,](#page-12-8) [62\]](#page-13-2), where all components are integrated within a single serving instance and scaled together as a unified entity. While recent LLM serving systems adopt prefill-decode (PD) disaggregation [\[46,](#page-12-9) [65\]](#page-13-3) to reduce the performance interference at the LLM backend, these optimizations remain text-centric and overlook the upstream stages of multimodal preprocessing and encoding, which are still tightly coupled within a monolithic serving instance.

This lack of modality-aware decoupling limits the efficiency and scalability of existing monolithic inference serving systems when serving workloads beyond text. These systems struggle to meet

1

time-to-first-token (TTFT) SLOs because multimodal input preprocessing and encoding are in the critical path. Figure 1 illustrates the challenges faced by a monolithic deployment in scaling as the number of images per request increases (a common scenario in multi-image or video workloads), resulting in sharp TTFT degradation. As a result, image-heavy requests can result in head-of-line (HoL) blocking, reducing system responsiveness and causing resource overprovisioning.

**Our Work.** In this paper, we present the first comprehensive systems analysis of two leading LMM architectures: cross-attention (*CroAttn*) and decoder-only (*DecOnly*), on both open-source LMMs and novel production LMM inference traces in Azure datacenters. We analyze their multi-stage inference pipelines, performance-resource tradeoffs, and production workload patterns, including variable request rates, diverse multimodal inputs, and bursty traffic. To advance research in this area, we release the *first open-source multimodal trace* from our production clusters, enabling the community to study real-world deployment patterns.

Our analysis identifies three key insights for optimizing LMM inference. First, different LMM inference stages exhibit diverse performance characteristics and varying sensitivity to resource and model configurations (e.g., batching and model sharding), necessitating decoupled execution. Second, image encoding is a major bottleneck for TTFT, requiring efficient encoder parallelization to reduce both latency and HoL blocking. Finally, production multimodal traffic exhibits distinct bursty patterns driven by increased images per request, highlighting the need for modality-aware routing strategies to manage bursts and mitigate tail latency spikes.

Based on these insights, we propose Modserve, a novel *modular architecture* for scalable and resource-efficient LMM serving which directly addresses the challenges identified in our analysis. Take Image-Text-to-Text tasks as an example, Modserve separates image- and text-specific inference stages into distinct instances for decoupled execution. In Modserve, *Image Instances* handle image preprocessing and encoding, while *Text Instances* manage LLM prefill and decoding (Figure 1). Text-only requests are served by *Text Instances*, whereas image-text requests go through *Image Instances* where images are converted to tokens before being forwarded to *Text Instances* for text generation.

ModServe's modular architecture unlocks stage-specific optimizations. ModServe manages *Image* and *Text Instances* independently with stage-aware autoscaling, model sharding, and batching. By autoscaling the stages separately, it minimizes resource overprovisioning. For example, during image-driven bursts observed in production traffic, *Image Instances* can scale out independently, making ModServe more resource-efficient than monolithic inference systems. To navigate the image encoding bottleneck, ModServe parallelizes encoding of a single request across multiple *Image Instances* (Figure 1), leveraging our finding that the images within a request do not attend to each other during encoding, and hence the requests can be parallelized at the image level.

Further, to manage image-driven bursts, ModServe implements modality-aware request routing for *Image and Text Instances*. For example, images from image-text requests are routed to *Image Instances* with the fewest pending image tokens to encode, reducing HoL blocking and tail latency spikes.

We implement ModServe on top of a high-performance inference system, vLLM [27], and demonstrate the effectiveness of ModServe through extensive evaluations on a 16-server (128 GPUs) cluster running production LMM inference traces from Azure. Compared to state-of-the-art baselines, ModServe achieves 3.3–5.5× higher throughput under static allocation and reduces LMM serving cost by 25–41.3% while meeting the P99 TTFT SLOs.

While existing techniques like PD disaggregation have shown promise for optimizing LLM inference by separating prefill and decode phases [46, 65], they fall short for multimodal workloads. The unique nature of bursty and heterogeneous multimodal traffic observed in production makes it nontrivial to extend PD disaggregation studied in the context of text-centric LLM inference; modality-specific optimizations such as parallelizing encoding, modality-aware routing, and stage autoscaling are crucial to meet SLOs while maintaining resource efficiency. Additionally, the choice of selecting PD disaggregation or colocation for the LLM backend depends on the workload conditions and optimization targets [65]. We show in Section 5 that Modserve is composable with both colocated LLM backends (mixed PD batching) as well as PD disaggregation for the text nodes, and improves serving latency with the proposed modality-aware techniques in both LLM backend configurations.

We focus on Image-Text-to-Text and Video-Text-to-Text (where videos are processed as image frame sequences [31]), but our insights extend to other multimodal scenarios, such as Audio-Text-to-Text tasks [23], which share similar model architectures and inference stages with the models we study.

**Summary.** This paper makes the following contributions:

- The first open-source dataset containing large-scale production LMM inference traces from Azure [16].
- A comprehensive system characterization on LMM serving, examining performance profiles and resource utilization patterns across diverse multimodal workloads in both open-source LMM deployments and production environments.
- Design and implementation of ModServe, a modular architecture for scalable and resource-efficient LMM serving.
- A thorough evaluation of ModServe in a 128-GPU cluster using large-scale production traces.

#### <span id="page-1-0"></span>2 Large Multimodal Models Background

LMMs extend text-centric LLMs by integrating multimodal understanding capabilities for tasks like visual question answering [52] and computer-using agents [42, 45]. Figure 2 shows the typical pipeline of LMM inference in visual understanding tasks [25], which consists of three key stages: (1) *image preprocessing*, where raw images are transformed into uniform-sized tiles; (2) *image encoding*, where an encoder extracts visual features and produces a sequence of image tokens; and (3) *text generation*, where an LLM backend processes the image and text tokens to generate output text tokens. There are two dominant LMM architectures that differ in how the LLM backend handles image tokens and text tokens: (1) *decoder-only* (DecOnly), used in models like DeepSeek's Janus [7], LLaVA-OneVision [31], InternVL [9], and NVLM-D [12]; and (2) *cross-attention-based* (CroAttn), found in Llama-3.2 Vision [11],

<span id="page-2-0"></span>![](_page_2_Figure_1.jpeg)

Figure 2: Model architecture for decoder-only and cross-attention-based LMMs in Image-Text-to-Text tasks [25].

NVLM-X [12], and Flamingo [3]. In this work, we analyze six open-source LMMs (listed in Table 1) across these architectures, varying image encoder sizes (400M–6B) and LLM scales (7B–72B).

**Image Preprocessing.** LMMs typically follow three preprocessing steps on CPU: (1) resize, rescale, pad, and normalize the raw image, (2) segment it into tiles [9, 11, 12] or patches [31], and (3) apply tile/patch-level transformations and sampling. The number of tiles varies, with higher image dimensions resulting in more tiles, which ultimately increases the number of image tokens. For example, an image with 896×896 pixels generates 4, 5, or 10 tiles of different sizes after preprocessing for six open-source LMMs (Table 1).

**Image Encoding.** The image encoder takes processed image tiles as input and produces image tokens that are then passed to the language model backend. Today's image encoders predominantly use the vision transformer architecture [4] to extract visual features from images. Table 1 shows that different LMMs use different encoders [4, 9, 31, 64], leading to variations in the number of image tokens when running image encoders on the same ShareGPT-40 dataset [8] (Figure 3). This is due to differences in the number of tiles and image tokens generated per tile by each encoder.

**Text Generation.** Image and text prompt tokens are combined and passed through LLM prefill and decode to generate output tokens, typically using one of two architectures:

Decoder-Only (DecOnly) LMMs. An unmodified LLM backend is reused in DecOnly LMMs (e.g., LLaVA-OV reuses Qwen2 LLM [31]), processing text and image tokens uniformly (shown as the "DecOnly" box in Figure 2). This works by attaching a connector [66] or modality-alignment module (e.g., MLP) that maps the image encoder output into the LLM's token space. While valued for their simplicity and unified modality handling, DecOnly models often require long sequences for high-resolution images, resulting in computational inefficiencies during inference.

Cross-Attention (CroAttn) LMMs. Unlike DecOnly LMMs, which leave the LLM backend unchanged, CroAttn-based models (e.g., Llama-3.2 Vision) integrate cross-attention layers to process image

<span id="page-2-1"></span>![](_page_2_Figure_9.jpeg)

Figure 3: Distribution of image token count (per request) for open-source LMMs on ShareGPT-40 dataset [8]. Different LMMs (e.g., LLaVA-OV 7B and 72B) can share the same image encoder so the number of image tokens is the same.

tokens, treating visual inputs like a "foreign language" in the LLM backend. While more complex to train, they improve inference efficiency by avoiding full image token unrolling in the LLM decoder, making them ideal for high-resolution inputs. Self-attention operates on text tokens, while the cross-attention layer attends to both text and image tokens ("CroAttn" box in Figure 2).

SLO Metrics for LMM Inference. Production LMM serving systems need to satisfy SLOs defined on tail latency (e.g., P99) for worst-case performance. These SLO metrics include *Time to First Token (TTFT)* and *Time Between Tokens (TBT)*. TTFT measures the latency from query (text/images) to the first response token, critical for interactive applications. In contrast to text-centric LLM serving, LMM TTFT includes the following stages of LMMs inference pipeline: (1) image preprocessing, (2) encoding, and (3) language model prefill time. TBT captures the delay between consecutive token generations during decoding, affecting output fluency. As multimodal preprocessing and encoding primarily influence TTFT, in this work, we focus on TTFT while leveraging state-of-the-art techniques [1, 46] to optimize TBT. An ideal LMM-serving system should meet TTFT/TBT SLOs while maximizing request throughput (i.e., goodput) and compute *utilization* (GPU cost).

LMM Deployments Today. State-of-the-art serving frameworks [5, 27, 62] deploy LMMs as monolithic systems to meet latency SLOs. In this setup, all inference components (i.e., image preprocessor, image encoder, and LLM backend) are co-located on the same hardware server as a single unit. While PD disaggregation can be applied within this monolithic setup to decouple prefill from decode phases and reduce interference during text generation, it still couples multimodal components with the prefill instances [27], preventing independent optimization of each component based on their distinct resource requirements and performance characteristics. These tightly coupled components share uniform batching and model parallelism strategies across the pipeline. Table 1 details the default model parallelism for our open-source LMMs. While this monolithic design is straightforward to implement and common in opensource LMM serving, it limits flexibility and suffers from sharp TTFT degradation under image-heavy workloads (Figure 1).

#### 3 Motivation and LMM Characterization

To further understand the limitations of monolithic deployments and explore unique characteristics that distinguish LMM serving from text-centric LLM serving, we characterize open-source LMMs

| LMM Model Name            | Abbreviation | Architecture    | Tile Size | 0               | Total Image Token Size<br>  (#Tiles × #TokensPerTile) | LLM Backend<br>(#Params) | Tensor<br>Parallelism | Avgerage Accuracy<br>(Open VLM Benchmark [15]) |
|---------------------------|--------------|-----------------|-----------|-----------------|-------------------------------------------------------|--------------------------|-----------------------|------------------------------------------------|
| Llama 3.2 Vision 11B [37] | Llama3.2-11B | Cross-attention | 560×560   | ViT-H/14 (630M) | 4 × 1601 × 1 = 6404                                   | Llama 3.1 (8B)           | TP-4                  | 57.8%                                          |
| Llama 3.2 Vision 90B [38] | Llama3.2-90B | Cross-attention | 560×560   | ViT-H/14 (630M) | 4 × 1601 × 1 = 6404                                   | Llama 3.1 (70B)          | TP-8                  | 63.4%                                          |
| LLaVA-OneVision 7B [34]   | LLaVA-OV-7B  | Decoder-only    | 384×384   | SigLIP (400M)   | $10 \times 729 \times 1 = 7290$                       | Qwen2 (7B)               | TP-4                  | 60.1%                                          |
| LLaVA-OneVision 72B [33]  | LLaVA-OV-72B | Decoder-only    | 384×384   | SigLIP (400M)   | $10 \times 729 \times 1 = 7290$                       | Qwen2 (72B)              | TP-8                  | 68%                                            |
| InternVL-2.5 26B [10]     | InternVL-26B | Decoder-only    | 448×448   | InternViT (6B)  | 5 × 256 = 1280                                        | InternLM (20B)           | TP-8                  | 71.6%                                          |
| NVLM-D 72B [13]           | NVLM-D-72B   | Decoder-only    | 448×448   | InternViT (6B)  | 5 × 256 = 1280                                        | Qwen2-Instruct (72B)     | TP-8                  | 67.6%                                          |

<span id="page-3-0"></span>Table 1: Model configurations for six representative open-source LMMs with an example input image of 896 × 896 pixels.

<span id="page-3-2"></span>![](_page_3_Figure_3.jpeg)

Figure 4: Image dimension distribution and text prompt length distribution of ShareGPT-40 Image dataset [8].

in the *Image-Text-to-Text* category [25]. We evaluate the performance and resource characteristics of heterogeneous inference stages under varying image inputs and model configurations (Section 3.1). Moreover, to understand multimodal traffic patterns at scale, we analyze sample production traces from one production LMM inference cluster at Azure (Section 3.2).

#### **Characterization Setup.** The following is our setup:

Models. We use six representative open-source LMMs across two different architectures (DecOnly and CroAttn) as listed in Table 1. We deploy the models on vLLM [27] in BF16 with the default, stable PD colocated mode because the experimental PD disaggregated mode of vLLM has limited support for multimodality, and no support for cross-attention models. However, we later show in Section 5.4 that PD disaggregation is an optimization that is trivially composable with our design in ModServe.

*Dataset.* We use the open-source ShareGPT-40 dataset [8], which includes 50K images of varying resolutions and text prompts from multimodal GPT-40 as shown in Figure 4.

Hardware. Our setup features a DGX-A100 server with 8 NVIDIA A100 GPUs (80GB each) connected via NVLINK [39]. It has 96 AMD Epyc™ 7V12 CPU cores and 1900 GiB DRAM.

# <span id="page-3-1"></span>3.1 Characterization on Open-Source LMMs

We characterize open-source LMMs to understand how different inference stages impact performance and resource efficiency. Additionally, we compare DecOnly and CroAttn models to highlight the need for model-specific optimization.

**Per-stage Latency Breakdown.** Figure 5 plots the split-up of TTFT across the three stages that comprise it; image preprocessing, image encoding, and LLM prefill. There are three key takeaways. First, image preprocessing, which occurs on the CPU, contributes minimally to the overall TTFT, while image encoding time contributes to a major portion in TTFT (especially for CroAttn models). For

instance, 79% and 65% of TTFT in Llama3.2-11B and Llama3.2-90B are from image encoding. For DecOnly models such as InternVL-26B and NVLM-D-72B, image encoding latency accounts for 25% and 54% of TTFT. Second, the image encoding time depends on the encoder model size. For instance, scaling from SigLIP-400M (in LLaVA-OV-7B) to InternViT-6B (in InternVL-26B), the median image encoding time increases by  $10\times$ . Since connectors [66] are extremely small (e.g., < 0.1% of total parameters in InternVL-26B), they contribute negligible latency (< 0.4% of TTFT). Finally, prefill computation is more efficient in CroAttn models because image tokens are attended to only in the CroAttn layers while the majority of LLM backends are self-attention layers, as described in Section 2.

Taken together, these results underline why image encoding emerges as a major bottleneck in multimodal model serving. First, the number of images per request can be substantial, especially in video understanding or multi-image tasks. Second, CroAttn models shift the computational load to the image encoder by reducing image-text interaction in the LLM. This lowers the computational load from LLM and keeps it in encoding.

<span id="page-3-3"></span><u>Insight 1:</u> A major portion of the TTFT is spent on image encoding, particularly for CroAttn models, making image encoding optimization critical to meet TTFT SLOs.

Compute Characteristics of LMM Stages. Image preprocessing on CPU and image encoding on GPU are compute-intensive processes. Figure 6a plots the impact of varying the number of CPU cores on preprocessing latency. Preprocessing is CPU-intensive and benefits from trivially parallelizing across cores. Both stages exhibit linear latency scaling with batch size, saturating compute without significant throughput gains from increased batching as shown in Figures 6b and 6c, respectively. Figure 6d further plots the GPU utilization metrics for a request batch size of one during image preprocessing and image encoding. We observe a consistent SM core activity near 100% during image encoding, with average DRAM utilization below 30%. Image encoding is, therefore, typically compute-bound, resembling the language model's prefill phase [26]. Moreover, when a request has multiple images in the input prompt (e.g., video workloads), there is typically no compute dependency between the images during image encoding; hence, image tiles can be parallelized across multiple encoders.

<span id="page-3-4"></span><u>Insight 2:</u> Image preprocessing and encoding are both computeintensive similar to LLM prefill stage. The independence of image computations in a multimodal request enables parallelization of image preprocessing and encoding across multiple instances.

4

<span id="page-4-0"></span>![](_page_4_Figure_1.jpeg)

Figure 5: Per-stage request latency breakdown analysis across representative open-source LMMs deployed using default tensor parallelism (TP) as described in Table 1. TTFT (dashed line) is the sum of the latency from each inference stage.

<span id="page-4-1"></span>![](_page_4_Figure_3.jpeg)

(a) Image preprocessing time (b) Image preprocessing time (c) Image encoding time varyvarying CPU allocation. varying batch size. ing batch sizes. (d) GPU utilization during image preprocessing and encoding.

Figure 6: Compute characteristics of image preprocessing and encoding. Both stages are compute-bound.

<span id="page-4-2"></span>![](_page_4_Figure_6.jpeg)

Figure 7: LMM accuracy vs. prefill/TTFT efficiency.

Compute characteristics of prefill and decode phases of the LLM backend have been well studied; the prefill phase is typically compute-bound, while the decode phase is memory-bound [1, 26, 46]. However, Figure 5 shows that LLM prefill is more efficient in CroAttn models than in DecOnly models, resulting in reduced compute intensity and an interesting tradeoff we describe below.

Latency-Accuracy Profiles across LMMs. Figure 7 shows the accuracy versus prefill/TTFT efficiency for different models. When comparing models with similar language model backend sizes across both architectures (*e.g.*, Llama3.2-11B vs. LLaVA-OV-7B and Llama3.2-90B vs. LLaVA-OV-72B vs. NVLM-D-72B), we observe that CroAttn models typically have up to an order of magnitude lower LLM prefill time, leading to lower TTFT. However, the CroAttn models usually achieve 5 points lower accuracy compared to their DecOnly counterparts on the Open VLM leaderboard [15]. For example, Llama3.2-90B scores 63.4, while the similarly sized LLaVA-OV-72B scores 68, but with significantly higher prefill latency and TTFT than Llama3.2-90B.

<span id="page-4-3"></span>![](_page_4_Figure_10.jpeg)

Figure 8: Median latency vs. batch size per LMM stage on GPUs. Latency is normalized to that at batch size one.

<span id="page-4-4"></span><u>Insight 3:</u> DecOnly models exhibit 10× worse LLM prefill latency than similar-sized CroAttn models, leading to less TTFT SLO headroom for the image encoding and thus necessitating higher scalability for image workloads.

**Impact of Batching.** In today's monolithic deployments, a single batch size is applied across all stages of the LMM on the GPU, which does not strike a balance between latency and throughput given heterogeneous compute characteristics observed across different stages. Figure 8 shows how the batch size affects the median latency of each LMM stage across architectures. As the batch size increases, the latency grows at varying rates, reflecting each stage's differing sensitivity to batch size and compute intensity.

Compute-intensive stages like image encoding and LLM prefill (in DecOnly models with longer image token inputs) show limited throughput gains and rising latency beyond small batch sizes. In contrast, the memory-bound decode stage benefits from linear throughput scaling. Due to their low text token count, CroAttn models uniquely gain from prefill batching, diverging from traditional LLM trends where prefill saturates compute even at a batch

<span id="page-5-1"></span>![](_page_5_Figure_1.jpeg)

Figure 9: Impact of the tensor parallelism (TP) degree on the median latency of each stage for CroAttn-based and DecOnly LMMs. Latency is normalized to that of TP-8.

size of one. Notably, DecOnly model NVLM-D with fewer image tokens also exhibits certain benefits in batching.

<span id="page-5-4"></span><u>Insight 4:</u> The effectiveness of batching varies for each LMM component and is model-specific. LMM request batching should thus be tailored to each stage.

Impact of Parallelism. Monolith deployments also limit the flexibility of model sharding within a GPU server which is typically done through tensor parallelism (TP) [59]. Figure 9 shows how increasing TP degrees affects latency across LMM stages. In Llama3.2-11B, the lowest LLM prefill latency occurs at TP-8, image encoding at TP-4, and TBT at TP-1. At TP-8, encoding latency rises due to the tradeoff between compute intensity and inter-GPU communication, making it inefficient to split a 630M-sized encoder across 8 devices.

In contrast, NVLM-D-72B, with a larger 6B image encoder, sees a 1.3× latency reduction when increasing TP from 4 to 8. However, this comes with diminishing returns relative to resource cost. To balance throughput and latency, operators can deploy two TP-4 encoders for higher throughput or one TP-8 encoder for lower latency, both using eight GPUs.

<span id="page-5-5"></span><u>Insight 5:</u> Treating the image encoder and LLM backend as a monolith limits parallelism flexibility and degrades performance. Decoupling them enables independent scaling and optimized efficiency through pipelining.

## <span id="page-5-0"></span>3.2 Production LMM Trace Analysis

Building on the insights from open-source LMM characterization, we further analyze multimodal traffic patterns at scale, leveraging production traces from one of Azure's LMM inference clusters. The traces capture a sample of multi-tenant traffic, including both text-only and image-text requests. Our study focuses on (1) temporal and burstiness patterns and (2) heterogeneity of multimodal requests. We have released these multimodal inference traces at https://github.com/Azure/AzurePublicDataset.

**Temporal Patterns and Burstiness.** Figure 10 shows the traffic of text-only and image-text requests separately to understand their dynamic behavior and overall impact on the system. The traces are collected over a span of one week. To understand the traffic patterns, we report the timeline of prompt (input) token rate, output token rate, request arrival rate, and input image rate. Our analysis reveals two key characteristics in production LMM inference:

<span id="page-5-2"></span>![](_page_5_Figure_11.jpeg)

Figure 10: Aggregated prompt/out token rate, request arrival rates in queries per minute (QPM), and image rate for a production LMM inference cluster in one week.

<span id="page-5-3"></span>![](_page_5_Figure_13.jpeg)

(a) Total prompt length. (b) #Images per request. Figure 11: LMM input characterization in production.

- Diverse Arrival Patterns. Image-text requests show up to 5× higher prompt token rates than text-only requests. In addition, their peak and trough occurrences are largely independent, showing minimal correlation.
- *Image-Driven Bursts*. Image-text requests experience significant burstiness, not only due to higher request arrival rates but also increased images per request (*e.g.*, video workloads). As a result, existing LLM traffic prediction methods [56] (which work well for LLM workloads with diurnal patterns) have a high average prediction error rate of 79%.

**Request Heterogeneity.** Figure 11a shows that prompt lengths vary significantly across modalities. Both image-text and text-only requests follow a heavy-tailed power-law distribution ( $\alpha$ =4.4 and 2.9, respectively) where a higher  $\alpha$  means a heavier tail with more extreme events occurring more frequently. In addition, image-text requests have longer median prompts due to image tokens but shorter tails than text-only requests. Figure 11b shows that the number of images per request also varies significantly with a heavy tail. In addition, among the top three services issuing text-image requests, we observe high inter-service variability. Some services (e.g., video) process 16× more images per request than others.

6

<span id="page-6-0"></span>![](_page_6_Figure_1.jpeg)

Figure 12: Llama3.2-11B (CroAttn) TTFT breakdown (left) and LLM prefill time breakdown (right) under various image-to-text token ratios in a request.

Comparing the image dimension distribution in our production traces with that of ShareGPT-40 image dataset [8], we observe similar distributions, with median image width and height around 500 pixels and P95 exceeding 1000 pixels.

<span id="page-6-3"></span><u>Insight 6:</u> Production LMM image traffic exhibits bursty behavior independent of the traffic patterns of text requests due to the nature of different services. Serving systems must dynamically scale resources to handle modality-specific bursts efficiently.

Impact of Mixed Modality. Given LMM requests' input heterogeneity, Figure 12 shows how varying image token percentages within a single request affects TTFT and LLM prefill time in a CroAttn model Llama3.2-11B, with detailed latency breakdowns. DecOnly models have no prefill time variation with varying token ratios as image and text tokens are treated in the same manner. We fix the total context length of each request at 16K tokens while varying the percentage of image tokens by adjusting the number of images (0–10 images in each case with 1601 tokens per image).

TTFT increases with the percentage of image tokens in a request due to the increased image encoding computation, resulting in a 1.5× TTFT degradation when transitioning from text-only to image-only inputs. However, this latency gain is significantly lower than DecOnly models because CroAttn models attend to image tokens only within the CroAttn layers, resulting in reduced LLM prefill time (shown in green) and partially offsetting the overhead from image encoding. The right figure in Figure 12 further illustrates this by breaking down the layer-wise LLM prefill time, highlighting a reduction in self-attention compute (*i.e.*, "Attn Layer" and "MLP Layer") as the proportion of image tokens increases. Although the cross-attention computation peaks at the 50% image tokens (due to the dependency on both image and text tokens), it contributes much less than self-attention computation because there are only 4 CroAttn layers (out of 40 layers).

<span id="page-6-4"></span><u>Insight 7:</u> DecOnly models maintain consistent prefill times regardless of token modality, making total token count the key factor for request routing. In contrast, CroAttn models experience reduced prefill latency as the image token percentage increases, requiring a modality-aware request routing strategy that balances both text and image token load in multimodal traffic.

<span id="page-6-1"></span>![](_page_6_Figure_8.jpeg)

Figure 13: Overview of the ModServe architecture.

## 4 ModServe Design and Implementation

Based on our insights from the characterization study of opensource LMM benchmarks and production LMM workloads, we propose ModServe, a novel decoupled architecture for scalable and resource-efficient LMM serving.

The key idea in ModServe is to separate image- and text-specific inference stages into distinct instances, given the need to optimize each stage separately (Insight 1 and 3) and enable seamless interaction between stages. Unlike monolithic infrastructures, ModServe enables independent optimization of each stage, improving resource efficiency while meeting performance SLOs. This decoupling also enables modality-aware request serving, addressing tail latency, heterogeneous bursts, and resource contention.

**Overview.** Figure 13 shows ModServe's design. A pool of *Image Instances* handles image preprocessing and encoding of image-text requests. The resulting image tokens are passed to a pool of *Text Instances*, which performs LLM prefill and decode operations to generate outputs. Text-only requests bypass the image components and are queued directly at the *Text Instances*. Two pools are managed by the *Image and Text Pool Managers*.

Modserve adopts a hierarchical architecture inspired by DynamoLLM [56]. Onboarding any new LMMs (e.g., Llama3.2-11B) starts with an offline profiling phase to build model-stage profiles that capture how model configurations and load impact performance (Section 4.1). Modserve uses these profiles to guide model configuration and instance scaling. After model deployment, Modserve then reconfigures resources periodically to adapt to workload patterns, scaling for increased image-text requests (or vice versa) (Section 4.2). For each request, Modserve selects the optimal LMM Text or Image Instances for execution (Section 4.3).

#### <span id="page-6-2"></span>4.1 Offline LMM Profile Generation

When onboarding a new LMM, ModServe generates resource-performance profiles by characterizing the image encoder and LLM backend independently. This profiling runs controlled inference workloads with varying model parallelisms (e.g., TP-2 and TP-8), batch sizes, and load (i.e., image tokens per second for image encoders and prompt tokens per second for LLM backends) to capture per-stage performance characteristics. To efficiently model performance across different load conditions, ModServe profiles a set of

representative load levels (up to the maximum throughput) and extrapolates the behavior for intermediate loads. The resulting profiles take load, parallelism, and batch size as inputs to predict key performance metrics, including encoding latency for *Image Instances* and prefill time and TBT for *Text Instances*. The *Pool Managers* use these profiles to guide model-specific or architecture-specific operational decisions (Section 4.2): (1) pool autoscaling to meet latency SLOs without overprovisioning, (2) model sharding that selects the optimal TP degree, and (3) maximum batch sizing for each stage.

Since multiple LMMs may share the same image encoder or LLM backend, ModServe minimizes overhead by reusing model profiles across deployments. These profiles are cached in cluster-local storage and synchronized via a global repository, enabling efficient sharing across clusters.

## <span id="page-7-0"></span>4.2 Decoupled Resource Management

ModServe's decoupled approach to resource management stems from our insights on stage-specific performance disparities in batching (Insight 4), independent scaling benefits (Insight 5), and modality-specific traffic patterns (Insight 6). Specifically, ModServe periodically reconfigures resources (*i.e.*, every five minutes to match the autoscaling overhead) to align with workload demands.

The *Image Pool Manager* maintains a pool of *Image Instances*, which preprocess images on CPU and encode image workloads on GPU for image-text requests. Meanwhile, the *Text Pool Manager* manages a pool of *Text Instances* responsible for the prefill and decode stages of both image-text and text-only requests. Based on model profiles (Section 4.1), each manager independently optimizes pool autoscaling, model sharding, and max batch sizing to minimize costs while meeting performance SLOs. Since connectors are extremely small in size and contribute negligible latency (< 0.4% of TTFT), dedicating separate GPUs to them would underutilize resources. Instead, ModServe colocates connectors with the LLM backend, which accommodates cases where the connector consumes features from multiple encoders [66].

Before their online operation, the initial number of Image Instances  $(N_i)$  is determined using the median image QPS multiplied by the median image encoding latency. The number of Text Instances  $(N_t)$  is set as  $N_i$  divided by the median number of images per request, based on historical LMM inference traces. If no history is available, Modserve initially overprovisions resources to both Text/Image Instances to ensure reliability.

**Token-Aware Pool Autoscaling.** The *Pool Managers* dynamically scale the number of *Image and Text Instances* based on real-time workload demands. For example, a surge in image-heavy requests leads to more image preprocessors and encoders, while an increase in text requests or requests' prompt lengths triggers the *Text Pool Manager* to scale LLM replicas to handle variations in prefill.

The number of replicas per stage is computed as  $\lceil \frac{ML}{MC} \rceil$  where ML is the modality-specific load (e.g., prompt tokens/sec for Text Instances, image tokens/sec for Image Instances) and MC is the maximum capacity each stage can handle without violating SLOs, based on the offline LMM profiling data. Unlike traditional web service autoscaling, which reacts to request rates, ModServe optimizes scaling based on token throughput (tokens/s), capturing variations in both request rates and request sizes (Insight 6 and Figure 11a).

For *Image Instances*, image token counts are precomputed based on the static mapping from image dimensions to the number of tokens per image (Figure 3). Autoscaling of *Text Instances* is based on text token load in CroAttn models but total tokens in DecOnly models due to homogeneous self-attention across modalities. Advanced autoscaling hysteresis prevention techniques [67] can be employed to avoid excessive scaling actions caused by transient workload fluctuations but are not covered in this paper.

Model Sharding. The *Pool Managers* also determine instance sharding for optimal tensor parallelism (TP) for image encoders and LLM backends. Our characterization (Section 3.1) shows image encoders achieve peak throughput with lower TP than LLMs. Therefore, the model sharding degree for each instance is configured separately for maximum throughput while ensuring SLO attainment on TTFT and TBT. By decoupling the components, ModServe ensures independent sharding, optimizing parallelism without unnecessary synchronization overhead.

When scaling beyond a single GPU server, ModServe prioritizes autoscaling over pipeline parallelism (PP) [53] to maximize throughput while avoiding communication overhead, seamlessly transitioning to batch-level optimizations as needed.

**Identifying Max Batch Size.** For each stage, the maximum batch size is configured to maximize throughput while meeting latency SLOs. Batch sizing decisions are guided by the offline model-stage profiles, which predict their impact on encoding and decoding latencies. *Image Instances* may forgo batching as small max batch sizes often achieve optimal GPU utilization (Insight 3). In contrast, *Text Instances* batch requests when beneficial, optimizing token throughput during prefill/decode based on TTFT and TBT SLOs, particularly for CroAttn LMMs (Insight 4).

## <span id="page-7-1"></span>4.3 Modality-Aware LMM Request Serving

For each incoming LMM request, ModServe dynamically routes and schedules workloads to balance load across *Image and Text Instances*. The *Pool Managers* optimize this process to minimize queueing delays and improve TTFT latency.

Request Routing Across Instances. To mitigate tail TTFT latency surges caused by modality-specific bursts and queuing delays, Modern Serve employs a modality-aware routing strategy that balances image and text workloads independently. Traditional request-level LLM load balancing (e.g., round-robin, memory-based [57]) overlooks the computational intensity of image encoding (Insight 2), making them vulnerable to load imbalances during image bursts (Insight 6), leading to high tail latencies.

Instead, ModServe routes requests by input modality. Image-text requests are assigned to *Image Instances* with the least image-token load. Large requests (*i.e.*, those with more images) are consequently distributed across multiple *Image Instances* for parallel processing and encoding (Insight 2), preventing degraded batching performance that would occur if all images were routed to a single instance. This effectively enables a form of request chunking [1], where images in a large request can be processed in an interleaved manner with other requests, reducing HoL blocking and improving scheduling flexibility.

To route traffic between Text Instances, text-only requests and image-text requests with completed image tokens are directed to the *Text Instance* with the least total pending tokens (text+image) for DecOnly models and the least total pending text tokens for CroAttn models because of the attention mechanism difference between the two model architectures (Insight 7). Modality-aware routing enables parallel image encoding and dynamically adapts to image or text traffic bursts, reducing queueing delays and improving TTFT, particularly at the tail.

Instance Request Scheduling. At the instance level, Modserve minimizes resource contention between image-text and text-only requests with priority scheduling based on modality and prompt size. While decoupling isolates image and text processing, contention can still arise in *Text Instances*, where both request types share prefill processing. This issue is particularly pronounced in DecOnly LMMs, which exhibits lower efficiency during the prefill stage (Figure 8). Performance degradation occurs from increased batching latency for all requests, while non-batched processing introduces HoL blocking and high queueing delays at tails due to request heterogeneity (Figure 11).

To address these challenges, Modserve replaces traditional FIFO scheduling—which may exacerbate HoL blocking [47, 50]—with an SLO-driven scheduling strategy that can prioritize shorter requests (e.g., text-only queries or small image-text requests with tight SLOs) to maintain low latency.

Pool Managers continuously monitor SLO attainment and trigger pool autoscaling when the rate falls below a predefined threshold (default 0.99 with a sensitivity study in Section 5.3), ensuring adaptive resource allocation under dynamic workloads (especially in cases of unpredictable traffic). ModServe can work with state-of-the-art batch scheduling techniques [1, 46] to optimize TBT during the decode stage, which we leave to future work as we do not observe TBT degradation in LMM characterization (Insight 4).

Image Token Transfer. Once image encoding is complete, Mod-Serve transfers image tokens from *Image Instances* to *Text Instances* via a pull-based RDMA mechanism. Push-based approaches immediately transmit image tokens as they are generated, requiring premature decisions about which *Text Instance* should receive them (and all subsequent tokens). This design increases synchronization overhead and risks suboptimal routing, as the system operates with incomplete information about token counts and runtime load.

In contrast, our pull-based design defers transfer until all image tokens for a request are ready. This many-to-one aggregation enables the scheduler to select the target *Text Instance* with full information, considering factors such as queue size, prefix caching, and payloads. Once the routing decision is made, the request carries RDMA addresses of the producing *Image Instances*, from which the chosen *Text Instance* pulls the image tokens.

ModServe colocates *Image and Text Instances* in the same server when each *Text Instance* is not taking up all GPUs on a server to avoid image transfer overhead and unallocated idle GPUs. For example, it may place one TP-4 Text Instance and two TP-2 Image Instances within the same 8-GPU server. Unlike monolithic deployments, colocated instances remain independently configurable and can serve corresponding stages of different requests independently.

## 4.4 Implementation

We implement ModServe using 5,000 lines of Python code. We base the *Text Instance* on vLLM [27] (v0.7.2), a state-of-the-art generative model inference platform, and build the *Image Instance* on HuggingFace Transformers [24]. The modular architecture of ModServe enables easy integration with other serving engines (e.g., TensorRT [43] and DeepSpeed [5]). We use numact1 to restrict CPU and memory usage of image preprocessing to a single NUMA node, which reduces memory access latency and performance variation. To ensure efficient GPU-to-GPU memory transfer of image tokens, we use PyTorch's distributed communication with the NCCL backend and GPU Direct RDMA.

We implement the *Image and Text Pool Managers* as lightweight gRPC servers (hosted on dedicated VMs) with low memory and compute requirements, drawing inspiration from DynamoLLM [56]. For failure detection and recovery, the *Pool Managers* use a simple heartbeat-based membership management protocol [17]. However, MODSERVE can be easily extended to adopt more robust leader election (*e.g.*, Raft [44]) and fault-tolerance algorithms [60].

#### <span id="page-8-0"></span>5 Evaluation

## 5.1 Experimental Setup

Models and Workloads. We use two representative multimodal models, Llama3.2-11B and InternVL-26B, for CroAttn and DecOnly LMMs, respectively. To ensure realistic workload distribution, we adopt the inter-arrival timestamps of requests and the number of images associated with each request (ranges from 0 to 16) from the production LMM inference trace (Section 3.2) and reuse the LMM text-image dataset from Section 3.1.

Hardware. We evaluate ModServe on a cluster with 16 DGX-A100 servers [39] (128 GPUs). Each server has the same configuration as the server used in our characterization study (Section 3.1). The GPUs within a server are connected with NVLINK 3.0 while cross-server connection is via InfiniBand.

Baselines and Systems. We compare ModServe against the state-of-the-art generative model inference serving system, vLLM [27], which supports LMM inference as a monolithic setup. We also evaluate ModServe with a few variants of ModServe implemented on top of vLLM: (1) vLLM with decoupled Image/Text Instances (i.e., ModServe-Decoup), (2) ModServe-Decoup plus modality-aware scheduling (i.e., ModServe-Sched), and (3) ModServe-Sched plus modality-aware routing (i.e., ModServe), for ablation study.

**SLO Definition.** We define the SLO metrics for LMM inference based on the TTFT/TBT during the isolated run of a single text-only request and text-image request (with one image) on the monolith baseline setup. Then, we scale the SLO metrics with a constant factor (*i.e.*, SLO factor) to evaluate how Modserve performs under tight/relaxed SLOs (Section 5.3). The SLOs are defined on P99 tail latency across requests over time.

# <span id="page-8-1"></span>5.2 End-to-end Performance

**Static Resource Allocation.** We begin by evaluating ModServe under a static resource allocation setup, where a fixed number of servers remain active at all times without autoscaling. This setup

<span id="page-9-1"></span>![](_page_9_Figure_1.jpeg)

Figure 14: TTFT comparison with fixed 16 servers (128 GPUs) without autoscaling.

<span id="page-9-2"></span>![](_page_9_Figure_3.jpeg)

Figure 15: Maximum load meeting SLO

isolates the benefits of decoupling, modality-aware request scheduling, and routing from pool autoscaling (which we explore independently). Figure 14 shows the average and tail (P99) TTFT achieved by Modserve and the baselines when serving different input loads over fixed resources (16 servers with 128 GPUs in total). In this setup, vLLM (monolith) deploys 32 instances (each with TP-4) while the other approaches (decoupled) deploy 20 Text Instances (TP-4) and 48 Image Instances (TP-1).

Compared to vLLM, statically decoupling (ModServe-Decoup) improves the average and P99 TTFT by 27% and 42% (for Llama3.2), 46% and 47% (for InternVL). This is because monolithic deployments process all modalities on shared GPU resources, leading to contention and inefficient utilization under imbalanced modality traffic. In addition, ModServe-Decoup with the same number of GPUs can deploy 16 extra Image Instances and enables image encoding parallelization that reduces TTFT significantly compared to the monolithic deployment on vLLM.

ModServe shows a more pronounced TTFT improvement over the monolith baseline when serving InternVL. This is because the monolith deployment faces resource contention with DecOnly models due to their high prefill latency (Insight 3), which contends with image encoding. Additionally, InternVL's image encoder has higher batching performance degradation (Insight 4) and thus benefits more from parallelization. Adding modality-aware request scheduling (ModServe-Sched) further reduces the average and P99 TTFT by 12% and 25%, modality-aware routing (ModServe) reduces the average and P99 TTFT by 14% and 32%, as it reduces HoL blocking and mitigates tail latency spikes.

Overall, Modserve achieves the lowest TTFT across all load levels, demonstrating the effectiveness of modular inference pipelines. We observe similar TBT performance in all approaches due to its compute insensitivity (as indicated by Figure 8). Figure 15 further evaluates the maximum throughput under the TTFT and TBT SLO when varying the static resource allocation from 4 to 16 servers. Modserve achieves a 3.3× and 5.5× throughput improvement over

<span id="page-9-3"></span>![](_page_9_Figure_9.jpeg)

Figure 16: GPU allocation with autoscaling (up to 16 servers) during a one-day interval on the production traces.

vLLM (monolith) for Llama3.2 and InternVL, respectively, which confirms that DecOnly models benefit more from decoupling.

Resource Allocation with Autoscaling. We now assess how Modserve and vLLM (monolith) baseline handle image-driven bursts seen in the production trace (Figure 10). Fundamentally, to serve traffic bursts, a system needs to scale up the resources to meet the workload demand while scaling down to avoid overprovisioning. Therefore, we enable autoscaling in both Modserve and vLLM and evaluate them on a one-day interval of the production trace that contains an image-driven burst. For a fair comparison, both Modserve and vLLM (monolith) use similar SLO-driven autoscaling heuristics based on offline model profiling (Section 4.2).

Figure 16 compares the number of GPUs used by ModServe and vLLM (monolith) to serve the image-driven burst in the production trace. ModServe takes 41.3% and 25% fewer GPUs compared to vLLM to serve Llama-3.2 (CroAttn) and InternVL (DecOnly) models respectively while meeting the tail latency SLOs. ModServe's cost reduction is higher for Llama-3.2 (CroAttn) model because the increase in image tokens caused by image-driven traffic bursts does not overwhelm the LLM backends in CroAttn models as observed in its latency profile (Figure 12). However, in InternVL (DecOnly), the LLM backend's latency increases with the increase in image tokens due to homogeneous self-attention. Therefore, to meet SLOs, ModServe scales up the number of Text Instances for InternVL more than for Llama-3.2 during image-driven bursts (light pink in Figure 16). Overall, ModServe's stage-aware autoscaling prevents unnecessarily scaling up LLM backends (done by vLLM due to monolith deployment) during image-driven bursts and prevents resource over-provisioning.

# <span id="page-9-0"></span>5.3 Sensitivity Study

**Impact of SLO Scale.** Figure 17 shows the maximum throughput ModServe can achieve when changing the SLO scale (higher values refer to more relaxed SLOs). As the SLO scale increases, ModServe consistently outperforms the vLLM, achieving up to 4.3× higher throughput for Llama-3.2 and 6.8× for InternVL. This

<span id="page-10-1"></span>![](_page_10_Figure_1.jpeg)

Figure 17: Throughput impact varying the SLO scale.

<span id="page-10-2"></span>![](_page_10_Figure_3.jpeg)

#### (a) Llama 3.2-11B CroAttn LMM.

![](_page_10_Figure_5.jpeg)

(b) InternVL-26B DecOnly LMM.

Figure 18: Impact of image request percentage (Y-axis) and instance allocation (X-axis), i.e., #Text Instances (TP4): #Image Instances (TP1) on 8 servers (64 GPUs).

trend highlights that ModServe better utilizes resources under the same latency requirements.

Impact of Image-to-Text Instance Ratio. Figure 18 shows the effect of varying the ratio of Image and Text Instances on 64 GPUs (8 servers) along the *X*-axis, in comparison to vLLM monolith with 16 instances. For instance, "4:48" denotes a configuration with 4 Text Instances (TP-4) and 48 Image Instances (TP-1). As the ratio of Text Instances increases, we observe that MODSERVE consistently achieves superior TTFT performance compared to vLLM (monolith) until the ratio reaches 10:24. However, at 12:16, the decoupled configuration contains the same number of image encoders but 4 fewer LLM backends, resulting in inferior performance. Moreover, reducing image encoders below the monolith baseline contradicts the core goal of decoupling to scale up/out the image encoders independently for multimodal processing.

Impact of Image:Text Request Ratio. Figure 18 also shows the impact of varying image-text request percentages in the workload (*Y*-axis). As this percentage increases from 10% to 90% (more image-heavy), TTFT for Llama-3.2 (CroAttn) increases. InternVL (DecOnly) follows a similar trend, except at lower *Text Instance* ratios (*e.g.*, 4:48), where P99 TTFT decreases from 3.8 to 3.3 seconds due to reduced text load. This stems from DecOnly models' poor prefill efficiency. For the same reason, at low image-text request

<span id="page-10-3"></span>![](_page_10_Figure_11.jpeg)

Figure 19: TTFT improvement with ModServe from a prefill-decode disaggregated setup for InternVL-26B (DecOnly).

percentages (*e.g.*, 10%), InternVL sees a lower P99 TTFT as more *Text Instances* help distribute the text-heavy load.

On the other hand, across all image-text request percentages, increasing the number of *Text Instances* raises P99 TTFT in Llama3.2 due to a reduced number of *Image Instances*, leading to longer image encoding times. However, regardless of distribution, ModServe outperforms the monolith baseline (by up to 18.4× for Llama3.2 and 9.2× for InternVL) when Image:Text Instance ratio exceeds 2.4, demonstrating its efficiency handling multimodal workloads.

Model Architectures. Our evaluation on open-source LMMs includes models up to 90B parameters, while production deployments may involve even larger model sizes affecting image encoding ratios in TTFT, which we defer to future work. We focus on visual LMMs but audio-based multimodal models [23] share similar architectures and parameter scales with vision multimodal models. We also note that hybrid multimodal architectures have been proposed [12], though no open-source hybrid models are currently available.

#### <span id="page-10-0"></span>5.4 Prefill-Decode Disaggregation Support

Modserve is complementary to existing techniques for LLM backend optimization, including prefill/decode (PD) disaggregation [46, 65], which splits LLM inference into two execution phases: prefill and decode (token-by-token generation). Our design fully supports PD disaggregation, which leads to a full EPD disaggregation.

To demonstrate this, we compare two deployment configurations under varying load, both incorporate PD disaggregation, deploy the InternVL-26B model, and use the same number of decode instances to match TBT latency (orthogonal to Modserve's contributions). The main difference between the two configurations comes in the LLM prefill instances: (1) PD-Monolith: 4 prefill instances are deployed, where each instance is distributed across 8 GPUs. Each prefill instance also hosts an image encoder for image preprocessing and encoding. (2) PD-Modserve: 3 prefill instances are deployed, each across 8 GPUs. Image encoders are fully decoupled from the LLM backends and run as 8 independent processes on the remaining GPU server. Both configurations use a total of 32 GPUs for image encoding and LLM prefill combined.

This setup allows us to isolate the benefits of stage-level decoupling in ModServe from PD disaggregation. Figure 19 demonstrates that ModServe (blue) provides additional TTFT reduction (up to 2.8× in average TTFT and 3.2× in P90 TTFT for InternVL-26B) beyond what PD disaggregation alone can offer (red). The TTFT improvement (for both mean and P90) becomes more pronounced when load increases as ModServe reduces resource contention

<span id="page-11-6"></span>![](_page_11_Figure_1.jpeg)

Figure 20: Image token transfer latency across token sizes

between the image encoding and LLM prefill stages and leverages encoder parallelization to reduce encoding latency (Insight 2).

#### 5.5 Token Transfer Overhead

Figure 20 shows the image token transfer overhead for varying-sized image embeddings, comparing different communication media of using Infiniband and Ethernet. With RDMA on Infiniband, we observed the P99 transfer latency of image tokens per image request is 5 ms, which corresponds to <0.5% and <0.3% TTFT for CroAttn and DecOnly models, respectively. TCP over Ethernet incurs significantly higher overheads, with a P50 of 100 ms and a P99 of 180 ms. ModServe supports both communication media. When evaluated over TCP, ModServe achieves a 35% TTFT reduction at high load for InternVL-26B and an 8.4% reduction at low load compared to the monolithic baselines (with Infiniband, the reduction is 46% and 13%, respectively, as mentioned in Section 5.2).

## 6 Related Work

**LMM Characterization.** Lee *et al.* [28] provides a comprehensive characterization of multimodal *generation* models at Meta, while we focus on LMMs with multimodal input (*e.g.*, visual understanding models). Hou *et al.* [19] focus on traditional multimodal models employing small-scale convolutional neural networks. In contrast, our work presents a detailed analysis of multimodal input workloads on both open-source LMM models and production traces, highlighting their unique execution and workload patterns.

LMM Serving Optimization. Recent research has introduced several techniques to optimize LMM serving by addressing key inefficiencies in inference computation and memory usage. Inf-MLLM [41] employs token caching strategies and attention bias to maintain performance with long contexts while reducing KV cache memory consumption. Elastic Cache [36] utilizes an importancedriven cache merging strategy to prune KV caches efficiently during inference. Dynamic-LLaVA [22], VTW [35], and QueCC [32] present various vision token sparsification and compression techniques to dynamically reduce redundancy in vision tokens, especially for video workloads. These optimizations primarily operate at the model level, trading off computational overhead with output quality (i.e., accuracy). They are orthogonal to our proposed system-level design for inference efficiency that does not impact model accuracy, which can further benefit from such model-level advancements, e.g., faster image encoding with subsampling [29].

To optimize LMM inference, concurrent works adopt a similar stage decoupling idea (e.g., EPD [54] and HydraInfer [14]) and parallel encoding (e.g., IRP [54]). In contrast, our work extends beyond

stage decoupling by incorporating stage-aware model configuration, modality-aware routing, and autoscaling, rooted in insights from a comprehensive systems analysis of production LMM inference workloads. In addition, our characterization and evaluation take a closer look at two representative LMM architectures, rather than being limited to decoder-only models.

**Text-Centric LLM Serving.** Recent studies have delved into disaggregating LLM prefill and decode phases for text-only LLM serving. Examples include Splitwise [46], DistServe [65], Mooncake [48], and MemServe [20]. Other optimizations for LLM serving include key-value cache management [27], continuous batching [63], request scheduling [1, 2, 47, 50, 57], and energy optimization [49, 55, 56]. While these optimizations can be applied in ModServe to enhance LLM backend prefill and decode efficiency, our work focuses on the unique characteristics of multimodal models.

#### 7 Conclusion

We present the first comprehensive systems analysis of LMMs on both open-source models and production LMM inference traces. Our insights lead to the design of ModServe, a scalable and resource-efficient LMM-serving framework that decouples inference stages for dynamic reconfiguration, adaptive scaling, and modality-aware scheduling. Evaluations show that ModServe achieves 25–41% cost savings compared to the state-of-the-art while efficiently serving production-scale LMM inference workloads.

# Acknowledgments

This project was partially supported in part by U.S. NSF grants NSF-2421782, NSF-2350425, NSF-2319988, NSF-2206522, Microsoft Research Faculty Fellowship 8300751, and the Commonwealth Cyber Initiative (CCI), an investment in the advancement of cyber research, innovation, and workforce development. For more information about CCI, visit cyberinitiative.org.

## References

- <span id="page-11-5"></span> Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI).
- <span id="page-11-7"></span>[2] Amey Agrawal, Haoran Qiu, Junda Chen, İñigo Goiri, Chaojie Zhang, Rayyan Shahid, Ramachandran Ramjee, Alexey Tumanov, and Esha Choukse. 2024. Medha: Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations. arXiv preprint arXiv:2409.17264 (2024).
- <span id="page-11-3"></span>[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a Visual Language Model for Few-Shot Learning. 2024 Conference on Neural Information Processing Systems (NeurIPS 2024) 35 (2022), 23716–23736.
- <span id="page-11-4"></span>[4] Dosovitskiy Alexey. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv: 2010.11929 (2020).
- <span id="page-11-1"></span>[5] Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, et al. 2022. DeepSpeed-Inference: Enabling efficient inference of transformer models at unprecedented scale. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC).
- <span id="page-11-0"></span>[6] Jun Chen, Han Guo, Kai Yi, Boyang Li, and Mohamed Elhoseiny. 2022. VisualGPT: Data-efficient adaptation of pretrained language models for image captioning. In Proceedings of the 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022). 18030–18040.
- <span id="page-11-2"></span>[7] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. 2025. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling.

- <span id="page-12-17"></span>[8] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. 2024. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. arXiv preprint arXiv:2404.16821 (2024).
- <span id="page-12-4"></span>[9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024). 24185–24198.
- <span id="page-12-23"></span>[10] Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and others. 2024. HuggingFace Model: OpenGVLab/InternVL2\_5-26B. https://huggingface.co/OpenGVLab/InternVL2\_5-26B.
- <span id="page-12-16"></span>[11] Jianfeng Chi, Ujjwal Karn, Hongyuan Zhan, Eric Smith, Javier Rando, Yiming Zhang, Kate Plawiak, Zacharie Delpierre Coudert, Kartikeya Upasani, and Mahesh Pasupuleti. 2024. Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations. arXiv preprint arXiv:2411.10414 (2024).
- <span id="page-12-15"></span>[12] Wenliang Dai, Nayeon Lee, Boxin Wang, Zhuolin Yang, Zihan Liu, Jon Barker, Tuomas Rintamaki, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. NVLM: Open Frontier-Class Multimodal LLMs. arXiv:2409.11402 [cs.CL] https://arxiv.org/abs/2409.11402
- <span id="page-12-24"></span>[13] Dai, Wenliang and Lee, Nayeon and Wang, Boxin and Yang, Zhuolin and Liu, Zihan and Barker, Jon and Rintamaki, Tuomas and Shoeybi, Mohammad and Catanzaro, Bryan and Ping, Wei. 2024. HuggingFace Model: nvidia/NVLM-D-72B. https://huggingface.co/nvidia/NVLM-D-72B.
- <span id="page-12-41"></span>[14] Xianzhe Dong, Tongxuan Liu, Yuting Zeng, Liangyu Liu, Yang Liu, Siyu Wu, Yu Wu, Hailong Yang, Ke Zhang, and Jing Li. 2025. HydraInfer: Hybrid Disaggregated Scheduling for Multimodal Large Language Model Serving. arXiv preprint arXiv:2505.12658 (2025).
- <span id="page-12-18"></span>[15] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, et al. 2024. VLMEvalKit: An open-source toolkit for evaluating large multi-modality models. https://huggingface.co/spaces/opencompass/open\_vlm\_leaderboard. In Proceedings of the 32nd ACM International Conference on Multimedia.
- <span id="page-12-12"></span>[16] GitHub. 2025. Azure Public Dataset: Azure LMM Inference Trace 2025. https://github.com/Azure/AzurePublicDataset/tree/master.
- <span id="page-12-31"></span>[17] Indranil Gupta, Tushar D Chandra, and Germán S Goldszmidt. 2001. On Scalable and Efficient Distributed Failure Detectors. In Proceedings of the Twentieth Annual ACM Symposium on Principles of Distributed Computing (PODC).
- <span id="page-12-6"></span>[18] Luxi He, Xiangyu Qi, Michel Liao, Inyoung Cheong, Prateek Mittal, Danqi Chen, and Peter Henderson. 2025. The Deployment of End-to-End Audio Language Models Should Take into Account the Principle of Least Privilege. arXiv preprint arXiv:2503.16833 (2025).
- <span id="page-12-34"></span>[19] Xiaofeng Hou, Cheng Xu, Jiacheng Liu, Xuehan Tang, Lingyu Sun, Chao Li, and Kwang-Ting Cheng. 2022. Characterizing and understanding end-to-end multi-modal neural networks on GPUs. *IEEE Computer Architecture Letters* 21, 2 (2022), 125–128.
- <span id="page-12-43"></span>[20] Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu, Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, et al. 2024. MemServe: Context caching for disaggregated LLM serving with elastic memory pool. arXiv preprint arXiv:2406.17565 (2024).
- <span id="page-12-0"></span>[21] Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo. 2023. PromptCap: Prompt-guided image captioning for VQA with GPT-3. In Proceedings of the 2023 IEEE/CVF International Conference on Computer Vision (ICCV 2023), 2963–2975.
- <span id="page-12-37"></span>[22] Wenxuan Huang, Zijie Zhai, Yunhang Shen, Shaoshen Cao, Fei Zhao, Xiangfeng Xu, Zheyu Ye, and Shaohui Lin. 2024. Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification. arXiv preprint arXiv:2412.00876 (2024).
- <span id="page-12-11"></span>[23] HuggingFace. 2024. Audio-Text-to-Text Models. https://huggingface.co/models? pipeline\_tag=audio-text-to-text.
- <span id="page-12-29"></span>[24] HuggingFace. 2024. HuggingFace Transformers. https://huggingface.co/docs/ transformers/en/index.
- <span id="page-12-7"></span>[25] HuggingFace. 2024. Image-Text-to-Text Models. https://huggingface.co/models? pipeline tag=image-text-to-text.
- <span id="page-12-26"></span>[26] Aditya K Kamath, Ramya Prabhu, Jayashree Mohan, Simon Peter, Ramachandran Ramjee, and Ashish Panwar. 2024. Pod-attention: Unlocking full prefill-decode overlap for faster LLM inference. arXiv preprint arXiv:2410.18038 (2024).
- <span id="page-12-8"></span>[27] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP).
- <span id="page-12-33"></span>[28] Yejin Lee, Anna Sun, Basil Hosmer, Bilge Acun, Can Balioglu, Changhan Wang, Charles David Hernandez, Christian Puhrsch, Daniel Haziza, Driss Guessous, et al. 2024. Characterizing and Efficiently Accelerating Multimodal Generation Model Inference. arXiv preprint arXiv:2410.00215 (2024).
- <span id="page-12-40"></span>[29] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L Berg, Mohit Bansal, and Jingjing Liu. 2021. Less is more: ClipBERT for video-and-language learning via

- sparse sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 7331–7341.
- <span id="page-12-5"></span>[30] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-OneVision: Easy Visual Task Transfer. arXiv preprint arXiv:2408.03326 (2024).
- <span id="page-12-10"></span>[31] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326 [cs.CV] https://arxiv.org/abs/2408.03326
- <span id="page-12-39"></span>[32] Kevin Y Li, Sachin Goyal, Joao D Semedo, and J Zico Kolter. 2024. Inference Optimal VLMs Need Only One Visual Token but Larger Models. arXiv preprint arXiv:2411.03312 (2024).
- <span id="page-12-22"></span>[33] Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Li, Yanwei and Liu, Ziwei and Li, Chunyuan. 2024. HuggingFace Model: lmms-lab/llava-onevision-qwen2-72b-ov-sft. https://huggingface.co/lmms-lab/llava-onevision-qwen2-72b-ov-sft.
- <span id="page-12-21"></span>[34] Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Li, Yanwei and Liu, Ziwei and Li, Chunyuan. 2024. HuggingFace Model: lmms-lab/llava-onevision-qwen2-7b-ov. https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov.
- <span id="page-12-38"></span>[35] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. 2024. Boosting Multimodal Large Language Models with Visual Tokens Withdrawal for Rapid Inference. arXiv preprint arXiv:2405.05803 (2024).
- <span id="page-12-36"></span>[36] Zuyan Liu, Benlin Liu, Jiahui Wang, Yuhao Dong, Guangyi Chen, Yongming Rao, Ranjay Krishna, and Jiwen Lu. 2025. Efficient inference of vision instructionfollowing models with elastic cache. In European Conference on Computer Vision (ECCV).
- <span id="page-12-19"></span>[37] Meta AI. 2024. HuggingFace Model: meta-llama/Llama-3.2-11B-Vision-Instruct. https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct.
- <span id="page-12-20"></span>[38] Meta AI. 2024. HuggingFace Model: meta-llama/Llama-3.2-90B-Vision-Instruct. https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct.
- <span id="page-12-25"></span>[39] Microsoft Azure. 2024. Azure VM NDm-A100-v4 sizes series. https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/ndma100v4-series.
- <span id="page-12-1"></span>[40] Ron Mokady, Amir Hertz, and Amit H Bermano. 2021. ClipCap: CLIP Prefix for Image Captioning. arXiv preprint arXiv:2111.09734 (2021).
- <span id="page-12-35"></span>[41] Zhenyu Ning, Jieru Zhao, Qihao Jin, Wenchao Ding, and Minyi Guo. 2024. Inf-MLLM: Efficient Streaming Inference of Multimodal Large Language Models on a Single GPU. arXiv preprint arXiv:2409.09086 (2024).
- <span id="page-12-13"></span>[42] Runliang Niu, Jindong Li, Shiqi Wang, Yali Fu, Xiyu Hu, Xueyuan Leng, He Kong, Yi Chang, and Qi Wang. 2024. ScreenAgent: A Vision Language Model-Driven Computer Control Agent. arXiv preprint arXiv:2402.07945 (2024).
- <span id="page-12-30"></span>[43] NVIDIA. 2024. NVIDIA TensoRT. https://github.com/NVIDIA/TensorRT.
- <span id="page-12-32"></span>[44] Diego Ongaro and John Ousterhout. 2014. In Search of An Understandable Consensus Algorithm. In Proceedings of the 2014 USENIX Conference on USENIX Annual Technical Conference (ATC).
- <span id="page-12-14"></span>[45] OpenAI. 2025. Computer-Using Agent: Introducing a universal interface for AI to interact with the digital world. https://openai.com/index/computer-using-agent.
- <span id="page-12-9"></span>[46] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Îñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient generative LLM inference using phase splitting. In Proceedings of the ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA).
- <span id="page-12-27"></span>[47] Archit Patke, Dhemath Reddy, Saurabh Jha, Haoran Qiu, Christian Pinto, Chandra Narayanaswami, Zbigniew Kalbarczyk, and Ravishankar Iyer. 2024. Queue Management for SLO-Oriented Large Language Model Serving. In Proceedings of the 15th ACM Symposium on Cloud Computing (SoCC).
- <span id="page-12-42"></span>[48] Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. 2024. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving. arXiv preprint arXiv:2407.00079 (2024).
- <span id="page-12-44"></span>[49] Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew Kalbarczyk, Tamer Başar, and Ravishankar K. Iyer. 2024. Power-aware Deep Learning Model Serving with μ-Serve. In USENIX Annual Technical Conference (USENIX ATC 2024).
- <span id="page-12-28"></span>[50] Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew T. Kalbarczyk, Tamer Başar, and Ravishankar K. Iyer. 2024. Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction. In The 5th International Workshop on Cloud Intelligence / AIOps at ASPLOS 2024.
- <span id="page-12-2"></span>[51] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. 2022. A-OKVQA: A benchmark for visual question answering using world knowledge. In Proceedings of the European Conference on Computer Vision (ECCV).
- <span id="page-12-3"></span>[52] Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu. 2023. Prompting large language models with answer heuristics for knowledge-based visual question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023).

- <span id="page-13-9"></span>[53] Mohammad Shazeer et al. 2020. Megatron-LM: Training Multi-Billion Parameter Language Models Using Pipeline Parallelism. arXiv preprint arXiv:1909.08053 (2020). https://arxiv.org/abs/1909.08053
- <span id="page-13-12"></span>[54] Gursimran Singh, Xinglu Wang, Ivan Hu, Timothy Yu, Linzi Xing, Wei Jiang, Zhefeng Wang, Xiaolong Bai, Yi Li, Ying Xiong, et al. 2024. Efficiently serving large multimedia models using EPD Disaggregation. arXiv preprint arXiv:2501.05460 (2024).
- <span id="page-13-14"></span>[55] Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Esha Choukse, Haoran Qiu, Rodrigo Fonseca, Josep Torrellas, and Ricardo Bianchini. 2025. TAPAS: Thermal-and Power-Aware Scheduling for LLM Inference in Cloud Platforms. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS). 1266–1281.
- <span id="page-13-7"></span>[56] Jovan Stojkovic, Chaojie Zhang, Íñigo Goiri, Josep Torrellas, and Esha Choukse. 2025. DynamoLLM: Designing LLM Inference Clusters for Performance and Energy Efficiency. In Proceedings of the IEEE International Symposium on High Performance Computer Architecture (HPCA).
- <span id="page-13-10"></span>[57] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. 2024. Llumnix: Dynamic Scheduling for Large Language Model Serving. arXiv preprint arXiv:2406.03243 (2024).
- <span id="page-13-0"></span>[58] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
- <span id="page-13-6"></span>[59] vLLM. 2024. Distributed Inference and Serving. https://docs.vllm.ai/en/latest/ serving/distributed serving.html.
- <span id="page-13-11"></span>[60] Cheng Wang, Xusheng Chen, Weiwei Jia, Boxuan Li, Haoran Qiu, Shixiong Zhao, and Heming Cui. 2018. PLOVER: Fast, Multi-core Scalable Virtual Machine Fault-tolerance. In Proceedings of the 15th USENIX Symposium on Networked Systems Design and Implementation (NSDI).

- <span id="page-13-1"></span>[61] A Waswani, N Shazeer, N Parmar, J Uszkoreit, L Jones, A Gomez, L Kaiser, and I Polosukhin. 2017. Attention is all you need. In 2017 Conference on Neural Information Processing Systems (NIPS 2017).
- <span id="page-13-2"></span>[62] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Association for Computational Linguistics, Online, 38–45. https://www.aclweb.org/anthology/2020.emnlp-demos.6
- <span id="page-13-13"></span>[63] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In Proceedings of the 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI).
- <span id="page-13-4"></span>[64] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid Loss for Language Image Pre-Training. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV).
- <span id="page-13-3"></span>[65] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI).
- <span id="page-13-5"></span>[66] Xun Zhu, Zheng Zhang, Xi Chen, Yiming Shi, Miao Li, and Ji Wu. 2025. Connector-S: A Survey of Connectors in Multi-modal Large Language Models. arXiv preprint arXiv:2502.11453 (2025).
- <span id="page-13-8"></span>[67] Ding Zou, Wei Lu, Zhibo Zhu, Xingyu Lu, Jun Zhou, Xiaojin Wang, Kangyu Liu, Kefan Wang, Renen Sun, and Haiqing Wang. 2024. OptScaler: A Collaborative Framework for Robust Autoscaling in the Cloud. Proceedings of the VLDB Endowment 17, 12 (Aug. 2024), 4090–4103. doi:10.14778/3685800.3685829