## **ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models** 

Chengzhang Wu[∗] Liyan Zheng[∗] Tsinghua University Tsinghua University Beijing, China Beijing, China wcz24@mails.tsinghua.edu.cn zhengly20@mails.tsinghua.edu.cn 

Kezhao Huang Tsinghua University Beijing, China hkz20@tsinghua.org.cn 

Zixuan Ma Tsinghua University Beijing, China mzx22@mails.tsinghua.edu.cn 

Haojie Wang Tsinghua University Beijing, China wanghaojie@tsinghua.edu.cn 

Dong Dong Tsinghua University Beijing, China dongd@tsinghua.edu.cn 

Jidong Zhai Tsinghua University Beijing, China zhaijidong@tsinghua.edu.cn 

## **Abstract** 

Diffusion models have become the dominant approach for generative tasks in images, videos, and other domains. However, diverse data properties in generation requests, which are critical for efficient serving, remain underexploited. 

To address this issue, we propose a diffusion model serving system ChituDiffusion. ChituDiffusion leverages the locality of data properties to recompose a diffusion pipeline into subgraphs with shared optimization opportunities, enabling thorough compile-time and runtime co-optimizations. During compilation, ChituDiffusion compiles each subgraph into multiple execution engines optimized for specific data properties. At runtime, heterogeneous requests are elaborately reorganized into fine-grained batching tasks with similar properties and then efficiently executed by matched engines. Evaluation on five diffusion applications shows that ChituDiffusion improves the throughput by up to 2 _._ 13× (1 _._ 58× on average) on A100 and 2 _._ 19× (1 _._ 51× on average) on H100 compared with existing frameworks. The code for ChituDiffusion and the production traces have been made open-source at https://github.com/thu-pacman/chitu/tree/ Diffusion. 

_**CCS Concepts:**_ • **Computer systems organization** → **Real-time systems** ; **Parallel architectures** ; • **Computing methodologies** → **Machine learning** ; _Computer vision_ . 

∗Both authors contributed equally to this research. 

This work is licensed under a Creative Commons Attribution 4.0 International License. _PPoPP ’26, Sydney, NSW, Australia_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 https://doi.org/10.1145/3774934.3786424 

_**Keywords:**_ Deep learining serving system, Diffusion models, Data-characteristic-aware optimization, Compiler 

## **ACM Reference Format:** 

Chengzhang Wu, Liyan Zheng, Haojie Wang, Kezhao Huang, Zixuan Ma, Dong Dong, and Jidong Zhai. 2026. ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models. In _Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP ’26), January 31 – February 4, 2026, Sydney, NSW, Australia._ ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3774934.3786424 

## **1 Introduction** 

Diffusion models have become a versatile class of generative algorithms across domains, including images [29, 51, 52, 67], videos [11, 16, 28, 30], 3D objects [32, 49], music [27, 43], and proteins [63]. Their flexibility supports applications from text-to-image [51] and text-to-video [11] generation to controllable editing [67], powering products like DALL·E 3 [20], Sora [6], and Firefly [2]. To meet application demands, diffusion pipelines integrate diverse input data and multiple DNNs, leading to high computational cost. Real-world requests further exhibit heterogeneous data properties—such as shared and partially duplicate inputs with varying generation shapes (Figure 1(a))—arising from both applications and user behaviors [62, 64]. Exploiting these properties offers acceleration opportunities beyond the scope of general ML frameworks [7, 48, 58]. 

Dedicated diffusion model frameworks, e.g., Diffusers [61], offer manually optimized pipelines designed for specific data properties, such as batched requests with identical prompts shown in Figure 1(b). Users can select the corresponding optimized pipeline for efficient executions. However, these manual optimizations address only a narrow subset of potential scenarios due to two challenges. 

537 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

**==> picture [241 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prompt<br>R1) (“A dog is walking”,                   ) Image<br>Prompt<br>R2) (“A dog is walking”,                   )<br>Image<br>Prompt<br>R3) (“A cat is walking”,                    )<br>Image<br>Requests Computation Generated images<br>(a)<br>Original pipeline Optimized pipelines<br>PromptPromptPrompt<br>Manual<br>Other inputsOther inputsOther inputsImageImageImage optimization …<br>Request data properties<br>Uniform properties Heterogeneous properties<br>promptSame shapeSame Raggedshape … R1 & R2 with the same promptR2 & R3 with the same shape …<br>(b)<br>**----- End of picture text -----**<br>


**Figure 1.** (a) Image generation requests with different data properties. Yellow and blue circles represent identical computations and computations with the same generation shape, respectively. (b) Existing diffusion frameworks with manual optimizations for uniform properties but lack support for heterogeneous properties. 

_1) Diverse input data properties_ . The complexity of diffusion models means each generation request may involve tens of inputs with independent properties. Since these properties can interact, existing methods must enumerate all possible combinations for optimization. However, the combinatorial explosion across inputs makes exhaustive optimization intractable. 

_2) Heterogeneous requests_ . When batching for efficiency, requests often contain heterogeneous and dynamic data properties. As shown in Figure 1(b), different requests may share prompts or generation shapes, creating optimization opportunities. However, existing methods assume uniform properties within a batch, making it difficult to exploit such heterogeneity. 

We present ChituDiffusion, a diffusion service system that leverages the data properties exhibiting _locality_ across computations. Consecutive operators often share optimization opportunities induced by these properties—e.g., circles with the same color in Figure 1(a). ChituDiffusion decomposes the pipeline into _dGraphs_[1] , which explicitly capture data properties, enabling decoupled handling through codesigned compilation and runtime optimizations. As a result, ChituDiffusion enables a wide range of data-propertyaware optimizations beyond existing frameworks, while maintaining a reasonable optimization cost. 

In ChituDiffusion, we make the following design decisions. 

> 1The prefix “d” signifies diffusion, data properties, and decomposition. 

During static compilation, ChituDiffusion decomposes the pipeline into dGraphs using symbolic variables to represent and propagate input data properties. Each dGraph is compiled into multiple dEngines, each specialized for a uniform property configuration. By reusing dEngines across requests with different properties, ChituDiffusion performs optimizations for diverse input properties while incurring only a small compilation overhead. 

At runtime, heterogeneous requests are decomposed into fine-grained dTask (dGraph-level tasks). A dynamic programming based scheduler identifies and batches dTasks with the same properties. These dTasks batches are efficiently processed by corresponding dEngines that have been optimized for uniform data properties. Input and output properties are inferred asynchronously, enabling the overlap of scheduling and execution processes to mitigate scheduling overhead. 

Specifically, ChituDiffusion implements two key dataproperty optimizations to accelerate diffusion pipelines (§2.2): redundancy and dynamic shapes. Redundancy propagation and elimination rules are designed to detect and remove dimension-level redundant computations and memory accesses in tensor algebra. To support batched requests with dynamic shapes (hereafter referred to as _raggedness_ ), ChituDiffusion applies _raggedness regularization_ , transforming dynamic-shape operations into kernel-compatible forms without requiring new ragged kernels. 

We evaluate ChituDiffusion on five diverse diffusion applications, covering image and video generation with several widely used model extensions. The evaluation shows that ChituDiffusion achieves a throughput up to 2 _._ 13× (1 _._ 58× on average) higher than widely-used DNN optimizers and diffusion-specific systems, showing the effectiveness of ChituDiffusion’s data-property-aware optimizations on diffusion models. 

This paper makes the following contributions: 

- We observe the key data characteristics of diffusion model requests for optimizations, which exhibit locality in pipelines. 

- We propose recomposed optimizations for pipelines and requests, orchestrating compile-time and runtime optimizations to exploit complex data properties. 

- We implement ChituDiffusion, a system for diffusion pipeline optimization, which outperforms the throughput of state-of-the-art frameworks by up to 2 _._ 13×. 

## **2 Background and Motivation** 

## **2.1 Diffusion Models** 

Diffusion models generate data by simulating a denoising process that transforms random noise into a target distribution. Figure 3 illustrates the pipeline of diffusion-based applications such as text-to-image and image-to-image generation. Prompts are first embedded by CLIP [50], then used to guide a U-Net [16] base model that iteratively denoises a latent 

538 

ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [505 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
1e6<br>1.0 80%<br>40 DataControlOthers 0.8 1.0 PromptsRequests 60% 512×768 32.7%<br>0.6<br>40%<br>20 0.4 0.5<br>7.7%<br>0.2 20% 512×512 .. .. ..<br>0 0.0 0.0 0%<br>T2I T2I with I2I Inpaint 0.0 0.2 0.4 0.6 0.8 1.0 1(unique) 2-4 6-10 >10<br>ControlNet (I2I) Number of duplicate occurrences Distribution of shapes<br>(a) (b) (c) (d)<br>Prompt counts<br>Number of inputs<br>Proportion of requests<br>**----- End of picture text -----**<br>


**Figure 2.** (a) The count of inputs of SDXL-based diffusion pipelines. T2I and I2I represent the text-to-image and image-toimage tasks, respectively. Inpainting means image partial repaint in the selected area. Control inputs influence the pipeline control flow, such as iteration counts. (b) Consecutive text-to-image requests in DiffusionDB [62], a request database of an online text-to-image service. (c) Duplicate prompt occurrences in DiffusionDB counting by prompts and requests. (d) Shape distribution of the top 300 popular generative images in Civitai [17], which contain 112 different shapes. Only images with generation settings attached are included. 

**==> picture [241 x 132] intentionally omitted <==**

**----- Start of picture text -----**<br>
Denoising iterations<br>Random Noise<br>Input  CLIP U-Net U-Net Post-<br>Language<br>prompts 1 🔤 model base refiner processing<br>Input  CLIP<br>Language<br>prompts 2 🔤 model Image<br>🖼<br>Image 🖼 VAE<br>encoder<br>Data Extensions<br>(ControlNet,  Controlling  Extensions Controlling<br>Model LoRA, ...)  parameters parameters<br>Text-to-image<br>Image-to-image<br>**----- End of picture text -----**<br>


**Figure 3.** Application pipelines based on the SDXL model. 

image initialized from random noise or an input image. A U-Net refiner further enhances the result through additional denoising rounds, and the final latent is decoded into a visible image, optionally followed by super-resolution [24, 25, 54]. 

While U-Net architectures dominate traditional diffusion models, transformer-based designs are gaining traction. Models such as DiT [46], SD3 [9], Hunyuan series models[40, 57] and FLUX [34, 35] leverage self-attention for improved generation quality, and extensions like DiT-3D [44], DiT-MoE [69], RFDiffusion [63] and TerDiT [42] further adapt this paradigm to 3D modeling, efficient scaling, protein structure prediction and resource-constrained deployment. 

## **2.2 Motivation** 

Resulting from the distinct computation patterns of diffusion models, their complex structures and unique data properties provide new optimization opportunities. 

**Complex pipelines with flexible customization.** Diffusion models exhibit high flexibility, necessitating numerous configurable inputs such as positive and negative prompts, one or more reference images, guidance strength, plug-andplay extensions, etc. As shown in Figure 2(a), a single diffusion pipeline based on the SDXL model is able to take more than 20 inputs for a generation request. 

To meet the requirements of different applications, diffusion pipelines are extensively customized to achieve flexible functionalities. For example, the image-to-image pipeline in Figure 3 extends the basic text-to-image pipeline by incorporating a VAE encoder model to encode input images into latent space. Additionally, application designers and users can apply various extensions, such as ControlNet [67] and LoRA [31], to generate results in different visual effects and artistic styles. Consequently, varied input data across applications offers unique optimization chances, necessitating effective application-specific optimization strategies. 

**Diverse Data Characteristics in Requests** Diffusion model serving systems must accommodate requests with diverse data characteristics. These mainly manifest as _correlative requests with partial redundancy_ and _variability in generation shapes_ , arising from both application and user sides. 

On the application side, diffusion models are inherently flexible with numerous input parameters, but deployments often fix certain values for specific use cases. In addition, generation shapes strongly affect output quality [47], such as the posture of individuals in generated images. 

From the user side, correlative requests with varied generation shapes are common, as users cannot anticipate input effects without execution. They often submit multiple related requests for grid searches, producing several correlated inputs and manually selecting the best results. Unlike conventional vision tasks, diffusion models cannot simply crop or pad inputs for batching. This workflow, supported by mainstream frameworks [3–5], improves productivity. Figure 2(b–c) shows that most prompts generate more than five correlated requests in a public text-to-image trace. 

Figure 2(d) shows that generation shapes are highly diverse: over 30% of requests use 512×768, yet more than 100 distinct shapes appear among the 300 most popular images. Naive batching by identical shapes leaves many requests unbatchable, reducing efficiency. While requests 

539 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

**==> picture [241 x 272] intentionally omitted <==**

**----- Start of picture text -----**<br>
Compiler<br>APP Pipelines<br>Application characteristic<br>SDXL+Refine Prompt: different or identical  dGraph recompose<br>Image:   uniform or ragged<br>SD15+ControlNet Diffusion pipelinedGraph 2<br>SVD+ControlNet Diffusion pipeline identification1 dGraphPrompts dGraph 1 dGraph 3<br>… Prompts Images<br>Images<br>Property-aware<br>2 compilation<br>Scheduler<br>dEngine library<br>User requests dTask pool<br>Redundant Ragged Redundant<br>𝑇� 𝑃1,512 × 768 Iteration<br>PromptsPrompts Images PromptsID<br>A dog/cat 𝑇� 𝑃1, 512 × 512 3 dTask<br>is walking… 𝑇𝑇�� 𝑃2, 512 × 512𝑃1, 768 × 768 schedulingPromptsPrompts ImagesImages RedundantPromptsIteration ID<br>… Async property<br>4 inference<br>Request outputs Executor<br>P1 P1 … PromptsPrompts Images PromptsIteration ID<br>P1 P2<br>Redundant computation dGraph<br>Redundant memory access dEngine<br>Compile-time process Promptsmpts Imagesages Iteration omptsID<br>Runtime process …<br>**----- End of picture text -----**<br>


**Figure 4.** Overview of ChituDiffusion. 

often share inputs and intermediate results—creating optimization opportunities—batching strategies face inherent trade-offs: kernels for varied shapes enable parallelism but incur overhead for uniform shapes. Consequently, neither approach alone achieves ideal performance. 

## **3 Overview** 

Figure 4 illustrates the workflow of ChituDiffusion on an image generation pipeline. ChituDiffusion accelerates it in four steps. At compile time, ChituDiffusion optimizes pipelines with the knowledge of application characteristics, i.e., requests probably have identical prompts but varied image resolutions. 

**1. dGraph identification (§4.1):** ChituDiffusion performs symbolic data property analysis to infer potential optimization opportunities, taking application characteristics as the initial input properties. According to the propagating results, diffusion pipelines are recomposed into dGraphs which share the same optimization enabling conditions. 

**2. Data-property-specialized compilation (§4.2):** By inferring possible input data properties for each dGraph, ChituDiffusion compiles it into a series of dEngines, each of which is specially optimized for certain data properties. For example, dEngine 1 is optimized for the scenario where inputs are redundant (§4.2) and ragged (§6.1). 

At runtime, ChituDiffusion recomposes user requests into dTasks according to the dGraph-level pipeline. 

**3. dTask scheduling (§5.1):** To efficiently execute requests with heterogeneous data properties, ChituDiffusion leverages dynamic programming to pack dTasks with conforming data properties into a batch, which is dispatched to the matched dEngines for efficient execution. In Figure 4, ChituDiffusion packs four dTasks into two batches to exploit the same prompt and the uniform generation shape. 

**4. Asynchronous property inference (§5.2):** ChituDiffusion infers the data properties of dTask outputs without actual execution. Therefore, ChituDiffusion is able to overlap dTask scheduling with actual execution, avoiding scheduling overhead. 

## **4 Compile-Time Optimizations** 

Due to the large number of inputs associated with diffusion models, directly optimizing a pipeline for all possible scenarios leads to an impractically large number of optimized execution engines. Taking a pipeline with _𝑛_ inputs for example, merely accounting for the presence or absence of tensor dimensional redundancy can yield up to 2 _[𝑛]_ execution engines, incurring prohibitive optimization costs. 

ChituDiffusion’s _compiler_ recomposes a pipeline into a set of dGraphs to enable fine-grained optimizations (§4.1) and then compiles each dGraph into multiple dEngines via property-aware optimizations (§4.2). 

## **4.1 Symbolic dGraph Identification** 

To infer how properties are propagated within the dGraph structure, ChituDiffusion is equipped with a set of symbolic data property propagation rules tailored for tensor operations. For conciseness, this section takes dimensional redundancy as a running example(§2.2). Examples include a batch of requests with partially duplicate inputs (redundancy in the batch dimension) and identical data in grayscale images (redundancy in the color channel dimension). The method can also be extended to deal with other data properties as well, such as dynamic tensor shapes. 

**Symbolic property propagation.** ChituDiffusion utilizes application characteristics to initialize propagation. Diffusion pipelines are usually tailored to a specific usages, developers have prior knowledge about pipeline inputs. Some inputs usually vary between requests, while certain inputs are intended to be the same for all requests (e.g., a fixed prompt in style transfer or a shared backbone in protein generation ). Additionally, there are inputs that may potentially be the same but can only be determined by the incoming request. To deal with the dynamics and complexity of data in diffusion models, ChituDiffusion leverages _symbolic data properties_ to perform analysis. Figure 5(a) shows a simplified example of a image generation task with gray scale image controlnet and LoRAs. ChituDiffusion assigns symbolic variables to the data properties of inputs, notated as _𝛼, 𝛽_ , and _𝛾_ . For inputs with a fixed property determined 

540 

ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [243 x 139] intentionally omitted <==**

**----- Start of picture text -----**<br>
latentlatent controlnet LoRA operator dEngine<br>AA B C                                                                                                              D E dGraph (d-e) Property of (B)<br>𝛼 a 𝛽 d 𝑇 𝛾 𝛽 𝛽= 𝑇 Compile 𝑇<br>b 𝛽= 𝐹 𝐹<br>e<br>c<br>𝛼 𝛽 dGraph (g-h) Properties of (f, D, E)<br>f<br>𝛼∧𝛽= 𝑇, 𝛽= 𝑇, 𝛾= 𝑇 𝑇, 𝑇, 𝑇<br>𝛼∧𝛽<br>𝛼∧𝛽∧𝛾 … g 𝛼∧𝛽= 𝑇, 𝛾= 𝑇, 𝛽= 𝐹 Pruned<br>h 𝛼∧𝛽= 𝑇, 𝛽= 𝐹, 𝛾= 𝑇 𝑇, 𝑇, 𝑇<br>𝛼∧𝛽∧𝛾<br>AA …<br>(a) dGraph recomposition (b) Property conditions (c) dEngines<br>**----- End of picture text -----**<br>


**Figure 5.** (a) Partitioning a DFG into dGraphs according to property expressions. _𝛼, 𝛽,_ and _𝛾_ are symbolic property variables. _𝑇_ and _𝐹_ are true and false. ∧ means logical AND. (b) Property condition expressions of dGraph (g-h). (c) dEngines with input property requirements of dGraph (d-e). 

by application characteristics, actual values are provided. In the tensor redundancy propagation, we use a boolean value to represent if a dimension is duplicate, such as _𝑇_ (i.e., true) for input _𝐶_ in Figure 5(a). 

To identify various forms of data properties propagated by tensor algebra, ChituDiffusion utilizes symbolic propagation rules. Table 1 presents typical dimension redundancy rules derived from boolean algebra. Due to the complexity of tensor operators, both input and output are represented as vectors, with each element denoting a per-dimension propery. For example, in 2D convolution, if both input tensors exhibit redundant length and width dimensions, the redundancy propagates to the output. To accommodate new operators, ChituDiffusion also supports user-defined propagation rules. 

During the propagation, ChituDiffusion maintains symbolic expressions for tensor properties. As illustrated in in Figure 5(a), nodes (a-c) generate outputs with redundancy if tensor _𝐴_ is redundant. 

ChituDiffusion takes a whole pipeline as a single data flow graph (DFG) for effective recomposition.In diffusion pipelines, denoising loops enable later updates to affect earlier tensors. To avoid conflicting expressions, ChituDiffusion unrolls initial iterations until loop inputs stabilize, which converges within a few steps since such loops are neither nested nor overlapping. 

**dGraph identification.** Based on symbolic propagation, ChituDiffusion partitions the pipeline into dGraphs by grouping consecutive operators with identical output property expressions, as common properties indicate shared optimization opportunities. As shown in Figure 5(a), each operator belongs to a dGraph delineated by red lines. 

ChituDiffusion employs output properties as the recompose criterion, since input properties usually enable only operator-specific optimization. For example, node g in 

Figure 5(a) can be optimized if _𝛼_ ∧ _𝛽_ or _𝛽_ are redundant, but the condition does not extend beyond this operator. ChituDiffusion considers these fine-grained optimization opportunities during dGraph compilations in §4.2. To avoid scheduling overhead, small dGraphs —such as the singlenode f—are merged into subsequent dGraphs during postprocessing. 

**Table 1.** Tensor redundancy propagation rules. _𝑎𝑖_ and _𝑏𝑖_ represent the redundancy properties of _𝑖_ -th dimensions (starting from 1) of the first and second inputs, respectively. ∧ means logical AND. 

|Category<br>Operators[Layout]<br>Output redundancy|Category<br>Operators[Layout]<br>Output redundancy|
|---|---|
|Unaryelementwise<br>ReLU,Tanh[2D]<br>[_𝑎_1_,𝑎_2]||
|Binaryelementwise<br>+_,_−_,_×_,_÷ [2D]<br>[_𝑎_1 ∧_𝑏_1_,𝑎_2 ∧_𝑏_2]||
|Linear<br>Batch Matmul[NHW]<br>[_𝑎_1 ∧_𝑏_1_,𝑎_2_,𝑏_3]||
||w/o padding<br>[_𝑎_1_,𝑏_1_,𝑎_3∧_𝑎_4∧_𝑏_3∧_𝑏_4_,_<br>_𝑎_3 ∧_𝑎_4 ∧_𝑏_3 ∧_𝑏_4]|
|Convolution<br>Conv2D<br>[NCHW]||
||w/ padding<br>[_𝑎_1_,𝑏_1_, 𝑁, 𝑁_]|



## **4.2 Data-Property-Specialized Compilation** 

By analyzing property expressions, ChituDiffusion selectively compiles a dGraph into several execution engines, which are named dEngines. Each dEngine is specially optimized for one specific input data property. 

**Selective dEngine generation.** For each dGraph, ChituDiffusion optimizes dEngines by enumerating potential property expressions of the dGraph inputs. For example, dGraph (d-e) in Figure 5(a) can be optimized when its inputs are redundant. ChituDiffusion identifies its property condition is _𝛽_ , which have two possible values, as shown in Figure 5(b). By optimizing a dGraph under two property conditions, ChituDiffusion generates two dEngines covering all optimization scenarios, and combining suitable dEngines achieves the same effect as monolithic pipelinelevel optimization. 

Furthermore, as a single dEngine can be reused across diffusion model requests with different data properties, it avoids the expensive cost of monolithic optimization of entire pipelines. However, not all dEngines are necessary, as certain condition combinations are rare or yield limited performance benefits. For instance, the third condition of dGraph (f-h) in Figure 5(c), _𝛼_ ∧ _𝛽_ = _𝑇,𝛾_ = _𝑇, 𝛽_ = _𝐹_ , is unsatisfiable since _𝛽_ can not be both redundant and not redundant. ChituDiffusion prunes two categories of dEngines. For conciseness, we discuss a dEngine with condition expressions _𝑒_ 1 ∧ _𝑒_ 2. 

The first is conflicting conditions. If _𝑒_ 1 ∧ _𝑒_ 2 is unsatisfiable for all input properties, the dEngine is never used and thus pruned. Naive enumeration of tensor properties often generates such cases by ignoring constraints from pipeline inputs and application semantics. For example, tensors _𝐵_ and _𝐸_ in 

541 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [241 x 61] intentionally omitted <==**

**----- Start of picture text -----**<br>
𝐾𝐾 𝑉𝑉 𝐾 𝑉<br>𝑄𝑄00 𝑄𝑄00𝐾𝐾 [𝑇][𝑇] σσ 𝑄𝑄10𝐾𝐾 [𝑇][𝑇] σσ 𝑄𝑄00𝐾𝐾 [𝑇][𝑇] 𝑉𝑉 𝑄𝑄10 𝑄𝑄10 [𝐾][𝑇] σ 𝑄𝑄10 [𝐾][𝑇] σ 𝑄𝑄10 [𝐾][𝑇] [𝑉]<br>(a) Original with redundant K and V (b) Eliminate redundant memory access<br>**----- End of picture text -----**<br>


**Figure 6.** Eliminating redundant memory access for the attention operation. _𝜎_ denotes Softmax. Element-wise operations are omitted. 

Figure 5(a) always share the same property, so optimizations assuming them to differ are invalid. 

The second is inessential conditions. If a condition _𝑒_ 1 ∧ _𝑒_ 2 enables only marginal optimizations, ChituDiffusion prunes it to reduce both optimization overhead and runtime scheduling cost. The importance of each condition is determined by its optimization speedup, and those below a threshold (5% in the current implementation) are discarded. For example, the condition _𝛼_ ∧ _𝛽_ = _𝑇,𝛾_ = _𝐹, 𝛽_ = _𝐹_ is pruned in dGraph (f–h) because it affects only a single operator. Users may also define more sophisticated pruning criteria, such as leveraging performance models to estimate dEngine speedup. 

**Redundancy elimination.** To optimize dGraphs under different inputs automatically, ChituDiffusion adopts the rulebased optimization method.ChituDiffusion is equipped with dimension-level redundant elimination rules tailored for each tensor operator. To eliminates redundant computations by analyzing operator inputs in execution order and applying optimization rules. Redundant dimensions are marked and later restored through broadcasting, ensuring maximal elimination while preserving equivalence. 

For redundant memory access, ChituDiffusion utilizes the equivalent transformations of linear algebra to transform them into redundancy-free computations based on existing kernels. Figure 6(a) shows an attention operation with redundant K and V tensors. To optimize it, ChituDiffusion compresses the K and V tensors along the redundant batch dimension and concatenates the Q tensors from different requests into a single one. 

## **5 Runtime Design** 

At runtime, ChituDiffusion ’s _scheduler_ dynamically groups dTasks with uniform data properties into batches, dispatches the corresponding pre-compiled dEngines to the _executor_ , and asynchronously infers data properties to overlap scheduling with execution (§5.1, §5.2). 

## **5.1 Heterogeneous dTask Scheduling** 

To handle heterogeneous requests efficiently, ChituDiffusion aggregates requests within a scheduling window 

**Algorithm 1** Data-aware execution plan generation. 

|1:|**Input:**dGraph_g_, dTask pool|**Input:**dGraph_g_, dTask pool|D, dEngine libraryL|D, dEngine libraryL|
|---|---|---|---|---|
|2:|**Output:**Execution planE||||
|3:|T ={∅: 0}|_⊲_Initialize the||map of execution time|
|4:|E ={∅:∅}|_⊲_Initialize the map of execution plans|||
|5:|R =Uniqe(D[_g_])|_⊲_Unique dTasks of the given dGraph|||
|6:|Search(G_,_R)||||
|7:|**return**E[R]||||
|8:|||||
|9:|**procedure**Search(_g,_|R)|_⊲_R is the remaining dTasks||
|10:|**if** R ∈T **then**||||
|11:|returnT [R]|||_⊲_Memorized results|
|12:|_𝑡𝑚𝑖𝑛_=∞||||
|13:|**for**S ∈L[_g_] **do**|||_⊲_Enumerate dEngines|
|14:|B = GetLargestBatch(R_,_S)||||
|15:|_𝑡𝑛𝑒𝑤_= Search(_g_,R −B)+EstTime(_g_,S)||||
|16:|**if**_𝑡𝑛𝑒𝑤< 𝑡𝑚𝑖𝑛_**then**||||
|17:|_𝑡𝑚𝑖𝑛_=_𝑡𝑛𝑒𝑤_||||
|18:|P =E[R −B] ∪{(B_,_S)}|||_⊲_Partial plan|
|19:|T [R] =_𝑡𝑚𝑖𝑛_|||_⊲_Memorization|
|20:|E[R] =P||||
|21:|return_𝑡𝑚𝑖𝑛_||||



|Scheduler|Scheduler|Scheduler|Executor|Executor|Executor|
|---|---|---|---|---|---|
|𝑇1<br>𝑃1,512 × 768<br>𝑇2<br>𝑃1, 512 × 512<br>𝑇3<br>𝑃1, 768 × 768<br>𝑇4<br>𝑃2,512 × 512<br>dTask pool||Property inference|Async<br>task<br>queue|dEngines library<br> <br>𝐸3<br>Redundant<br>Uniform<br>𝐸4<br>Redundant<br>Ragged<br>𝐸1<br>Uniform<br>𝐸2<br>Ragged||
|||Schedule plans|||Uniform|
|||𝐸2: 𝑇1, 𝑇2, 𝑇3, 𝑇4|||Ragged|
|||𝐸1: 𝑇4 | 𝐸4: 𝑇1, 𝑇2, 𝑇3|||Redundant<br>Uniform|
|||𝐸4: 𝑇1, 𝑇3 | 𝐸1: 𝑇2, 𝑇4|||Redundant<br>Ragged|



**Figure 7.** Scheduling on dTasks and dEngines of a dGraph. dEngine inputs are non-redundant with uniform shapes at default without annotation. 

(parameterized by window size), decomposes them into finegrained dTasks, and stores them in a shared task pool (Figure 7). It then schedules dTasks from different dGraphs independently, following their execution order in the dGraphlevel pipelines. 

For dTasks based on the same dGraph, the heterogeneity of data properties carried by user requests provides a large scheduling space. For example, Figure 7 shows four dTasks of the same dGraph, each taking two inputs. We can batch all dTasks and run dEngine _𝐸_ 2, which is most general but less efficient since data properties are not used. Alternatively, we can execute _𝑇_ 1 _,𝑇_ 2, and _𝑇_ 3 with _𝐸_ 4 to exploit the identical prompt, or execute _𝑇_ 2 and _𝑇_ 4 with _𝐸_ 1 to create a batch of uniform shape. Similar trade-offs exist for dTasks with more inputs and diverse data properties. 

To determine an efficient batching plan for dTasks, ChituDiffusion employs a dynamic programming approach. Algorithm 1 details this workflow. For each dGraph, ChituDiffusion merges dTasks with identical inputs into a single dTask (Algorithm 1) and broadcasts the results after 

542 

ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [505 x 155] intentionally omitted <==**

**----- Start of picture text -----**<br>
Data Data with redundancy Operation Operation with redundant computation eliminated memory access eliminatedOperation with redundant  Ragged data or operations<br>imagesLatent Inputs PromptsPrompts Fine-tuning Finetuning weightsweights imagesLatent Inputs PromptsPrompts Fine-tuning Finetuning weightsweights imagesLatent Inputs PromptsPrompts Fine-tuning Finetuning weightsweights imagesLatent Inputs PromptsPrompts Fine-tuning Finetuning weightsweights<br>Concat<br>MM LoRA MM LoRA MM LoRA MM LoRA MM LoRA MM LoRA MM LoRA MM LoRA MM LoRA<br>… dGraph 2<br>Add Add Add Add Add Add Add Add Add Cross  (loop invariant)<br>` Attention<br>Batch Matmul Batch Matmul Matmul<br>…<br>Softmax Softmax Softmax Cross<br>Attention<br>Batch Matmul Batch Matmul Matmul<br>(a) dGraph decomposition (b) Redundancy elimination (c) Ragged operation regularization (d) Invariant tensor optimization<br>dGraph 1<br>**----- End of picture text -----**<br>


**Figure 8.** Optimizations on the core structure of a SDXL layer. The blue circle in (d) shows the entry of the denoising loop. 

execution. Then, ChituDiffusion explores different combinations of batches by enumerating properties required by dEngines (Algorithm 1).For a given property requirement, ChituDiffusion finds the largest batch satisfying it (Algorithm 1). 

To avoid duplicate searches, ChituDiffusion always includes the first remaining dTask in the current batch. Since a plain dEngine with the most general requirements always exists, legal plans are guaranteed. Each batch’s execution time is estimated using a performance model tailored to its dEngine, and the remaining dTasks are recursively combined into batches in the same manner. After enumerating all dEngines for a given search state, ChituDiffusion records the estimated optimal execution time and batching plan to accelerate subsequent scheduling (Algorithm 1). 

ChituDiffusion adopts a lightweight performance model tailored for diffusion models to achieve both efficiency and accuracy. Since compute-intensive operations dominate execution, their costs scale either linearly with input size (e.g., convolution, linear layers) or quadratically (e.g., attention). To capture this behavior, ChituDiffusion applies ordinary least squares regression between input-related metrics and execution time, using a dataset constructed from example inputs of varying sizes and their batch prefixes. For models that involve complex computations such as temporal and spatial attentions in video generation models, ChituDiffusion replaces the total tensor size in the performance model with the corresponding dimension sizes, allowing for a more precise estimation of the execution time. 

## **5.2 Asynchronous Data Property Inference** 

In ChituDiffusion, the scheduler and executor operate asynchronously to hide decomposition and scheduling overhead, but the absence of data properties for unexecuted dTask inputs poses a key challenge. 

To address this problem, ChituDiffusion uses data property expressions in §4.1 to infer data properties for each 

dTask. By evaluating output property expressions with real input properties, ChituDiffusion efficiently infers data properties of each dTask output separately. 

To infer redundancy across multiple requests, ChituDiffusion develops a tensor _fingerprint_ technique to recognize redundant tensors without real execution. For request inputs, ChituDiffusion uses a lightweight hash function to calculate their fingerprint in a time complexity that is linear to the number of elements in tensors. Additionally, applications and users are also able to directly mark duplicate inputs in correlative requests, in which case ChituDiffusion assigns the same unique values to them as fingerprints without hashing. 

ChituDiffusion calculates fingerprints on each dGraph in the execution order. Calculating fingerprints operator-byoperator can be time-consuming for complex dGraphs. Since dGraphs perform deterministic computation, their outputs are only dependent on inputs. According to the DFG of a dGraph, ChituDiffusion records which inputs impose influence on outputs for each operator and propagate this information through the data-flow equation. This enables us to directly derive output fingerprints from the input fingerprints. To calculate output fingerprints, ChituDiffusion uses 

FP[ _𝑜𝑢𝑡𝑝𝑢𝑡𝑖_ ] = Φ(FP _𝑔,_ FP[ _𝑖𝑛𝑝𝑢𝑡𝑑𝑒𝑝𝑖,_ 0 ] _, . . . ,_ FP[ _𝑖𝑛𝑝𝑢𝑡𝑑𝑒𝑝𝑖,𝑛_ ]) , where Φ is an operand-commutative hash function, FP _𝑔_ is a unique identifier of the current dGraph to distinguish computations of different dGraphs, and _𝑖𝑛𝑝𝑢𝑡𝑑𝑒𝑝𝑖,_ 0 _, . . . ,𝑖𝑛𝑝𝑢𝑡𝑑𝑒𝑝𝑖,𝑛_ have influence on _𝑜𝑢𝑡𝑝𝑢𝑡𝑖_ . Thus, ChituDiffusion can identify redundant dTask inputs by simple fingerprint comparison without actual tensor comparisons. 

## **6 Implementation** 

Figure 8 illustrates how ChituDiffusion applies data-property-aware optimizations to the core SDXL layer. Given requests with ragged image shapes but identical prompts and fine-tuning weights, ChituDiffusion firstly detects two 

543 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

**==> picture [241 x 140] intentionally omitted <==**

**----- Start of picture text -----**<br>
Regular  Ragged  Regular  Ragged data-sharing  Ragged data-independent<br>Data data operation operation operation<br>𝑏, ෝ𝑚, 𝑘 𝑘, 𝑛 𝑛, 𝑐, ℎ, ෝ𝑤 [෠] 𝑐, 1, 1 𝑛, 𝑐, ℎ, ෝ𝑤 [෠] 𝑓, 𝑐, 𝑟, 𝑠<br>Matmul Add Conv<br>𝑏, ෝ𝑚, 𝑘 𝑛, 𝑐, ℎ, ෝ𝑤 [෠] 𝑐, 1, 1 𝑛, 𝑐, ℎ, ෝ𝑤 [෠] 𝑓, 𝑐, 𝑟, 𝑠<br>Ragged<br>T/R Regular T/R T/R Im2col T/R<br>𝑏ෝ𝑚, 𝑘 𝑘, 𝑛 𝑛ℎෝ𝑤, 𝑐 [෠] 𝑐 𝑛ℎෝ𝑤, 𝑐𝑟𝑠 [෠] 𝑐𝑟𝑠, 𝑓<br>Matmul Add Matmul<br>T/R T/R T/R<br>(a) Linear (b) Elementwise (c) Convolution<br>**----- End of picture text -----**<br>


**Figure 9.** Ragged operation regularization. _𝑥_ ˆ denote a ragged dimension _𝑥_ . If dimension _𝑥_ ˆ is merged with the batch dimension, the new merged dimension is a regular dimension. T/R represents transpose and reshape operators. 

dGraphs and optimizes them independently, eliminating redundant computation and memory accesses from shared inputs (Figure 8b). It then applies ragged operation regularization (§6.1) to transform irregular operations into kernelcompatible equivalents (Figure 8c). Finally, ChituDiffusion identifies invariant tensors within the pipeline (§6.2). 

## **6.1 Ragged Operation Regularization** 

To enable data-property-aware optimizations for ragged requests(input with various shapes), ChituDiffusion must both identify dGraphs with common raggedness patterns for compile-time dEngine construction and infer output shapes efficiently at runtime. Building on the symbolic shape propagation, ChituDiffusion represents ragged dimensions as symbolic variables and substitutes actual inputs at runtime to obtain final output shapes. Since Handcrafting ragged kernels for all operators is costly and challenging for existing automatic kernel generators [19, 66], ChituDiffusion employs _ragged operation regularization_ . Operations are classified as data-sharing operations with shared inputs or weights (e.g., convolution, linear layers) and data-independent operations without shared weights (e.g., transpose, reduce). Ragged operations are opportunistically transformed into standard operators and ragged data-independent operations, enabling efficient execution using existing kernel libraries with minimal effort. 

Since data-independent operations do not share data across requests in a batch, embarrassingly parallel execution is efficient for each request. Based on existing tiling plans and computing microkernels for regular (non-ragged) operators, ChituDiffusion partitions each request into a set of tiles, which are mapped to GPU thread blocks with a round-robin policy during batched execution. 

ChituDiffusion supports ragged data-sharing operations by transforming them into equivalent regular operators with the help of ragged data-independent operations. For instance, 

a ragged Matmul can be regularized by fusing the batch dimension _𝑏_ and ragged dimension _𝑚_ ˆ via transpose and reshape, similar to concatenating matrix heights (Figure 6(b)). Other operations, such as ragged elementwise ops and convolutions, can be handled with transpose and image to column [60] (Figure 9(bc)). These transformations are feasible because shared weights have fixed dimensions, allowing ragged inputs to be flattened across shared data. ChituDiffusion thus applies a set of graph transformation rules to regularize ragged data-sharing operations, supplemented with a few ragged data-independent operations, and can flexibly extend this rule set for new models. 

## **6.2 Invariant Tensor Elimination** 

Diffusion models contain two types of _invariant tensors_ : constants, predetermined by applications, and loop-invariants resulting from iterative denoising. 

ChituDiffusion employs a lightweight four-state detection algorithm (constant, loop-invariant, loop-variant, unknown) to identify them. Properties are initialized from tensor definitions and iteratively propagated along operators with a priority hierarchy. 

Detected constants are precomputed at compile time, while loop-invariants are hoisted outside loops. ChituDiffusion further supports multi-value constants, allowing selective input fixing to trade off performance and generation diversity. 

## **6.3 Extensibility and Architecture Agnosticism** 

ChituDiffusion achieves architecture-agnostic support for diverse diffusion models—spanning from pure Transformers to mixed Transformer-Convolutional structures—through two core mechanisms: 

- **Universal Intermediate Representation:** By utilizing Data Flow Graphs (DFGs) as a universal IR, ChituDiffusion expresses arbitrary pipelines uniformly. This approach decouples optimization logic from specific neural network layouts, enabling the system to exploit broad optimization opportunities derived from high-level application requirements (§2 _._ 2) rather than being limited by architectural specifics. 

- **Operator-Centric Optimization:** ChituDiffusion performs analysis and transformations at the operator level. Its optimization rules target individual computational steps rather than monolithic network blocks. Integrating novel operators requires only minimal, localized effort: (1) defining symbolic propagation rules for data properties, and (2) providing corresponding optimized kernels. 

Once these definitions are provided, the ChituDiffusion scheduler automatically orchestrates the rest of the workflow—applying existing optimization rules and compiling the DFG into high-performance _𝑑𝐸𝑛𝑔𝑖𝑛𝑒𝑠_ without further manual intervention. 

544 

ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [498 x 147] intentionally omitted <==**

**----- Start of picture text -----**<br>
(A) PyTorch (B) PyTorch + Inductor (C) TensorRT (D) Stable Fast (K) Katz(per GPU) (E) Chitu-Diffusion<br>0.4 A100 2.1 × 1.9 × 1.4 × 1.4 × 1.1 ×<br>20 0.02 0.75 1.0<br>0.2 0.50<br>10 0.01 0.5<br>0.25<br>0.0 0 0.00 0.00 0.0<br>A B C D E A B C K D E A B C D E A B C D E A B C D E<br>0.6 H100 2.2 × 30 1.6 × 1.5 × 1.3 × 1.5 1.0 ×<br>0.06<br>0.4 20 0.04 1.0 1.0<br>0.2 10 0.02 0.5 0.5<br>0.0 0 0.00 0.0 0.0<br>A B C D E A B C K D E A B C D E A B C D E A B C D E<br>refine edit video venti grande<br>Throughput (req/s) Out of range<br>Throughput (req/s) Out of range Out of range<br>**----- End of picture text -----**<br>


**Figure 10.** Throughput improvement on multiple diffusion model applications. 

## **7 Evaluation** 

## **7.1 Experiment Setup** 

**ChituDiffusion implementation.** We implement ChituDiffusion based on both C++ and Python, and reuse some components from Diffusers [61], Triton [59], Stable Fast [1], and FlashAttention [21, 22]. Users are able to customize diffusion model pipelines in ChituDiffusion to support various applications. To support ragged batching, we implement four ragged data-independent operation kernels based on Triton and CUDA. 

**Platform.** The evaluation was conducted across two server configurations: one equipped with an NVIDIA A100 40GB PCIe GPU and another with an NVIDIA H100 80GB PCIe GPU. Experiments involving UNet-structured models were evaluated using CUDA 12.1. For the DiT series models, the evaluation utilized CUDA 12.8. The open-source release features a comprehensive infrastructure upgrade, now fully supporting and leveraging PyTorch 2.9 for enhanced performance and compatibility. 

**Workloads.** We evaluate ChituDiffusion on 5 UNet-based diffusion model applications (Table 2) built upon SD1.5 [51], SDXL [47], and SVD [11]. We also conduct the three evaluations on DiT structure diffusion generation scenarios built upon Hunyuan series modelsl[40, 57] and FLUX[34, 35]. 

The applications cover image and video synthesis, with or without controlled generation extensions. To emulate realworld usage, we synthesize non-correlated request traces using default settings unless otherwise noted. Prompt distributions and ragged input shapes follow DiffusionDB [62] and Civitai [17] (Figure 2c–d). Each application adopts its default denoising steps, and random seeds are uniformly sampled. 

## **7.2 Throughput Improvement** 

We evaluate ChituDiffusion against PyTorch v2.1, PyTorchInductor v2.1, TensorRT v8.6, and the diffusion-specific framework Stable Fast v1.0 [1], all tuned to saturate GPU throughput. The applications in Table 2 include both correlative 

**Table 2.** Evaluated applications and shapes of generation results. T, I, and V in the type column represent text, image, and video, respectively. The shapes indicate the height and width of the generated images, as well as the frames, height, and width of the generated videos. 

||Name<br>refne<br>edit|Type<br>T2I<br>I2I|Brief description<br>Generate images refned<br>by diferent prompts<br>Transfer images into<br>multiple styles|Model and extention<br>SDXL, SDXL refner<br>SDXL, LoRA,<br>ControlNet|Shape<br>[1024,1024]<br>[512,512]|
|---|---|---|---|---|---|
||video<br>venti|I2V<br>T2I|Generate videos from images<br>with diferent control efects<br>Generate images from texts|SVD, ControlNet<br>SDXL|[14,576,1024]<br>Ragged|
||grande|T2I|Generate images from texts|SD1.5|Ragged|
||refne-mix<br>refne-dit|T2I<br>T2I|Generate images refned<br>by diferent prompts<br>Generate images refned<br>by diferent prompts|FLUX.1 S, SDXL refner<br>Hunyuanimage,<br>Hunyuanimage refner|[1024,1024]<br>[1024,1024]|
||edit-dit|T2I|Generate images with<br>diferent control stages|Hunyuan-DiT,<br>Hunyuan-DiT ControlNet|[1024,1024]|



requests and standard text-to-image services. As shown in Figure 10, ChituDiffusion delivers up to 2 _._ 13× speedup (1 _._ 58× on average) over the best baseline. For refine, edit, and video, where users issue correlative requests and manually select the best generations for efficient image and video creation [3–5].(e.g., prompt grid search or varying ControlNet periods), baselines process each request independently, missing the awareness of redundancies. TensorRT fails on video due to oversized tensors. By detecting shared inputs, ChituDiffusion eliminates redundant computation, yielding up to 2 _._ 2× on H100 and 2 _._ 1× on A100. 

For venti (SD1.5) and grande (SDXL), which provide standard text-to-image services, redundancy is minimal since requests are independent. Nevertheless, ChituDiffusion still captures optimizations from shared prompts and iteration IDs and the potentially same prompt shown in Figure 4, while effectively batching ragged requests that existing frameworks cannot. The scheduler balancing the efficiency of the uniform-shape dEngines and the larger batch size enabled by the ragged-shape dEngines, which will be further studied in §7.7, yielding 1 _._ 4× speedup on venti and 1 _._ 1× on 

545 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

**==> picture [217 x 77] intentionally omitted <==**

**----- Start of picture text -----**<br>
(A) Torch (B) Torch+Inductor (C) Stable Fast (E) Chitu-Diffusion<br>2.16× 2.97× 0.125 1.40×<br>1.5 0.15 0.100<br>1.0 0.10 0.075<br>0.050<br>0.5 0.05<br>0.025<br>0.0 0.00 0.000<br>A B C E A B C E A B C E<br>Refiner-mix Refiner-DiT Edit-DiT<br>Throughput (req/s) Compile\ Fail<br>**----- End of picture text -----**<br>


**Figure 11.** Throughput improvement on DiT based diffusion model applications. 

grande since SDXL is a much larger model and gains less speedup from batching. 

We further evaluate Katz [37], a state-of-the-art diffusion serving system that supports ControlNet-as-a-service, in the edit application. For fairness, we only test ControlNet (since Katz’s LoRA serving is mathematically inequivalent). In contrast to ChituDiffusion’s single-GPU deployment, we evaluate Katz with 4 H100 GPUs, which is the minimal hardware requirement for Katz to serve a single ControlNet. To fairly compare these two works we normalized the metric as throughput per GPU. As shown in the table, Katz achieves ∼ 0.03s latency per request by serving sequentially, but its throughput per GPU is significantly lower than ChituDiffusion. In the edit scenario, which involves only one iteration with SDXL-Turbo, Katz’s multi-GPU communication overhead severely limits its end-to-end throughput. 

## **7.3 Generalization on DiT-based Models** 

To demonstrate the generality of ChituDiffusion, we extend our data-aware optimization to Diffusion Transformer (DiT) architectures. As ChituDiffusion operates at the operator level rather than relying on specific model topologies, it is inherently architecture-agnostic. DiT components—such as QKV projections, MHSA, and MLPs—are naturally identified as dGraphs within our DFG, allowing the system to decompose and execute common computations across requests efficiently. Figure 12 illustrates the performance on DiTbased applications. ChituDiffusion achieves a 2.2–3.0× speedup in refine scenarios and a 1.4× speedup in edit scenarios. 

Specifically, in refine scenarios, while static compilation baselines like stable-fast failed to support the HunyuanImage refiner pipeline due to inflexible compilation rules, ChituDiffusion leveraged the torch.compile backend, gaining a 3.0× speedup through our data-aware dGraph scheduling. In edit scenarios, dealing with DiT-based ControlNets presents challenges due to varying architectural implementations (e.g., FLUX vs. Hunyuan). We evaluated a dynamic scenario where ControlNet is active only during specific iteration stages (controlled by control_guidance_start/end in diffusers). Unlike static approaches, ChituDiffusion effectively handles these ragged control flows. While emerging 

**==> picture [217 x 114] intentionally omitted <==**

**----- Start of picture text -----**<br>
Data Computation dGraph dTask<br>𝑑𝐺𝑟𝑎𝑝ℎ1<br>Input  𝐼1, 𝑳, 𝑪 𝑑𝐺𝑟𝑎𝑝ℎ3<br>ControlNet<br>image 𝐼2,𝑳, 𝑪 𝑅1 𝐼1, 𝐿, 𝐶, 𝑆1<br>𝑑𝐺𝑟𝑎𝑝ℎ2 𝑳, 𝑪, 𝑆1 𝑅𝑅23 𝐼𝐼11, 𝐿, 𝐶,𝑆, 𝐿,𝐶, 𝑆23<br>Latent  𝑳, 𝑪, 𝑆2 𝑅𝑅45 𝐼𝐼22, 𝐿, 𝐶, 𝑆, 𝐿, 𝐶, 𝑆12<br>image 𝑳, 𝑪, 𝑆3 𝑅6 𝐼2, 𝐿, 𝐶, 𝑆2<br>Condi-<br>tioning Style LoRA<br>······ Post<br>U-Net layer<br>······<br>U-Net layer U-Net layer U-Net layer processing<br>**----- End of picture text -----**<br>


**Figure 12.** Optimizations of application edit. Data _𝐼, 𝐿,𝐶,𝑆_ in dTasks represent input image, latent image, U-Net conditioning, and style LoRA, respectively. The data marked in bold is redundant across requests. 

DiT models introduce complex pipelines and evolving ControlNet support, ChituDiffusion’s flexible design allows it to adapt to these new challenges effectively. 

## **7.4 Ablation Study** 

Application edit provides style transfer services by generating multiple candidate images for each user-input image, preserving object positions but varying visual styles. It leverages SDXL Turbo [55], a variant of SDXL requiring only one denoising step, along with two fine-tuning extensions: ControlNet [67] for spatial control via Canny edges [12], and LoRA for diverse styles. To produce multiple stylized outputs, the application fixes latent noise and U-Net conditioning while applying 16 different LoRA weights. 

As shown in Figure 12, ChituDiffusion exploits these characteristics by recomposing the pipeline into 3 dGraphs at compile time, which are then compiled into dEngines. For example, U-Net is partitioned into input-dependent and input-independent layers, where _𝑑𝐺𝑟𝑎𝑝ℎ_ 2 encapsulates the input-independent computations, enabling reuse across different images and styles. 

At compile time, ChituDiffusion decomposes the pipeline into three dGraphs, compiled into dEngines. U-Net is split into input-dependent and input-independent parts, with _𝑑𝐺𝑟𝑎𝑝ℎ_ 2 containing the latter. At runtime, ChituDiffusion detects only 2 unique dTasks for _𝑑𝐺𝑟𝑎𝑝ℎ_ 1 and 3 for _𝑑𝐺𝑟𝑎𝑝ℎ_ 2, and executes them with data-aware batching to exploit shared inputs (Figure 12). Moreover, invariant tensor elimination identifies _𝑑𝐺𝑟𝑎𝑝ℎ_ 2 outputs as constant, enabling compile-time caching with multi-value support(§6.2). 

Figure 13(a) shows the results of an ablation study. We create a baseline named ChituDiffusion-base by disabling all data-aware optimizations in ChituDiffusion, which achieves comparable performance to the other baseline system. By progressively enabling dTask scheduling, multiversion dEngine compilation, and invariant tensor elimination, the throughput is improved to 1 _._ 29×, 1 _._ 56×, and 

546 

ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

**==> picture [241 x 83] intentionally omitted <==**

**----- Start of picture text -----**<br>
30 100<br>20.95 22.99 73.25 72.57<br>20 17.31 57.56<br>12.34 13.41 50<br>10<br>0 0<br>Stable C.-Diff. +SCH +COMP +IRE Stable C.-Diff. +IRE<br>Fast base Fast base<br>(a) Batch execution (b) Sequential execution<br>Latency (ms)<br>Throughput (req/s)<br>**----- End of picture text -----**<br>


**Figure 13.** Ablation study of application edit. SCH, COMP, and IRE mean dGraph scheduling, dEngine property-aware compilation, and invariant redundancy elimination optimizations, respectively. 

1 _._ 71×. Figure 13(b) presents the request latency when sequentially serving requests without batching, which disables inter-request optimizations in scheduling and multiversioned dEngines. ChituDiffusion achieves a speedup of 1 _._ 3× compared to the best baseline with the help of invariant tensor elimination. This demonstrates that while ChituDiffusion primarily focuses on throughput optimization, its techniques also provide benefits in latency-critical scenarios. ChituDiffusion is also able to achieve consistent speedup over baselines with different batch sizes (§7.6). 

## **7.5 dGraph Recomposition Analysis** 

To demonstrate dGraph decomposition, Table 3 reports redundancy optimization for SDXL U-Net. Without decomposition, the monolithic strategy produces numerous engines for all input property combinations, incurring high compilation cost. By symbolic property analysis, ChituDiffusion identifies inputs such as time embeddings and prompt conditions to form dGraphs, then optimizes them separately while pruning unsatisfiable cases, significantly reducing overhead. Fine-tuning extensions further exacerbate the problem. As 

**==> picture [242 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
60 Run time 3 5 Schedule 20<br>dEngine 4 GPU idle<br>40 2 3 15<br>2 10<br>20 1<br>1 5<br>0 0 0 0<br>0 5 10 15 0 5 10 15<br>Schedule window Size Schedule window Size<br>Time (ms) Time (ms)<br>GPU idle ratio (%)<br>Count of dEngine executions<br>**----- End of picture text -----**<br>


**Figure 14.** Performance of an edit request under different scheduling window sizes. 

**==> picture [242 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
Chitu-Diffusion-uniform Chitu-Diffusion-ragged Chitu-Diffusion<br>2.0<br>1.5<br>1.5<br>1.0<br>1.0<br>0.5 0.5<br>0.0 0.0<br>0 25 50 75 100 0 25 50 75 100<br>Raggedness ratio (%) Raggedness ratio (%)<br>(a) SD1.5 (b) SDXL<br>Normalized QPS<br>**----- End of picture text -----**<br>


**Figure 15.** Throughput under requests of different raggedness ratios on text-to-image diffusion pipelines. 

## **7.6 Scheduling Window Size and Overhead** 

Figure 14(a) shows that ChituDiffusion achieves higher performance with larger scheduling windows, as more dTasks are batched into fewer dEngine executions, exposing greater inter-request optimization. Figure 14(b) shows that scheduling cost is under 10% of dEngine runtime due to efficient dynamic programming, and further hidden with larger windows by improved batching and overlap, leaving less than 5% GPU idle time including cold start. 

## **7.7 Data-Aware Batching** 

**Table 3.** Compilation statistics for SDXL U-Net. The monolithic strategy treats U-Net as a whole with dGraph recomposition disabled. 

|Model|Strategy|# Inputs|# dGraphs|# Engines|Estimated<br>compilation time|
|---|---|---|---|---|---|
|SDXL UNet|Monolithic<br>dGraph|4<br>4|1 (N/A)<br>3|16<br>4|16 min<br>4 min|
|SDXL UNet|Monolithic|14|1 (N/A)|16384|11 d|
|w\ ControlNet|dGraph|14|4|7|7 min|



Table 3 shows, enabling ControlNet adds ten inputs to U-Net. The monolithic strategy must enumerate 2[14] input property combinations, leading to prohibitive compilation time. In contrast, ChituDiffusion leverages symbolic analysis to verify that ControlNet inputs share identical optimization conditions, treating them as one and exponentially reducing specialized dEngines, thus mitigating overhead. 

To show the effectiveness of data-aware batching, we create two baselines without it for ragged requests. ChituDiffusion-uniform only batches uniform-shape requests with uniform dEngines, while ChituDiffusion-ragged always batches all requests with ragged dEngines. Figure 15 shows the evaluation results. At raggedness ratio 0%, ChituDiffusion-uniform outperforms ChituDiffusion-ragged since ragged dEngines have more overhead than regular ones, such as index computation. As raggedness increases, ChituDiffusion-ragged achieves up to 1 _._ 5× higher throughput, benefiting from increased parallelism and reduced redundant memory access to model weights. 

The baselines are not always optimal. In contrast, ChituDiffusion uses data-aware scheduling to select appropriate dEngines, achieving the best performance across all raggedness ratios. For raggedness from 25% to 50%, ChituDiffusion combines regular and ragged dEngines, yielding up to 10% higher throughput. 

547 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

**Performance model.** We evaluate the model on SDXL U-Net dEngines with varied batch sizes between 1 to 16 and shapes between 256 to 768. By profiling 16 samples as inputs for the performance model, it achieves a 0.998 coefficient of determination. A 96-sample evaluation set generated in the same method yields a 0.996 coefficient and an RMSE less than 3 _𝜇𝑠_ , showing high prediction accuracy. As ChituDiffusion usually batches requests to fully utilize hardware, the performance model presents enough ability to estimate execution time for diffusion models, which are majorly composed of compute-intensive DNNs. 

## **8 Related Work** 

**Diffusion models and acceleration.** Significant advancements in image synthesis like Imagen [53] and Stable Diffusion [51], as well as in other domains [11, 27], appear since the denoising diffusion probabilistic model [29]. Optimizations such as distillation [55], quantization [39], and caching similar intermediate results [8, 36], and parallelism techniques [36, 37] are widely explored. Different from these nonequivalent approximate accelerations, ChituDiffusion focuses on equivalent optimizations and is promising to work with nonequivalent methods simultaneously. 

**Batching.** Efficient batching is widely studied [14, 41, 56, 56, 70]. VELTAIR [41] avoids resource interference in batch requests. DVABatch [19] flexibly reorganizes requests to process requests with the best batch sizes. Different from the above work, ChituDiffusion focuses on exploiting dynamic data properties, such as redundancy and raggedness, for diffusion models, which can be combined with these techniques. 

**Data-Property-Aware Optimization** Existing work has explored optimization techniques that leverage data properties at different granularities. Redundancy elimination has been extensively studied in general-purpose compilers [10, 15, 23, 45, 65], primarily for scalar computations. Modern DNN frameworks such as TensorFlow [7], PyTorch [48], TensorRT [58], Stable Fast [1], and TVM [13] provide performance optimizations for diffusion models but do not exploit fine-grained subrequest-level redundancies. Request-level caching approaches like Clipper [18] cannot handle partial redundancies, and KV-cache techniques [33, 68] designed for auto-regressive models are ineffective for diffusion models using bidirectional attention over full latent tensors. 

For ragged computations, specialized kernels for irregularshaped data have been developed, including optimized ragged matrix multiplication on GPUs [38] and CoRa, which extends TVM to support ragged Transformers [26]. These approaches focus on improving execution efficiency for inputs with dynamic or non-uniform shapes. 

## **9 Conclusion** 

We propose ChituDiffusion, a diffusion model serving system recomposing pipelines and requests to exploit data properties. By leveraging the locality of data properties, ChituDiffusion orchestrates compile-time and runtime techniques, ChituDiffusion outperforms existing frameworks by up to 2 _._ 19×. 

## **Acknowledgments** 

We would like to thank the anonymous reviewers and our shepherd Baolin Li for their insightful comments. We are grateful to Yuyang Chen and Chengyu Shi for their significant contributions to the implementation and maintenance of the ChituDiffusion system. This work is supported by the National Key R&D Program of China under Grant 2023YFB3002002, NSFC for Distinguished Young Scholar under Grant 62225206, National Natural Science Foundation of China under Grants 62532006, U23A6007, Beijing Natural Science Foundation under Grant L242017, and the Strategic Priority Research Program of Chinese Academy of Sciences under Grant XDB0500103. Jidong Zhai is the corresponding author of this paper. 

## **References** 

- [1] 2023. Stable Fast. https://github.com/chengzeyi/stable-fast. 

- [2] (Accessed on 05/06/2024). Adobe firefly. https://www.adobe.com/ products/firefly.html. 

- [3] (Accessed on 05/06/2024). ComfyUI community manual. https://blenderneko.github.io/ComfyUI-docs/Interface/Textprompts/ #adding-random-choices. 

- [4] (Accessed on 05/06/2024). Stable Diffusion Dynamic Prompts extension. https://github.com/adieyal/sd-dynamic-prompts/tree/main. 

- [5] (Accessed on 05/06/2024). Stable Diffusion WebUI documentation. https://github.com/AUTOMATIC1111/stable-diffusion-webui/ wiki/Features#prompts-from-file-or-textbox. 

- [6] (Accessed on 05/06/2024). Video generation models as world simulators. https://openai.com/index/video-generation-models-as-worldsimulators/. 

- [7] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. 2016. Tensorflow: A system for large-scale machine learning. In _12th USENIX symposium on operating systems design and implementation (OSDI 16)_ . 265–283. 

- [8] Shubham Agarwal, Subrata Mitra, Sarthak Chakraborty, Srikrishna Karanam, Koyel Mukherjee, and Shiv Kumar Saini. 2024. Approximate Caching for Efficiently Serving Text-to-Image Diffusion Models. In _21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)_ . USENIX Association, Santa Clara, CA, 1173–1189. https:// www.usenix.org/conference/nsdi24/presentation/agarwal-shubham 

- [9] Stability AI. [n. d.]. Stable Diffusion 3: Multimodal Diffusion with Transformer Architecture. Technical report published by Stability AI, March 2024. https://stability.ai/news/stable-diffusion-3-researchpaper. 

- [10] Joel Auslander, Matthai Philipose, Craig Chambers, Susan J Eggers, and Brian N Bershad. 1996. Fast, effective dynamic compilation. _ACM SIGPLAN Notices_ 31, 5 (1996), 149–159. 

- [11] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. 2023. Stable video diffusion: Scaling latent 

548 

ChituDiffusion: A Data-Characteristic-Aware Serving System for Diffusion Models 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

   - video diffusion models to large datasets. _arXiv preprint arXiv:2311.15127_ (2023). 

- [12] John Canny. 1986. A computational approach to edge detection. _IEEE Transactions on pattern analysis and machine intelligence_ 6 (1986), 679– 698. 

- [13] Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Q. Yan, Haichen Shen, Meghan Cowan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. 2018. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning. In _13th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2018, Carlsbad, CA, USA, October 8-10, 2018_ , Andrea C. ArpaciDusseau and Geoff Voelker (Eds.). USENIX Association, 578–594. 

- [14] Seungbeom Choi, Sunho Lee, Yeonjae Kim, Jongse Park, Youngjin Kwon, and Jaehyuk Huh. 2022. Serving heterogeneous machine learning models on Multi-GPU servers with Spatio-Temporal sharing. In _2022 USENIX Annual Technical Conference (USENIX ATC 22)_ . 199– 216. 

- [15] Fred Chow, Sun Chan, Robert Kennedy, Shin-Ming Liu, Raymond Lo, and Peng Tu. 1997. A new algorithm for partial redundancy elimination based on SSA form. _ACM Sigplan Notices_ 32, 5 (1997), 273–286. 

- [16] Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 2016. 3D U-Net: learning dense volumetric segmentation from sparse annotation. In _Medical Image Computing and Computer-Assisted Intervention–MICCAI 2016: 19th International Conference, Athens, Greece, October 17-21, 2016, Proceedings, Part II 19_ . Springer, 424–432. 

- [17] civitai 2022. Civitai. https://github.com/civitai/civitai. 

- [18] Daniel Crankshaw, Xin Wang, Guilio Zhou, Michael J Franklin, Joseph E Gonzalez, and Ion Stoica. 2017. Clipper: A Low-Latency online prediction serving system. In _14th USENIX Symposium on Networked Systems Design and Implementation (NSDI 17)_ . 613–627. 

- [19] Weihao Cui, Han Zhao, Quan Chen, Hao Wei, Zirui Li, Deze Zeng, Chao Li, and Minyi Guo. 2022. DVABatch: Diversity-aware MultiEntry Multi-Exit Batching for Efficient Processing of DNN Services on GPUs. In _2022 USENIX Annual Technical Conference_ . 183–198. 

- [20] DALL-e-3 2023. Improving Image Generation with Better Captions. https://cdn.openai.com/papers/dall-e-3.pdf. 

- [21] Tri Dao. 2023. Flashattention-2: Faster attention with better parallelism and work partitioning. _arXiv preprint arXiv:2307.08691_ (2023). 

- [22] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In _Advances in Neural Information Processing Systems_ . 

- [23] Yufei Ding and Xipeng Shen. 2017. Glore: Generalized loop redundancy elimination upon ler-notation. _Proceedings of the ACM on Programming Languages_ 1, OOPSLA (2017), 1–28. 

- [24] Chao Dong, Chen Change Loy, Kaiming He, and Xiaoou Tang. 2014. Learning a deep convolutional network for image super-resolution. In _European conference on computer vision_ . Springer, 184–199. 

- [25] Chao Dong, Chen Change Loy, and Xiaoou Tang. 2016. Accelerating the super-resolution convolutional neural network. In _European conference on computer vision_ . Springer, 391–407. 

- [26] Pratik Fegade, Tianqi Chen, Phillip Gibbons, and Todd Mowry. 2022. The CoRa tensor compiler: Compilation for ragged tensors with minimal padding. _Proceedings of Machine Learning and Systems_ 4 (2022), 721–747. 

- [27] Seth Forsgren and Hayk Martiros. 2022. Riffusion - Stable diffusion for real-time music generation. (2022). https://riffusion.com/about 

- [28] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. 2022. Imagen video: High definition video generation with diffusion models. _arXiv preprint arXiv:2210.02303_ (2022). 

- [29] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. _Advances in neural information processing systems_ 

   - 33 (2020), 6840–6851. 

- [30] Tobias Höppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, and Andrea Dittadi. 2022. Diffusion models for video prediction and infilling. _arXiv preprint arXiv:2206.07696_ (2022). 

- [31] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. _arXiv preprint arXiv:2106.09685_ (2021). 

- [32] Animesh Karnewar, Andrea Vedaldi, David Novotny, and Niloy J Mitra. 2023. Holodiffusion: Training a 3D diffusion model using 2D images. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ . 18423–18433. 

- [33] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles_ . 611–626. 

- [34] Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/ flux. 

- [35] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. 2025. FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv:2506.15742 [cs.GR] https://arxiv.org/abs/2506.15742 

- [36] Muyang Li, Tianle Cai, Jiaxin Cao, Qinsheng Zhang, Han Cai, Junjie Bai, Yangqing Jia, Ming-Yu Liu, Kai Li, and Song Han. 2024. DistriFusion: Distributed Parallel Inference for High-Resolution Diffusion Models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ . 

- [37] Suyi Li, Lingyun Yang, Xiaoxiao Jiang, Hanfeng Lu, Dakai An, Zhipeng Di, Weiyi Lu, Jiawei Chen, Kan Liu, Yinghao Yu, Tao Lan, Guodong Yang, Lin Qu, Liping Zhang, and Wei Wang. 2025. Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters. In _Proc. USENIX ATC_ . 

- [38] Xiuhong Li, Yun Liang, Shengen Yan, Liancheng Jia, and Yinghan Li. 2019. A coordinated tiling and batching framework for efficient GEMM on GPUs. In _Proceedings of the 24th symposium on principles and practice of parallel programming_ . 229–241. 

- [39] Xiuyu Li, Yijiang Liu, Long Lian, Huanrui Yang, Zhen Dong, Daniel Kang, Shanghang Zhang, and Kurt Keutzer. 2023. Q-diffusion: Quantizing diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 17535–17545. 

- [40] Zhimin Li, Jianwei Zhang, Qin Lin, Jiangfeng Xiong, Yanxin Long, Xinchi Deng, Yingfang Zhang, Xingchao Liu, Minbin Huang, Zedong Xiao, Dayou Chen, Jiajun He, Jiahao Li, Wenyue Li, Chen Zhang, Rongwei Quan, Jianxiang Lu, Jiabin Huang, Xiaoyan Yuan, Xiaoxiao Zheng, Yixuan Li, Jihong Zhang, Chao Zhang, Meng Chen, Jie Liu, Zheng Fang, Weiyan Wang, Jinbao Xue, Yangyu Tao, Jianchen Zhu, Kai Liu, Sihuan Lin, Yifu Sun, Yun Li, Dongdong Wang, Mingtao Chen, Zhichao Hu, Xiao Xiao, Yan Chen, Yuhong Liu, Wei Liu, Di Wang, Yong Yang, Jie Jiang, and Qinglin Lu. 2024. Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding. arXiv:2405.08748 [cs.CV] 

- [41] Zihan Liu, Jingwen Leng, Zhihui Zhang, Quan Chen, Chao Li, and Minyi Guo. 2022. VELTAIR: towards high-performance multi-tenant deep learning services via adaptive compilation and scheduling. In _Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ . 388–401. 

- [42] Xudong Lu, Aojun Zhou, Ziyi Lin, Qi Liu, Yuhui Xu, Renrui Zhang, Yafei Wen, Shuai Ren, Peng Gao, Junchi Yan, and Hongsheng Li. 2024. TerDiT: Ternary Diffusion Models with Transformers. arXiv:2405.14854 [cs.CV] 

549 

C. Wu, L. Zheng, H. Wang, K. Huang, Z. Ma, D. Dong, and J. Zhai 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

- [43] Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. 2021. Symbolic music generation with diffusion models. _arXiv preprint arXiv:2103.16091_ (2021). 

- [44] Shentong Mo, Enze Xie, Ruihang Chu, Lewei Yao, Lanqing Hong, Matthias Nießner, and Zhenguo Li. 2023. DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation. _arXiv preprint arXiv: 2307.01831_ (2023). 

- [45] Etienne Morel and Claude Renvoise. 1979. Global optimization by suppression of partial redundancies. _Commun. ACM_ 22, 2 (1979), 96–103. 

- [46] William Peebles and Saining Xie. 2022. Scalable Diffusion Models with Transformers. _arXiv preprint arXiv:2212.09748_ (2022). 

- [47] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2024. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. In _The Twelfth International Conference on Learning Representations_ . https://openreview.net/forum?id=di52zR8xgf 

- [48] PyTorch 2017. Tensors and Dynamic neural networks in Python with strong GPU acceleration. https://pytorch.org. 

- [49] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. 2023. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. _arXiv preprint arXiv:2306.17843_ (2023). 

- [50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020 [cs.CV] 

- [51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ . 10684–10695. 

- [52] Chitwan Saharia, William Chan, Huiwen Chang, Chris Lee, Jonathan Ho, Tim Salimans, David Fleet, and Mohammad Norouzi. 2022. Palette: Image-to-image diffusion models. In _ACM SIGGRAPH 2022 conference proceedings_ . 1–10. 

- [53] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. 2022. Photorealistic text-to-image diffusion models with deep language understanding. _Advances in Neural Information Processing Systems_ 35 (2022), 36479–36494. 

- [54] Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. 2022. Image super-resolution via iterative refinement. _IEEE Transactions on Pattern Analysis and Machine Intelligence_ 45, 4 (2022), 4713–4726. 

- [55] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. 2023. Adversarial diffusion distillation. _arXiv preprint arXiv:2311.17042_ (2023). 

- [56] Haichen Shen, Lequn Chen, Yuchen Jin, Liangyu Zhao, Bingyu Kong, Matthai Philipose, Arvind Krishnamurthy, and Ravi Sundaram. 2019. Nexus: A GPU cluster engine for accelerating DNN-based video analysis. In _Proceedings of the 27th ACM Symposium on Operating_ 

_Systems Principles_ . 322–337. 

- [57] Tencent Hunyuan Team. 2025. HunyuanImage 2.1: An Efficient Diffusion Model for High-Resolution (2K) Text-to-Image Generation. https://github.com/Tencent-Hunyuan/HunyuanImage-2.1. 

- [58] TensorRT 2017. NVIDIA TensorRT: Programmable Inference Accelerator. https://developer.nvidia.com/tensorrt. 

- [59] triton 2021. Introducing Triton: Open-source GPU programming for neural networks. https://openai.com/research/triton. 

- [60] Aravind Vasudevan, Andrew Anderson, and David Gregg. 2017. Parallel Multi Channel Convolution using General Matrix Multiplication. arXiv:1704.04428 [cs.CV] 

- [61] Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul, Mishig Davaadorj, and Thomas Wolf. 2022. Diffusers: State-of-the-art diffusion models. https://github.com/ huggingface/diffusers. 

- [62] Zijie J. Wang, Evan Montoya, David Munechika, Haoyang Yang, Benjamin Hoover, and Duen Horng Chau. 2022. DiffusionDB: A LargeScale Prompt Gallery Dataset for Text-to-Image Generative Models. _arXiv:2210.14896 [cs]_ (2022). https://arxiv.org/abs/2210.14896 

- [63] Joseph L. Watson, David Juergens, Nathaniel R. Bennett, Brian L. Trippe, Jason Yim, Helen E. Eisenach, Woody Ahern, Andrew J. Borst, Robert J. Ragotte, Lukas F. Milles, Basile I. M. Wicky, Nikita Hanikel, Samuel J. Pellock, Alexis Courbet, William Sheffler, Jue Wang, Preetham Venkatesh, Isaac Sappington, Susana Vázquez Torres, Anna Lauko, Valentin De Bortoli, Emile Mathieu, Regina Barzilay, Tommi S. Jaakkola, Frank DiMaio, Minkyung Baek, and David Baker. 2022. Broadly applicable and accurate protein design by integrating structure prediction networks and diffusion generative models. doi:10.1101/2022. 12.09.519842 Pages: 2022.12.09.519842 Section: New Results. 

- [64] Yutong Xie, Zhaoying Pan, Jinge Ma, Luo Jie, and Qiaozhu Mei. 2023. A prompt log analysis of text-to-image generation systems. In _Proceedings of the ACM Web Conference 2023_ . 3892–3902. 

- [65] Jingling Xue and Qiong Cai. 2006. A lifetime optimal algorithm for speculative PRE. _ACM Transactions on Architecture and Code Optimization (TACO)_ 3, 2 (2006), 115–155. 

- [66] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A distributed serving system for Transformer-Based generative models. In _16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)_ . 521–538. 

- [67] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ . 3836–3847. 

- [68] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. 2023. Efficiently Programming Large Language Models using SGLang. _arXiv preprint arXiv:2312.07104_ (2023). 

- [69] Changqian Yu Debang Li Jusnshi Huang Zhengcong Fei, Mingyuan Fan. 2024. Scaling Diffusion Transformers to 16 Billion Parameters. _arXiv preprint_ (2024). 

- [70] Zhe Zhou, Xuechao Wei, Jiejing Zhang, and Guangyu Sun. 2022. PetS: A Unified Framework for Parameter-Efficient Transformers Serving. In _2022 USENIX Annual Technical Conference (USENIX ATC 22)_ . 489–504. 

Received 2025-09-01; accepted 2025-11-10 

550 

