# **ELF: Embedded Language Flows** 

**Keya Hu[*] Linlu Qiu[*] Yiyang Lu Hanhong Zhao Tianhong Li Yoon Kim Jacob Andreas Kaiming He** 

MIT 

*Equal contribution; order decided by a coin flip. 

**Code:** `https://github.com/lillian039/ELF` 

## **Abstract** 

Diffusion and flow-based models have become the _de facto_ approaches for generating continuous data, _e.g._ , in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today’s leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that _continuous_ DLMs can be made effective with minimal adaptation to the discrete domain. We propose _Embedded Language Flows (ELF)_ , a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, _e.g._ , classifier-free guidance (CFG). Experiments show that ELF substantially outperforms leading discrete and continuous DLMs, achieving better generation quality with fewer sampling steps. These results suggest that ELF offers a promising path toward effective continuous DLMs. 

**==> picture [191 x 73] intentionally omitted <==**

**----- Start of picture text -----**<br>
1024 steps w/o distill<br>100<br>w/ distill<br>80 1024 steps<br>32 steps 1024 steps<br>60<br>32 steps 32 steps<br>40 1024 steps<br>32 steps<br>20<br>0<br>MDLM Duo FLM LangFlow ELF<br>Gen. PPL<br>**----- End of picture text -----**<br>


Figure 1: **ELF** achieves lower generative perplexity with fewer sampling steps than prior DLMs, without using distillation. ELF achieves this while using 10 _×_ fewer training tokens. (Model size: 105M for ELF and 170M for others; dataset: OWT. Detailed comparison in Fig. 7.) 

## **1 Introduction** 

Diffusion models [63, 64, 26] and flow-based models [37, 38, 3] have become prominent paradigms for generating continuous data, demonstrating strong performance at synthesizing images, videos, and data in other continuous domains. These advances have driven growing interest in extending diffusion methods to language modeling, leading to extensive work on diffusion language models (DLMs). DLMs are commonly formulated in one of two ways: continuous or discrete. Continuous DLMs map discrete tokens into continuous representations and perform denoising in the resulting continuous space [34, 13, 19]. Discrete DLMs, in contrast, operate directly in token space and formulate a probabilistic diffusion model over discrete random variables [5, 23, 40, 56, 57]. Recent progress in DLMs has been mostly in the discrete regime, in large part due to the stronger empirical performance of discrete DLMs [33, 48, 76, 58]. But it remains an open question whether the current performance gap of continuous DLMs is due to the inherently discrete nature of language modeling or to underexplored algorithmic design choices. 

In this work, we introduce Embedded Language Flows ( **ELF** ), a class of continuous DLMs based on Flow Matching [37, 38, 3]. ELF is continuous in two senses. First, it operates in _continuous_ 

Preprint. 

**==> picture [396 x 83] intentionally omitted <==**

**----- Start of picture text -----**<br>
t = 0.00 t = 0.25 t = 0.50 t = 0.99 t = 1.00<br>cat<br>dog bird<br>**----- End of picture text -----**<br>


Figure 2: **Conceptual illustration of ELF.** Orange points denote data represented in continuous embedding space, and purple lines show denoising trajectories from Gaussian noise to clean embeddings. Discretization is applied only at the final time step ( _t_ = 1) using a shared-weight network. 

_embedding_ space by directly denoising continuous representations throughout the flowing process, with discretization considered only at the final time step. Second, it is formulated with _continuous time_ , following Flow Matching [37, 38, 3], which allows us to define the velocity field via the time derivative. This formulation enables ELF to benefit from advances in Flow Matching, which is now widely used to instantiate diffusion models in image and video generation [43, 14, 6, 70]. 

Following Latent Diffusion Models (LDM) [54], ELF constructs the continuous embedding space by applying an encoder model to the input discrete tokens. The encoder can be pretrained, jointly trained, or frozen with random weights. _Unlike_ latent diffusion, ELF does not require a separate decoder and thus introduces no additional component at inference time. This design is based on the observation that the final time step in Flow Matching can be naturally repurposed to map continuous embeddings back to discrete tokens, eliminating the need for an explicit decoder. As such, a shared-weight network is trained to perform denoising at all but the final step, and decoding (i.e. discretization) at the final step (see Fig. 2). 

ELF builds on prior continuous DLMs, but aims for a minimalist design that addresses the interface between continuous and discrete spaces. In contrast to pioneering works on continuous DLMs [34, 13, 19] and many others that employ a per-step discretization loss ( _e.g._ , cross-entropy), ELF performs denoising in continuous embedding space at nearly all steps, thereby offering maximal flexibility for the flow dynamics. And unlike latent diffusion methods [41, 45, 62], which typically operate in a _compressed_ latent space and rely on a separate decoder, ELF directly operates in a high-dimensional latent space [32] and requires no extra decoder. 

Empirically, we show that ELF outperforms leading methods on discrete DLMs and existing continuous DLMs (Fig. 1), following the evaluation protocols established in those works. ELF achieves better generation quality with fewer sampling steps than leading discrete DLMs ( _e.g._ , MDLM [56] and Duo [57]) and concurrent continuous DLMs ( _e.g._ , FLM [30] and LangFlow [10]). Moreover, ELF achieves this performance using 10 _× fewer_ training tokens and _without_ any distillation. We further show that ELF performs strongly on machine translation [7] and summarization [46]. Overall, these results suggest that continuous DLMs can be highly competitive while requiring only minimal treatment of discretization, offering a promising direction for diffusion-based language modeling. 

## **2 Background & Related Work** 

**Diffusion-/Flow-based models.** Diffusion models [63, 26, 64] and flow-based models [37, 38, 2] transform noise into data through ordinary or stochastic differential equations (ODEs/SDEs). In DDPM-style formulations, generation is defined by transitions between successive states [63, 26, 47], which may be discrete or continuous. Discrete states require categorical transition distributions, as in discrete DLMs [5, 56]; continuous states are commonly modeled through score or noise prediction under Gaussian corruption [64, 26, 14]. Flow Matching extends this view to continuous time by learning the velocity field along a continuous path [37, 38, 2], where noise, data, and velocity predictions can be reparameterized into one another [14, 32]. Our method adopts Flow Matching to formulate language generation in continuous embedding space and continuous time. 

**Continuous diffusion language models.** Continuous DLMs map discrete tokens to a continuous space to perform denoising. _Embedding-space_ methods, such as Diffusion-LM [34], CDCD [13], and DiffuSeq [19], add Gaussian noise directly to token embeddings [66, 79, 21, 72, 77, 36, 74, 15]. A complementary direction studies _simplex-based_ representations, including SSD-LM [22] and TESS [44, 68], as well as related manifold-based formulations [27]. Although these methods provide 

2 

**==> picture [358 x 96] intentionally omitted <==**

**----- Start of picture text -----**<br>
input tokens � �� �<br>Training ��....12 embed .... corrupt .... network ELF  .... or<br>��<br>�� ��+1 output tokens<br>Sampling ( gaussian )�0 .... denoiser ELF .... (t=1) ELF unembed ��....�12�<br>**----- End of picture text -----**<br>


Figure 3: **During training** , discrete tokens are encoded into clean embeddings _**x**_ and corrupted to _**z** t_ , which ELF uses to predict _**x**_ ˆ. The model is trained with either the denoising loss _L_ MSE or the token-wise cross-entropy loss _L_ CE. **During inference** , ELF starts from Gaussian noise _**z**_ 0 and iteratively denoises embeddings from _**z** t_ to _**z** t_ +1. Only at the final step does ELF switch to decoding mode and project the final embeddings back to discrete tokens through an unembedding layer. 

continuous relaxations of discrete tokens, their trajectories often remain tied to the discrete token space through mechanisms such as rounding losses, simplex constraints, and token-level cross-entropy objectives. In contrast, ELF denoises entirely in continuous embedding space without per-step token-level supervision and discretizes only at the final step. 

Another line applies _latent diffusion_ to frozen encoder representations, represented by LD4LG [41] and follow-up work [81, 59, 42, 45, 62]. Like many diffusion methods described above, these approaches typically follow DDPM-style or score-based formulations with DDPM noise schedules [26, 47], and additionally rely on a separately trained decoder to recover tokens. In contrast, ELF uses a continuous-time Flow Matching formulation with a linear (rectified-flow) interpolant [37, 38, 2], and does not require a separate decoder. This brings flow-based training and sampling into language diffusion, allowing ELF to benefit from recent advances in Flow Matching. 

Several concurrent works also revisit continuous flow-based language modeling. DFM [51], CFM [55], FLM/FMLM [30], and LangFlow [10] all incorporate token-level cross-entropy supervision along the flow trajectory, though they differ in the continuous state space, including simplex space, one-hot token encodings, and embedding space. Some of these methods further introduce distillation for few-step generation, such as distilled DFM/CFM and FMLM. In contrast, ELF keeps the denoising trajectory entirely in an unrestricted continuous embedding space, applying token-level supervision only at the final decoding step. A more comprehensive survey is provided in Appendix A. 

**Discrete diffusion language models.** Due to the discrete nature of language, another line of work applies diffusion directly in token space. D3PMs [5] define general discrete corruption processes, including absorbing and uniform transitions. Masked diffusion models, such as MDLMs [56], use a special `[MASK]` absorbing state and generate samples through iterative unmasking [23, 48, 76]. Subsequent work improves sampling and efficiency through remasking, adaptive inference [71, 73], and semi-autoregressive block diffusion, including E2D2 [4]. Uniform-state diffusion models, such as Duo [57], instead diffuse tokens toward a uniform categorical distribution, enabling repeated token revision during inference [57, 12, 58]. Recent studies further scale discrete DLMs and extend them to code and multimodal generation [20, 65, 75, 78, 31]. Overall, discrete diffusion models currently remain the dominant paradigm in diffusion-based language modeling [33]. 

## **3 Embedded Language Flows** 

In this section, we present our flow-based formulation for language modeling (Fig. 3). Our method leverages the iterative nature of flow models to perform denoising primarily in continuous embedding space, converting clean embeddings back to discrete tokens only at the final step. Following prior work [56, 57, 30, 10], we describe our method in the simpler setting of unconditional generation. The framework can be extended to conditional generation, as discussed in Sec. 3.3. 

## **3.1 The ELF Framework** 

**From discrete tokens to continuous embeddings.** To apply continuous diffusion to language, we first map discrete tokens to continuous representations. Given a sentence, we tokenize it into a sequence of tokens _**s**_ = [ _s_ 1 _, . . . , sL_ ] _∈ V[L]_ , where each _si_ is drawn from the vocabulary _V_ 

3 

and _L_ denotes the sequence length. We then map the discrete token sequence into a continuous embedding space. The choice of the embedding method is flexible. By default, we use a pretrained T5 encoder [53] for bidirectional contextual embeddings. We also explore other jointly trained and randomized embeddings (see Sec. 4.1). The encoder is only used during training, which does not incur additional modules at inference. 

**Flow Matching on continuous embeddings.** After obtaining continuous language representations, we formulate the denoising process in the resulting embedding space using Flow Matching [37, 38, 3]. Flow Matching defines a continuous flow path from noise to data in this space. Let _**x** ∼ p_ data( _**x**_ ) denote the embedding distribution and _**ϵ** ∼ p_ noise( _**ϵ**_ ) denote the noise distribution ( _e.g._ , _**ϵ** ∼N_ (0 _,_ **I** )). The noisy latent variable is defined by linear interpolation (“rectified flows”): _**z** t_ = _t_ _**x**_ + (1 _− t_ ) _**ϵ**_ , where _t ∈_ [0 _,_ 1], and _**z**_ 0 _∼ p_ noise and _**z**_ 1 _∼ p_ data. In continuous time, the flow velocity _**v**_ is defined as the time derivative of _**z**_ , that is, _**v**_ = _d_ _**z** /dt_ = _**x** −_ _**ϵ** ._ 

While standard Flow Matching directly parameterizes _**v**_ via a neural network, ELF follows recent advances on image generation and instead parameterizes _**x**_ [32] ( _**x**_ **-prediction** ). Specifically, let _**x** θ_ = `net` _θ_ ( _**z** t, t_ ) denote the network’s immediate output. We train the model by minimizing the mean squared error (MSE) between the predicted velocity and the target velocity: 

**==> picture [335 x 23] intentionally omitted <==**

where we leverage the relation _**v**_ ( _**z** t, t_ ) = ( _**x** −_ _**z** t_ ) _/_ (1 _− t_ ) [32]. 

The _**x**_ -prediction parameterization is important for ELF. First, it enables Flow Matching to perform effectively on high-dimensional representations ( _e.g._ , 768-d per-token embeddings), consistent with observations in [32] (see Appendix C.1 for ELF’s ablations on prediction targets). Second, predicting clean embeddings ( _i.e._ , _**x**_ ) aligns naturally with the objective of predicting clean discrete tokens at the final step (discussed next), whereas the standard _**v**_ -prediction in Flow Matching does not. Although _**v**_ can be predicted by a network and transformed into _**x**_ , the weight sharing that ties the denoising (MSE loss) and decoding (cross-entropy loss) objectives is compromised. Empirically, we observe that _**v**_ -prediction works poorly when weights are shared with the final discretization step. 

**Back to discrete tokens.** As the generation output consists of discrete tokens, we convert the clean embeddings back into tokens at the final time step ( _i.e._ , at _t_ = 1). By considering the final time step of ELF naturally as continuous-to-discrete decoding, our method does not require a separate decoder (or equivalently, it can be thought of as a decoder sharing weights with the denoiser). 

The network input at this time step should be _**z** t_ in the limit _t →_ 1. But because _**z** t →_ _**x**_ as _t →_ 1, we introduce a token-level corruption process at this final step to create a nontrivial training input,denoted as _**z**_ ˜ (detailed in Appendix B.1). The same network `net` _θ_ maps _**z**_ ˜ to a clean embedding _**x** θ_ (˜ _**z**_ ), which is subsequently projected by a learnable “unembedding” matrix _W_ to obtain logits. We minimize a per-token cross-entropy (CE) loss against the ground-truth token _**s**_ : 

**==> picture [271 x 11] intentionally omitted <==**

The network _**x** θ_ shares weights with that in Eq. (1) and is conditioned on a binary “mode” token (denoise or decode) in addition to the time condition _t_ = 1. At inference time, we evaluate _W_ _**x** θ_ ( _**z** t_ ) only at the final step _t_ = 1, and apply argmax to obtain a discrete token. 

## **3.2 Pseudocode** 

The core concepts of ELF are summarized in Alg. 1 and Alg. 2 (detailed in Appendix Fig. 9). 

**Training.** As in standard Flow Matching, ELF employs a single network `net` _θ_ to model all time steps, conditioned on _t_ . This includes the final time step _t_ = 1, which uses different pre-processing (corruption) and post-processing (loss computation). For clarity, we illustrate this distinction using an explicit “ `if` ” branch in Alg. 1. In practice, samples from both branches are processed within a _single_ batch, and masking is used to selectively apply the appropriate corruption and unembedding operations as well as the corresponding loss terms. The network is further conditioned on a binary “mode” token that indicates whether the operation is “denoise” or “decode”. 

**Inference.** During inference, ELF iteratively transforms noisy samples into clean embeddings. Starting from _**z**_ 0 _∼N_ (0 _,_ **I** ), ELF solves the ODE: _d_ _**z** t/dt_ = _**v** θ_ ( _**z** t, t_ ), which is approximated with 

4 

## **Algorithm 1** ELF: training. 

Two-branch computation is batched, adding no extra training cost. 

## **Algorithm 2** ELF: inference. 

We show ODE for simplicity. SDE sampler is also applicable. 

```
#net(z,t,mode):ELFnetwork#shape:shapeofembeddedsequences
#s:asequenceofdiscretetokens#ts:samplingtimeschedule,from0to1
x=encode(s)z=randn(shape)
ifuniform(0,1)<threshold:foriinrange(len(ts)-1):
#denoisingbrancht=ts[i]
t=sample_t()dt=ts[i+1]-ts[i]
e=randn_like(x)x_pred=net(z,t,mode="denoise")
z=t*x+(1-t)*e
v=x-e#convertxpredictiontovelocity
x_pred=net(z,t,mode="denoise")v=(x_pred-z)/(1-t)
v_pred=(x_pred-z)/(1-t)z=z+dt*v
loss=mse_loss(v_pred,v)
else:#finalstep
#decodingbranch(t=1)h=net(z,t=1,mode="decode")
z=corrupt(x)
x_pred=net(z,t=1,mode="decode")#unembedding
s_pred=unembed(x_pred)token_logits=unembed(h)
loss=ce_loss(s_pred,s)tokens=argmax(token_logits)
```

a numerical ( _e.g._ , Euler) solver. At the final time step _t_ = 1, we apply the network under the “decode” mode and perform unembedding and discretization. 

Besides the ODE formulation, our method also supports an SDE-inspired sampler. The underlying SDE associated with Flow Matching can be derived following [43], where the dynamics can be interpreted as injecting infinitesimal noise at each step. In practice, we adopt a simpler approximation to emulate this behavior: we inject small noise at each step while correspondingly shifting the time variable _t_ toward the noise regime (detailed in Appendix, Alg. 6). For brevity, we refer to the resulting SDE-inspired sampler as the “SDE” variant, while noting that it primarily captures the per-step stochastic behavior. We experimentally compare the ODE formulation with this SDE variant. 

## **3.3 Conditioning and Guidance** 

Controlling model generation is an important aspect of generative modeling. In image diffusion models, classifier-free guidance (CFG) [25] has been established as a highly effective technique for steering the generated output.[1] CFG also enables a trade-off between generation quality and diversity. Because CFG was originally formulated for continuous quantities ( _e.g._ , score functions or velocity fields), it is naturally applicable to ELF. This stands in contrast to discrete counterparts, where CFG remains largely unexplored and has been shown less effective [30, 51]. 

In the absence of class labels, we employ _self-conditioning_ [9] to construct the conditioning signals required for CFG. Given that self-conditioning is already a standard component in DLMs [79, 13, 66, 41, 44, 59, 60], incorporating CFG introduces only marginal computational overhead. In what follows, we first describe the self-conditioning used in ELF and then introduce CFG. 

**Self-conditioning.** In a standard Flow Matching model ( _i.e._ , without self-conditioning), a forward pass at a given time step yields a single prediction. We denote this prediction by _**x**_ ˆ _[′]_ in our case, indicating that it corresponds to a prediction of the clean embedding _**x**_ . During training, selfconditioning [9] performs a second forward pass, conditioned on _**x**_ ˆ _[′]_ , which serves as an intermediate ˆ ˆ ˆ prediction. The output of the second pass, denoted as _**x**_ , can be written as _**x**_ = `net` _θ_ ( _**z** t |_ _**x**[′] , t_ ). This is implemented by concatenatingconditioned on _**x**_ ˆ _[′]_ with probability 50%, and uses a null condition [ _**z** t,_ ˆ _**x**[′]_ ] as the network input [ **0** otherwise (see Appendix, Fig.9]. During training, the model is 9 for details). During inference, the model conditions on the prediction from the previous time step, thus introducing no extra forward passes for inference. 

The intermediate prediction _**x**_ ˆ _[′]_ serves as a condition for the network. As such, it can be treated as the conditioning signal _**c**_ in the application of CFG, introduced next. 

> 1CFG was historically introduced for _class_ -conditional generation. However, the notion of a condition can be generalized to other inputs, _e.g._ , a text prompt. We use CFG in this broader sense, as our setting does not involve class labels. 

5 

**CFG with self-conditioning.** CFG [25] combines the unconditional and conditional predictions through a linear extrapolation. Formally, given a conditioning signal _**c**_ , CFG in Flow Matching defines a velocity field as _**v**_ cfg( _**z** t |_ _**c**_ ) = _ω_ _**v**_ ( _**z** t |_ _**c**_ ) + (1 _− ω_ ) _**v**_ ( _**z** t |_ ∅), where ∅ denotes the unconditional counterpart and _ω_ is the guidance scale. As discussed, our conditioning signal _**c**_ is obtained from self-conditioning. In its original form [25], CFG is applied at inference time, requiring two forward passes per step. 

To avoid inference-time overhead, we adopt _training-time_ CFG techniques [8, 69, 16, 17] previously developed for image generation. These methods use a single network pass to model _**v**_ cfg instead of _**v**_ (in our case, _**x**_ cfg instead of _**x**_ ). Because ELF is formulated similarly to its image-generation counterpart, adapting it to training-time CFG is straightforward, further illustrating the advantages of our continuous-based formulation. The implementation details, following the form in [16, 17], are in Appendix (Alg. 3, 4, & 5). 

**Extension to conditional generation.** Thus far, we have presented our method in the setting of unconditional generation, as in prior work [56, 57, 30, 10]. Our method can be naturally extended to conditional generation, in which outputs are conditioned on an input sequence ( _e.g._ , a prompt). In this setting, we prepend the clean embeddings of the conditioning sequence to the model input and preserve them without corruption during both training and inference. The model can then condition on them through self-attention. 

CFG remains applicable in the conditional setting. The conditioning _**c**_ now consists of both the self-conditioning and the prefix clean embeddings; the unconditional counterpart is obtained by zeroing out _**c**_ . Analogous to text-to-image generation [14], CFG is effective in controlling generation quality in our scenario, which can be viewed as “text-to-text” generation. 

## **4 Experiments** 

**Dataset and evaluation.** For unconditional generation, we follow the experimental design used in past work [56, 57, 30, 10]. We train on the OpenWebText (OWT) dataset [18], which has around 9B tokens, and pack sequences to length _L_ = 1024. For evaluation, we generate 1,000 samples and report generative perplexity (Gen. PPL), _i.e._ , the perplexity of generated samples under a pretrained GPT-2 Large model [52]; together with average unigram entropy as a measure of sample diversity.[2] For conditional generation, we consider machine translation and summarization. For machine translation, we use the WMT14 German-to-English (De-En) dataset [7] with sequence length _L_ = 128 (condition length 64, target length 64; 144M total target tokens), and evaluate using BLEU [49]. For summarization, we use the XSum dataset [46] with sequence length _L_ = 1088 (condition length 1024, target length 64; 6M total target tokens), and report ROUGE-1 (R1), ROUGE-2 (R2), and ROUGE-L (R-L) [35]. We treat both as sequence-to-sequence tasks and do not use sequence packing for conditional generation. 

**Model.** We use contextual embeddings from a frozen pretrained T5-small encoder [53] (35M) with embedding dimension 512. We use a bottleneck design that linearly projects embeddings into a lower-dimensional space of size 128, and then projects them back to the hidden size of the model [32]. We consider three model sizes: ELF-B (105M), ELF-M (342M), and ELF-L (652M), and use ELF-B as the default for ablations. Detailed configurations are shown in Appendix Tab. 3. 

**Training and inference.** We train our model using the Muon optimizer [28] with a learning rate of 0 _._ 002 and a batch size of 512. The model is trained for 5 epochs on OWT (around 95K steps), and for 100 epochs on WMT14 and XSum (around 880K and 40K steps, respectively). Depending on the selected model mode, the network is trained with either the MSE loss in Eq. 1 (80%) or the CE loss in Eq. 2 (20%). During inference, we use the ODE or SDE sampler to generate samples. 

## **4.1 Ablations** 

We begin by ablating several key design choices of our model on the simpler setting of unconditional generation on OWT, using the default ELF-B model and a 64-step ODE Euler sampler unless otherwise specified. More ablation studies are shown in Appendix C. 

> 2We do not use validation perplexity, since likelihood evaluation for flow-based models can require additional likelihoodspecific training [1]. 

6 

**Classifier-free guidance (CFG).** Our flow-based continuous formulation is naturally compatible with CFG, a highly effective technique in standard diffusion models. Therefore, we first study the effect of the CFG scale. As shown in Fig. 4, increasing the CFG scale lowers generative perplexity but also reduces entropy, reflecting a quality–diversity trade-off. The preferred direction is toward the lower-right region of the plot, corresponding to lower generative perplexity and higher entropy. For most of the remaining ablations, we evaluate this quality–diversity trade-off by sweeping the CFG scale. Each point on the curve is computed from 1,000 generated samples at a specific CFG scale. 

**==> picture [143 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
CFG=0.5<br>100<br>80 CFG=1<br>60 CFG=1.5<br>CFG=2<br>40 CFG=2.5<br>CFG=3<br>Better<br>20<br>5.2 5.3 5.4 5.5<br>Entropy<br>Gen. PPL<br>**----- End of picture text -----**<br>


Figure 4: **Ablations on guidance.** We evaluate the generative perplexity–entropy trade-off across CFG scales: increasing the scale lowers generative perplexity but reduces entropy. 

**==> picture [396 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
Learnable emb. layer Separate denoiser & decoder 250 SDE<br>140 Gaussian emb. layer Shared-weight denoiser & decoder ODE<br>120 Pretrained emb. layer Scratch encoder 100 200 H=5.6 Dataset<br>Pretrained encoder<br>100 80 150<br>H=5.5<br>80 H=5.5<br>60 100<br>60 H=5.4 H=5.4<br>40 Better 40 Better 50 H=5.4 H=5.3 H=5.2 H=5.3H=5.1 H=5.2 H=5.2 H=5.1<br>5.0 5.1 5.2 5.3 5.4 5.5 5.2 5.3 5.4 5.5 8 16 32 64 128 256 512 1024<br>Entropy  Entropy  Sampling Steps<br>(a) (b) (c)<br>Gen. PPL  Gen. PPL  Gen. PPL<br>**----- End of picture text -----**<br>


Figure 5: **Ablations on key design choices.** (a) Embedding choices: we compare contextual _vs._ noncontextual embeddings, as well as frozen _vs._ learnable embeddings; pretrained contextual embeddings achieve the best trade-off. (b) Decoding strategies: We compare a shared-weight denoiser-decoder with a two-stage, separately trained decoder. Both strategies achieve similar trade-offs, but the shared-weight variant extends further toward the regime of low generative perplexity. (c) Samplers: we compare ODE and SDE-inspired samplers across different sampling steps; SDE-inspired sampler consistently achieves lower generative perplexity in fewer steps. 

**Embedding choices.** Since ELF operates in a continuous embedding space, we next study how the choice of embeddings affects performance. We ablate the continuous embeddings along two axes: whether the embeddings are contextual ( _i.e._ , from an encoder) or non-contextual ( _i.e._ , from a single embedding layer), and whether they are fixed or learnable. For contextual embeddings, we evaluate those from an off-the-shelf T5 encoder [53] and embeddings from an encoder trained from scratch on OWT using the original T5 objective. For non-contextual embeddings, we consider token embeddings from the pretrained T5 model, frozen Gaussian embeddings, and learnable embeddings. See Appendix D.3 for detailed setup. We show the results in Fig. 5a. Contextual embeddings achieve a better generative perplexity–entropy trade-off. Embeddings from an encoder trained from scratch on OWT perform well, but slightly lag behind those from a pretrained encoder. Among the noncontextual variants, pretrained token embeddings outperform frozen Gaussian embeddings. Learnable embeddings perform the worst, likely due to the difficulty of jointly optimizing the embeddings and the denoiser. Overall, these results suggest that _pretrained contextual embeddings_ are favorable representations of language for ELF. 

**Decoding strategies.** Since we use contextual embeddings as our continuous representations, we need to decode them back into discrete tokens. We use a shared-weight network, with training interleaving _L_ MSE and _L_ CE. Alternatively, we explore a two-stage strategy. In the first stage, we train a decoder from scratch with a frozen pretrained T5 encoder to reconstruct tokens from masked and noisy embeddings using _L_ CE. In the second stage, we freeze both the encoder and decoder, and train a separate denoiser using _L_ MSE (see Appendix D.3 for details). As shown in Fig. 5b, both strategies achieve a similar trade-off, but the shared-weight variant extends further toward the regime of low generative perplexity , while also simplifying the pipeline by avoiding an extra training stage. 

**Samplers.** Since ELF is formulated in continuous time and continuous space, it naturally supports both deterministic ODE sampling and stochastic SDE-like sampling; see Appendix Alg. 6 for details. We compare ODE and SDE samplers across different sampling budgets with a self-conditioning CFG scale of 1. As shown in Fig. 5c, SDE sampling achieves substantially lower generative perplexity than 

7 

ODE sampling in the few-step regime. These results suggest that introducing stochasticity during sampling can effectively reduce error accumulation and provide a better quality–efficiency trade-off. 

**Model scales.** We study the scaling behavior of ELF across three model sizes: **ELF-B** (105M), **ELF-M** (342M), and **ELF-L** (652M) (detailed in Appendix Tab. 3). We evaluate each model using both ODE and SDE sampling. As shown in Fig. 6, scaling consistently improves the generative perplexity–entropy frontier. In particular, at matched entropy, larger models achieve lower generative perplexity, indicating higher sample quality with comparable diversity. Conversely, at similar generative perplexity, larger models maintain higher entropy. The effect of the sampler is consistent across model sizes: SDE sampling improves over ODE sampling by pushing the frontier in a more optimal direction. These results suggest that ELF scales effectively, demonstrating the potential of model scaling. See Appendix Tab. 7 for the detailed numbers. 

**==> picture [144 x 117] intentionally omitted <==**

**----- Start of picture text -----**<br>
100 ELF-B<br>ELF-M<br>ELF-L<br>ODE<br>80<br>SDE<br>Dataset<br>60<br>40<br>20 Better<br>5.1 5.2 5.3 5.4 5.5<br>Entropy<br>Gen. PPL<br>**----- End of picture text -----**<br>


Figure 6: **Scaling of ELF models.** We compare ELF-B, ELF-M, and ELFL. Scaling model size consistently improves the Gen. PPL–entropy frontier. 

**==> picture [396 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
500 MDLM 200 MDLM + SDTT Base training<br>Duo Duo + DCD 700 Extra distillation<br>FDLM 175 FMLM<br>400 ELF Dataset 150 ELFDataset 600 550B  (12x) 550B  (12x) 577B  (13x)<br>500<br>300 125<br>400<br>100<br>200 300<br>H=5.6 75 H=5.5<br>100 H=5.3 H=5.5 50 H=5.3 H=5.4 200<br>H=5.4 H=5.2 H=5.2 H=5.3 25 H=5.4 H=5.2 H=5.2 100 45B  (1x)<br>0 8 16 32 64 128 256 512 1024 8 16 32 0 ELF MDLM Duo FLM<br>Sampling Steps Sampling Steps (+ SDTT) (+ DCD) (FMLM)<br>(a) (b) (c)<br>Gen. PPL  Gen. PPL<br>Training Tokens (B)<br>**----- End of picture text -----**<br>


Figure 7: **System-level comparison.** ELF-B outperforms both discrete and continuous DLMs trained under similar settings (a), rivals distilled variants of other baselines that require additional rounds of training (b), and uses substantially fewer training tokens (c). 

## **4.2 System-Level Comparison on Unconditional Generation** 

We first compare ELF-B against both discrete DLMs, including MDLM [56] and Duo [57], and continuous DLMs, including FLM [30] and LangFlow [10], under a comparable setting. All models are trained on the OWT dataset. ELF has 105M parameters, while the compared baselines have around 170M parameters. For ELF, we use our best configuration: SDE sampling with self-conditioning CFG scale of 3 (see Appendix D.2 for details). We show results in Fig. 7a. ELF achieves a generative perplexity of 24 using only 32 sampling steps, requiring substantially less inference-time compute than prior methods. ELF remains strong even compared with distilled models, which require extra training to distill a student model for few-step generation. As shown in Fig. 7b, in the few-step regime, ELF outperforms distilled models, including MDLM+SDTT [56, 11], Duo+DCD [57], and FMLM [30], even without any additional distillation. 

ELF is also substantially more data-efficient in terms of estimated training tokens, as shown in Fig. 7c. While prior DLMs typically use over 500B tokens, ELF uses only 45B.[3] Together, these results show that, when combined with proper sampling and guidance, ELF achieves strong system-level performance. It not only improves inference efficiency, but also achieves strong performance with a much smaller training budget, demonstrating the potential of our flow-based language model. See Fig. 8 for qualitative examples of ELF-B’s generations. 

## **4.3 System-Level Comparison on Conditional Generation** 

We compare ELF-B with autoregressive and diffusion-based baselines at a similar model scale. These include discrete DLMs (MDLM [56], Duo [57], and E2D2 [4]) and continuous DLMs (SeqDif- 

> 3A per-method breakdown of training token counts is provided in Appendix Tab. 5. We also experimented with training on more tokens, but did not observe further performance improvement. 

8 

|**Model**|**Size**|**De-En**_†_<br>BLEU_↑_|ROUGE-1_↑_|**XSum**_‡_<br>ROUGE-2_↑_|ROUGE-L_↑_|
|---|---|---|---|---|---|
|AR|99M|25.2|30.5 _±_0.13|10.2 _±_0.11|24.4 _±_0.12|
|MDLM [56]|99M|18.4|33.4 _±_0.11|11.6 _±_0.10|25.8 _±_0.10|
|Duo [57]|170M (+35M)|21.3_‡_|31.4 _±_0.12|10.1 _±_0.10|25.0 _±_0.12|
|E2D2 [4]|99M|24.8|28.4 _±_0.11|8.3 _±_0.09|22.0 _±_0.10|
|SeqDiffuSeq [79]|-|21.3|19.3_†_|1.7_†_|14.1_†_|
|CDCD [13]|-|24.9|-|-|-|
|Ours|105M (+35M)|**26.4**|**36.0**_±_0.13|**12.2**_±_0.11|**27.8**_±_0.12|



Table 1: **Results on machine translation and summarization.** We evaluate ELF-B on WMT14 German-to-English (De-En) translation and XSum summarization, comparing against baselines of similar parameter scale. _[†]_ denotes results taken directly from prior work and is the default source for De-En, while _[‡]_ denotes results we reproduced using public codebases and is the default source for XSum. For XSum, we additionally report the standard error across evaluation examples when available. ELF achieves the best performance on both settings. 

**==> picture [12 x 12] intentionally omitted <==**

## **Unconditional Generation** 

**==> picture [54 x 5] intentionally omitted <==**

**----- Start of picture text -----**<br>
Gen. PPL : 20.9        H : 5.2<br>**----- End of picture text -----**<br>


....But I still haven't found a book that I don't think would be relevant to write. Maybe I could even do a very small survey and develop a book based on my experience. Is there anything to this that you would say that's relevant to your work?.....Jesse Leren is a 30-year-old journalist and student writer. She was born in Sydney and is interested in becoming a member of the writing community. She is writing around the world for students in Australia and abroad....... 

**==> picture [12 x 12] intentionally omitted <==**

## **Translation** 

**BLEU** : 71.2. 

**==> picture [213 x 22] intentionally omitted <==**

**----- Start of picture text -----**<br>
Context:       Es zeigt einen Ring aus schwarzen Löchern, 430 Millionen Lichtjahre von der Erde entfernt.<br>Reference:   It shows a ring of black holes, 430 million light years away from the Earth.<br>Generated:   It shows a ring of black holes, 430 million lights years from the Earth.<br>**----- End of picture text -----**<br>


**Summarization ROUGE-1:** 66.7 **ROUGE-2:** 38.7 **ROUGE-L** : 60.6 **Context:** Joe Cardle slotted in his eighth of the season before setting up Kallum Higginbotham to extend the Pars' lead. Dumbarton's Craig Barr hit the crossbar with a secondhalf header but the Sons then had Andy Dowie sent off for two bookings. Substitutes Nicky Clark and Andy Ryan added late goals to add gloss to the visitors' victory. That is four wins on the bounce for Allan Johnston's men and on this form Dunfermline appear to be more than capable of sustaining a promotion challenge. Of the six players playing in.... **Reference:** Dunfermline climbed to the top of the Championship with an emphatic victory at Dumbarton. **Generated:** Dunfermline climbed off the bottom of the Championship with a comfortable victory at Dumbarton to move within seven points. 

Figure 8: **Qualitative examples** of text generated by ELF-B. We show an unconditional sample, a German-to-English translation example, and a summarization example, along with their automatic evaluation metrics. Some text is omitted due to space limits; see Appendix E for more examples. 

fuSeq [79] and CDCD [13]). Some results are taken from the literature and others are reproduced from public codebases. See Appendix Tab. 8 for a summary. We use the best sampling configuration selected on the validation set: a 64-step ODE sampler with the self-conditioning CFG scale set to 1 and the input-condition CFG scale set to 2. 

We show the results in Tab. 1. On WMT14 De–En, ELF-B achieves a BLEU score of 26.4, outperforming all compared baselines. On XSum, ELF-B also outperforms all compared baselines across all ROUGE metrics. These results demonstrate the effectiveness of ELF on conditional generation tasks. Qualitative examples in Fig. 8 show that ELF-B generally follows the input context and produces outputs that semantically align with the ground-truth references. 

## **5 Conclusion** 

We introduced **Embedded Language Flows** (ELF), a continuous diffusion language model that formulates language generation in continuous embedding space using continuous-time Flow Matching. In contrast to prior DLMs, ELF keeps the denoising trajectory continuous and applies discretization only at the final step, enabling straightforward adaptation of techniques from continuous diffusion models. Empirically, compared with leading discrete DLMs and existing continuous DLMs, ELF achieves a strong quality–efficiency trade-off across language generation tasks, attaining lower generative perplexity with fewer sampling steps and fewer training tokens. These results suggest that continuous DLMs remain a promising direction for diffusion-based language modeling. 

## **Acknowledgments and Disclosure of Funding** 

We thank Mingyang Deng, Belinda Li, Itamar Pres, and Laura Ruis, for their helpful feedback and insightful discussions. We thank Google TPU Research Cloud (TRC) for granting us access to TPUs. 

9 

## **References** 

- [1] Xinyue Ai, Yutong He, Albert Gu, Ruslan Salakhutdinov, J Zico Kolter, Nicholas Matthew Boffi, and Max Simchowitz. Joint distillation for fast likelihood evaluation and sampling in flow-based models. In _ICLR_ , 2026. 6 

- [2] Michael Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. _JMLR_ , 2025. 2, 3, 15 

- [3] Michael Samuel Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants. In _ICLR_ , 2023. 1, 2, 4 

- [4] Marianne Arriola, Yair Schiff, Hao Phung, Aaron Gokaslan, and Volodymyr Kuleshov. Encoderdecoder diffusion language models for efficient training and inference. In _NeurIPS_ , 2025. 3, 8, 9, 27 

- [5] Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. In _NeurIPS_ , 2021. 1, 2, 3 

- [6] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. FLUX.1 Kontext: Flow matching for in-context image generation and editing in latent space. _arXiv preprint arXiv:2506.15742_ , 2025. 2 

- [7] Ondrej Bojar, Christian Buck, Christian Federmann, Barry Haddow, Philipp Koehn, Johannes Leveling, Christof Monz, Pavel Pecina, Matt Post, Herve Saint-Amand, Radu Soricut, Lucia Specia, and Ales Tamchyna. Findings of the 2014 workshop on statistical machine translation. In _ACL Workshop on Statistical Machine Translation_ , 2014. 2, 6 

- [8] Huayu Chen, Kai Jiang, Kaiwen Zheng, Jianfei Chen, Hang Su, and Jun Zhu. Visual generation without guidance. In _ICML_ , 2025. 6, 18 

- [9] Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In _ICLR_ , 2023. 5, 18 

- [10] Yuxin Chen, Chumeng Liang, Hangke Sui, Ruihan Guo, Chaoran Cheng, Jiaxuan You, and Ge Liu. Langflow: Continuous diffusion rivals discrete in language modeling. _arXiv preprint arXiv:2604.11748_ , 2026. 2, 3, 6, 8, 15, 25 

- [11] Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast LLMs via self-distillation through time. In _ICLR_ , 2025. 8 

- [12] Justin Deschenaux, Caglar Gulcehre, and Subham Sekhar Sahoo. The diffusion duality, chapter ii: _ψ_ -samplers and efficient curriculum. In _ICLR_ , 2026. 3 

- [13] Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Rémi Leblond, Will Grathwohl, and Jonas Adler. Continuous diffusion for categorical data. _arXiv preprint arXiv:2211.15089_ , 2022. 1, 2, 5, 9, 15, 27 

- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. Scaling rectified flow Transformers for high-resolution image synthesis. In _ICML_ , 2024. 2, 6 

- [15] Zhujin Gao, Junliang Guo, Xu Tan, Yongxin Zhu, Fang Zhang, Jiang Bian, and Linli Xu. Empowering diffusion models on the embedding space for text generation. In _NAACL_ , 2024. 2, 15 

- [16] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. In _NeurIPS_ , 2025. 6, 18 

10 

- [17] Zhengyang Geng, Yiyang Lu, Zongze Wu, Eli Shechtman, J Zico Kolter, and Kaiming He. Improved mean flows: On the challenges of fastforward generative models. _arXiv preprint arXiv:2512.02012_ , 2025. 6, 18 

- [18] Aaron Gokaslan and Vanya Cohen. Openwebtext corpus, 2019. 6, 25 

- [19] Shansan Gong, Mukai Li, Jiangtao Feng, Zhiyong Wu, and LingPeng Kong. Diffuseq: Sequence to sequence text generation with diffusion models. In _ICLR_ , 2023. 1, 2, 15 

- [20] Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation. In _ICLR_ , 2026. 3 

- [21] Ishaan Gulrajani and Tatsunori B Hashimoto. Likelihood-based diffusion language models. In _NeurIPS_ , 2023. 2, 15 

- [22] Xiaochuang Han, Sachin Kumar, and Yulia Tsvetkov. SSD-LM: Semi-autoregressive simplexbased diffusion language model for text generation and modular control. In _ACL_ , 2023. 2, 15 

- [23] Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuan-Jing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. In _ACL_ , 2023. 1, 3 

- [24] Alex Henry, Prudhvi Raj Dachapally, Shubham Shantaram Pawar, and Yuxuan Chen. Query-key normalization for Transformers. In _Findings of EMNLP_ , 2020. 24 

- [25] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In _NeurIPS Workshops_ , 2021. 5, 6 

- [26] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In _NeurIPS_ , 2020. 1, 2, 3, 15, 16 

- [27] Jaehyeong Jo and Sung Ju Hwang. Continuous diffusion model for language modeling. In _NeurIPS_ , 2025. 2, 15 

- [28] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. Technical report, Keller Jordan blog, 2024. 6, 23 

- [29] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In _NeurIPS_ , 2022. 23 

- [30] Chanhyuk Lee, Jaehoon Yoo, Manan Agarwal, Sheel Shah, Jerry Huang, Aditi Raghunathan, Seunghoon Hong, Nicholas M Boffi, and Jinwoo Kim. Flow map language models: One-step language modeling via continuous denoising. _arXiv preprint arXiv:2602.16813_ , 2026. 2, 3, 5, 6, 8, 15, 25 

- [31] Lijiang Li, Zuwei Long, Yunhang Shen, Heting Gao, Haoyu Cao, Xing Sun, Caifeng Shan, Ran He, and Chaoyou Fu. Omni-diffusion: Unified multimodal understanding and generation with masked discrete diffusion. _arXiv preprint arXiv:2603.06577_ , 2026. 3 

- [32] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. _arXiv preprint arXiv:2511.13720_ , 2025. 2, 4, 6, 20, 21, 22 

- [33] Tianyi Li, Mingda Chen, Bowei Guo, and Zhiqiang Shen. A survey on diffusion language models. _arXiv preprint arXiv:2508.10875_ , 2025. 1, 3 

- [34] Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-LM improves controllable text generation. In _NeurIPS_ , 2022. 1, 2, 15 

- [35] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In _ACL Workshop on Text Summarization Branches Out_ , 2004. 6 

11 

- [36] Zhenghao Lin, Yeyun Gong, Yelong Shen, Tong Wu, Zhihao Fan, Chen Lin, Nan Duan, and Weizhu Chen. Text generation with diffusion language models: A pre-training approach with continuous paragraph denoise. In _ICML_ , 2023. 2, 15 

- [37] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In _ICLR_ , 2023. 1, 2, 3, 4, 15 

- [38] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In _ICLR_ , 2023. 1, 2, 3, 4, 15 

- [39] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _ICLR_ , 2019. 23 

- [40] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In _ICML_ , 2024. 1 

- [41] Justin Lovelace, Varsha Kishore, Chao Wan, Eliot Shekhtman, and Kilian Q Weinberger. Latent diffusion for language generation. In _NeurIPS_ , 2023. 2, 3, 5, 15, 27 

- [42] Justin Lovelace, Varsha Kishore, Yiwei Chen, and Kilian Q Weinberger. Diffusion guided language modeling. In _Findings of ACL_ , 2024. 3, 15 

- [43] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring flow and diffusion-based generative models with scalable interpolant Transformers. In _ECCV_ , 2024. 2, 5, 19 

- [44] Rabeeh Karimi Mahabadi, Hamish Ivison, Jaesung Tae, James Henderson, Iz Beltagy, Matthew E Peters, and Arman Cohan. Tess: Text-to-text self-conditioned simplex diffusion. In _EACL_ , 2024. 2, 5, 15 

- [45] Viacheslav Meshchaninov, Egor Chimbulatov, Alexander Shabalin, Aleksandr Abramov, and Dmitry Vetrov. Cosmos: Compressed and smooth latent space for text diffusion modeling. In _NeurIPS_ , 2025. 2, 3, 15 

- [46] Shashi Narayan, Shay B. Cohen, and Mirella Lapata. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In _EMNLP_ , 2018. 2, 6 

- [47] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In _ICML_ , 2021. 2, 3, 16 

- [48] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In _NeurIPS_ , 2025. 1, 3 

- [49] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In _ACL_ , 2002. 6 

- [50] William Peebles and Saining Xie. Scalable diffusion models with Transformers. In _ICCV_ , 2023. 18, 24 

- [51] Peter Potaptchik, Jason Yim, Adhi Saravanan, Peter Holderrieth, Eric Vanden-Eijnden, and Michael S Albergo. Discrete flow maps. _arXiv preprint arXiv:2604.09784_ , 2026. 3, 5, 15 

- [52] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. _OpenAI blog_ , 2019. 6 

- [53] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. _JMLR_ , 2020. 4, 6, 7, 25 

- [54] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In _CVPR_ , 2022. 2 

- [55] Daan Roos, Oscar Davis, Floor Eijkelboom, Michael Bronstein, Max Welling,[˙] Ismail[˙] Ilkan Ceylan, Luca Ambrogioni, and Jan-Willem van de Meent. Categorical flow maps. _arXiv preprint arXiv:2602.12233_ , 2026. 3, 15 

12 

- [56] Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In _NeurIPS_ , 2024. 1, 2, 3, 6, 8, 9, 25 

- [57] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The diffusion duality. In _ICML_ , 2025. 1, 2, 3, 6, 8, 9, 25, 27 

- [58] Subham Sekhar Sahoo, Jean-Marie Lemercier, Zhihan Yang, Justin Deschenaux, Jingyu Liu, John Thickstun, and Ante Jukic. Scaling beyond masked diffusion language models. _arXiv preprint arXiv:2602.15014_ , 2026. 1, 3 

- [59] Alexander Shabalin, Viacheslav Meshchaninov, Egor Chimbulatov, Vladislav Lapikov, Roman Kim, Grigory Bartosh, Dmitry Molchanov, Sergey Markov, and Dmitry Vetrov. TEncDM: Understanding the properties of the diffusion model in the space of language model encodings. In _AAAI_ , 2025. 3, 5, 15 

- [60] Alexander Shabalin, Simon Elistratov, Viacheslav Meshchaninov, Ildus Sadrtdinov, and Dmitry Vetrov. Why gaussian diffusion models fail on discrete data? _arXiv preprint arXiv:2604.02028_ , 2026. 5 

- [61] Noam Shazeer. GLU variants improve Transformer. _arXiv preprint arXiv:2002.05202_ , 2020. 24 

- [62] Junzhe Shen, Jieru Zhao, Ziwei He, and Zhouhan Lin. Codar: Continuous diffusion language models are more powerful than you think. _arXiv preprint arXiv:2603.02547_ , 2026. 2, 3, 15 

- [63] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In _ICML_ , 2015. 1, 2 

- [64] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In _ICLR_ , 2021. 1, 2, 15 

- [65] Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Yonghui Wu, and Hao Zhou. Seed diffusion: A large-scale diffusion language model with high-speed inference. _arXiv preprint arXiv:2508.02193_ , 2025. 3 

- [66] Robin Strudel, Corentin Tallec, Florent Altché, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, and Rémi Leblond. Selfconditioned embedding diffusion for text generation. _arXiv preprint arXiv:2211.04236_ , 2022. 2, 5, 15 

- [67] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. _Neurocomputing_ , 568:127063, 2024. 24 

- [68] Jaesung Tae, Hamish Ivison, Sachin Kumar, and Arman Cohan. Tess 2: A large-scale generalist diffusion language model. In _ACL_ , 2025. 2, 15 

- [69] Zhicong Tang, Jianmin Bao, Dong Chen, and Baining Guo. Diffusion models without classifierfree guidance. _arXiv preprint arXiv:2502.12154_ , 2025. 6, 18 

- [70] Wan Team, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. _arXiv preprint arXiv:2503.20314_ , 2025. 2 

- [71] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. In _NeurIPS_ , 2025. 3 

- [72] Renzhi Wang, Jing Li, and Piji Li. InfoDiffusion: Information entropy aware diffusion process for non-autoregressive text generation. In _Findings of EMNLP_ , 2023. 2, 15 

13 

- [73] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. In _ICLR_ , 2026. 3 

- [74] Tong Wu, Zhihao Fan, Xiao Liu, Hai-Tao Zheng, Yeyun Gong, Jian Jiao, Juntao Li, Jian Guo, Nan Duan, and Weizhu Chen. AR-Diffusion: Auto-regressive diffusion model for text generation. In _NeurIPS_ , 2023. 2, 15 

- [75] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. In _NeurIPS_ , 2025. 3 

- [76] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. _arXiv preprint arXiv:2508.15487_ , 2025. 1, 3 

- [77] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Mingxuan Wang. DINOISER: Diffused conditional sequence learning by manipulating noises. _Transactions of the Association for Computational Linguistics_ , 2024. 2, 15 

- [78] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. _arXiv preprint arXiv:2505.16933_ , 2025. 3 

- [79] Hongyi Yuan, Zheng Yuan, Chuanqi Tan, Fei Huang, and Songfang Huang. Seqdiffuseq: Text diffusion with encoder-decoder transformers. In _NAACL_ , 2024. 2, 5, 9, 15 

- [80] Biao Zhang and Rico Sennrich. Root mean square layer normalization. In _NeurIPS_ , 2019. 24 

- [81] Yizhe Zhang, Jiatao Gu, Zhuofeng Wu, Shuangfei Zhai, Josh Susskind, and Navdeep Jaitly. PLANNER: Generating diversified paragraphs via latent language diffusion model. In _NeurIPS_ , 2023. 3, 15 

14 

||||**Train**|**Infer.**||
|---|---|---|---|---|---|
|**Method**|**Process**_†_|**State**_‡_|**per-step**|**per-step**|**Sep. dec.**|
||||**discr.**|**discr.**||
|_Embedding-space Diffusion LMs_||||||
|Diffusion-LM [Li et al.,2022]|DDPM|learn emb|Yes|Yes||
|SED [Strudel et al.,2022]|DDPM|fx emb|Yes|||
|CDCD [Dieleman et al.,2022]|Score-ODE|learn emb|Yes|||
|DiffuSeq [Gong et al.,2023]|DDPM|learn emb|Yes|Yes||
|GENIE [Lin et al.,2023]|DDPM|learn emb|Yes|||
|AR-Diffusion [Wu et al.,2023]*|DDPM|learn emb|Yes|||
|Plaid [Gulrajani and Hashimoto,2023]|VLB|learn emb|Yes|||
|InfoDiffusion [Wang et al.,2023]|DDPM|learn emb|Yes|||
|Difformer [Gao et al.,2024]|DDPM|learn emb|Yes|||
|SeqDiffuSeq [Yuan et al.,2024]|DDPM|learn emb|Yes|||
|DINOISER [Ye et al.,2024]|SDE/DDIM|learn emb|Yes|||
|_Simplex Diffusion LMs_||||||
|SSD-LM [Han et al.,2023]*|DDPM|simplex|Yes|Yes||
|TESS [Mahabadi et al.,2024]|DDPM|simplex|Yes|Yes||
|RDLM [Jo and Hwang,2025]|RDM|simplex|Yes|||
|TESS 2 [Tae et al.,2025]|DDPM|simplex|Yes|Yes||
|_Latent Diffusion LMs_||||||
|LD4LG [Lovelace et al.,2023]*|DDPM|fx enc|||Yes|
|PLANNER [Zhang et al.,2023]*|DDPM|fx enc|||Yes|
|DGLM [Lovelace et al.,2024]*|VP-DDPM|fx enc|||Yes|
|TEncDM [Shabalin et al.,2025]|VP-DDPM|fx enc|||Yes|
|Cosmos [Meshchaninov et al.,2025]|VP-DDPM|fx enc|||Yes|
|CoDAR [Shen et al.,2026]*|VP-SDE|fx enc|||Yes|
|_Flow-based LMs_||||||
|CFM [Roos et al.,2026]|FM|simplex|Yes|||
|FLM [Lee et al.,2026]|FM|one-hot|Yes|||
|DFM [Potaptchik et al.,2026]|FM|simplex|Yes|||
|LangFlow [Chen et al.,2026]|Bregman FM|learn emb|Yes|||
|**ELF (ours)**|**FM**|**fx enc**||||



> † _Process:_ FM = Flow Matching [37, 38, 2]; DDPM = Denoising Diffusion Probabilistic Model [26]; VPDDPM/-SDE = variance-preserving DDPM / stochastic differential equation [64]; Score-ODE = probabilityflow ODE [64]; SDE/DDIM = continuous-time SDE [64] integrated with the deterministic DDIM solver; VLB = variational lower bound, specifically Plaid’s _T →∞_ continuous-time limit [21]; RDM = Riemannian Diffusion Mixture, applied to the categorical sphere by RDLM [27]; Bregman FM = Flow Matching with a Bregman-divergence regression objective, used by LangFlow [10]. 

> ‡ _State:_ learn emb = jointly trained token embedding matrix; fix emb = frozen pretrained embedding lookup; fix enc = frozen pretrained encoder, optionally with a compressed autoencoder bottleneck on top; simplex = vocabulary-shaped logit simplex or square-root simplex on the sphere; one-hot = per-token one-hot stack over the vocabulary. 

Table 2: **Survey of continuous diffusion and flow-based language models.** We summarize representative continuous diffusion and flow-based language models along several design axes. _Process_ denotes the diffusion or flow process, with green indicating continuous-time formulations and red indicating discrete-time formulations. _State_ denotes the continuous state in which denoising is performed. _Train per-step discr._ marks methods that convert intermediate denoising states to token predictions during training and apply token-level supervision such as cross-entropy loss at intermediate steps. _Infer. per-step discr._ marks methods that project intermediate sampling states back to token-aligned states during generation. _Sep. dec._ marks methods that require a separately trained decoder to map latent representations back to text. Blank entries indicate absence.[*] denotes autoregressive or block-autoregressive generation. 

## **A Continuous Diffusion Language Model Survey** 

**Survey details.** We provide a detailed survey in Tab. 2. The survey summarizes representative continuous diffusion and flow-based language models along several design axes, including the underlying diffusion or flow process, the continuous state in which denoising is performed, whether intermediate denoising states are discretized during training or inference, and whether a separately trained decoder is required to map latent states back to text. 

In particular, the _Train per-step discr._ and _Infer. per-step discr._ columns distinguish two different uses of intermediate discretization. _Train per-step discr._ indicates that intermediate denoising states are 

15 

**==> picture [396 x 167] intentionally omitted <==**

**----- Start of picture text -----**<br>
input tokens � �� �� �<br>�1<br>....�2 embed corrupt self-conditon add control ELF or<br>�� ....<br>corrupt �� self-condition �� add control<br>�� ��<br>� time embed<br>add condition project<br>�� � [’] CFG scale<br>model mode<br>**----- End of picture text -----**<br>


Figure 9: **Illustration of our training pipeline.** Starting from the clean embeddings _**x**_ , we apply different noise schedules in the two modes to obtain corrupted embeddings _**z** t_ . We then apply selfconditioning by concatenating either **0** or the previous prediction _**x**_ ˆ _[′]_ along the channel dimension, and project the concatenated embeddings back to the original dimension to form _**z**_ ˆ _t_ . Next, we prepend control tokens to the embedding sequence, including time tokens in [0 _,_ 1], CFG scale tokens in [0is fed into ELF to produce the final prediction _._ 5 _,_ 5], and model-mode tokens indicating either denoising or decoding. _**x**_ ˆ, which is supervised using either a denoising lossThe resulting sequence _L_ MSE or a token-wise cross-entropy loss _L_ CE. 

mapped to token predictions during training and supervised with token-level objectives such as crossentropy loss. This provides direct vocabulary-level guidance, but also couples intermediate denoising states to categorical predictions. _Infer. per-step discr._ indicates that intermediate sampling states are explicitly projected back to token-aligned representations during generation, such as nearest-neighbor rounding in embedding space or argmax projection on a simplex. Methods without inference-time per-step discretization keep the sampling trajectory continuous and discretize only at the final step. The _Sep. dec._ column indicates whether a method requires a separately trained decoder to map continuous latent representations back to discrete text. 

**Positioning of ELF.** Tab. 2 shows that existing continuous DLMs differ substantially in where the denoising process is defined and how continuous states are mapped back to text. Many embeddingspace and simplex-based methods use training-time per-step discretization through token-level objectives, commonly cross-entropy, at intermediate denoising steps. These objectives provide direct token-level guidance, while making the denoising trajectory more tightly coupled to vocabulary-level prediction. Latent Diffusion LMs often avoid such per-step vocabulary supervision, but typically rely on DDPM-style or score-based formulations with DDPM noise schedules [26, 47] and require a separately trained latent-to-text decoder, such as an autoregressive decoder, non-autoregressive decoder, or latent decompressor, to recover discrete tokens. 

ELF occupies a different design point. It formulates language generation as continuous-time Flow Matching in a frozen contextual embedding space and keeps the sampling trajectory continuous, applying discretization only at the final decoding step. Unlike prior latent Diffusion LMs, ELF does not require a separately trained decoder: a single shared-weight network performs intermediate denoising and recovers tokens at the final step through the unembedding layer. 

## **B Method Details** 

## **B.1 Training** 

We show the full training pipeline in Fig. 9. The input tokens are first encoded into clean embeddings _**x**_ , which then go through three key steps before being fed into the ELF model: corruption, selfconditioning, and adding control tokens for conditioning and guidance. In the denoising branch, the model predicts clean embeddings _**x**_ ˆ and is supervised with _L_ MSE. In the decoding branch, the same 

16 

**Algorithm 3** ELF denoiser training with conditioning and guidance. 

```
#net(z,t,c,w,mode):ELFnetworkwithin-contextconditioning
#self_cond_proj(z):Self-conditioningprojectionlayerthatconvertsconcatenated
embeddingsbacktotheoriginalembeddingdimension
#self_cond_prob:Self-conditioningprobability
#s:asequenceofdiscretetokens
#c:condition(onlyforconditionalgeneration)
```

```
x=encode(s)
t=sample_t()
w=sample_sc_cfg_scale()
e=randn_like(x)
z=t*x+(1-t)*e
v=x-e
```

```
#zw/oself-conditioning
z_no_sc=self_cond_proj(concat([z,zeros_like(z)],dim=-1))
x_no_sc=net(z_no_sc,t,c,w,mode="denoise")
v_no_sc=(x_no_sc-z)/(1-t)
```

```
#zw/self-conditioning
z_sc=self_cond_proj(concat([z,stopgrad(x_no_sc)],dim=-1))
x_sc=net(z_sc,t,c,w,mode="denoise")
v_sc=(x_sc-z)/(1-t)
```

```
#ComputeCFGtarget
v_target=v+(1-1/w)*(v_sc-v_no_sc)
```

```
#Applyper-exampleself-conditioningmask
self_cond_mask=uniform(x.shape[0])<self_cond_prob
v_pred=where(self_cond_mask,v_sc,v_no_sc)
v_target=where(self_cond_mask,v_target,v)
v_target=stopgrad(v_target)
```

```
#Computev-loss
loss=mse_loss(v_pred,v_target)
```

shared-weight network predicts embeddings that are then passed through an unembedding layer and supervised with _L_ CE. The full training algorithm is shown in Alg. 3 and Alg. 4. 

**Embedding corruption.** First, we corrupt the clean embeddings _**x**_ by adding noise. Specifically, we use _**z** t_ = _t_ _**x**_ + (1 _− t_ ) _**ϵ**_ to obtain noisy embeddings _**z** t_ , where _**ϵ**_ is Gaussian noise and _t_ is the time step. Before corruption, we first normalize the clean embeddings using the estimated mean and standard deviation from the OWT dataset. We use different noise schedules for different modes. For the denoising branch, we sample the time step _t_ from a logit-normal distribution for each _sequence_ . Specifically, we draw _t[′] ∼N_ ( _P_ mean _, P_ std[2][)][ and map it to the unit interval via] _[ t]_[=] _[σ]_[(] _[t][′]_[)][, where] _[ σ]_[(] _[·]_[)] denotes the sigmoid function. In all experiments, we use _P_ mean = _−_ 1 _._ 5 and _P_ std = 0 _._ 8. We rescale the Gaussian noise by a factor of 2. 

For the decoding branch, we train final-step discretization by conditioning the model on the decoder mode, _i.e._ , _t_ = 1. At this time step, _**z** t_ corresponds to clean embeddings. Therefore, to make the final-step input nontrivial, we corrupt the clean embeddings with a per-token corruption level _p_ sampled from a different noise schedule.˜ Specifically, we draw _p_ from a logit-normal distribution with _P_ mean = 0 _._ 8 and _P_ std = 0 _._ 8, and form _**z**_ = _p_ _**x**_ +(1 _−p_ ) _**ϵ**_ , multiplying _**ϵ**_ by a noise scale. We use noise scales of 5 and 1 for OWT and conditional generation tasks, respectively. As a result, the corruption level varies across tokens within the same sequence. This design encourages the shared-weight decoder mode to recover corrupted embeddings from their surrounding context, making final-step discretization more robust to imperfect embeddings produced by the denoiser at inference time. 

17 

**Algorithm 4** ELF decoder training with conditioning and guidance. 

```
#net(z,t,c,w,mode):ELFnetworkwithin-contextconditioning
#self_cond_proj(z):Self-conditioningprojectionlayerthatconvertsconcatenated
embeddingsbacktotheoriginalembeddingdimension
#s:asequenceofdiscretetokens
#c:condition(onlyforconditionalgeneration)
```

```
x=encode(s)
p=sample_per_token_p()
w=sample_sc_cfg_scale()
e=randn_like(x)
z=p*x+(1-p)*e
#usezw/oself-conditioning
z=self_cond_proj(concat([z,zeros_like(z)],dim=-1))
h=net(z,t=1,c,w,mode="decode")
s_pred=unembed(h)
loss=ce_loss(s_pred,s)
```

**Self-conditioning.** We apply self-conditioning following prior work [9]. During training, with a certain probability, we perform an additional forward pass to obtain the predicted embeddings _**x**_ ˆ _[′]_ , which are concatenated with the noisy embeddings _**z** t_ along the channel dimension. We stop the gradient through the predicted embeddings _**x**_ ˆ _[′]_ . For the remaining examples, we concatenate _**z** t_ with all-zero embeddings **0** instead. Since this concatenation doubles the channel dimension, we project it back to the original dimension using a linear layer. We apply self-conditioning with _**x**_ ˆ _[′]_ in the denoising branch with 50% probability. For the decoding branch, we always use **0** as the self-conditioning input, as shown in Alg. 4. 

**Training-time CFG.** As discussed in Sec. 3.3, our model performs training-time CFG [16, 17, 8, 69] with self-conditioning. In training-time CFG, the network is designed to model the post-combination quantity _**v** θ_[cfg][, rather than the pre-combination quantity] _**[ v]**[θ]_[.][Following [][16][,][ 17][], the regression target] _**v**_ target is now: 

**==> picture [334 x 25] intentionally omitted <==**

where _ω_ is the guidance scale. When _ω_ = 1, this reduces to the case without training-time CFG. In this case, the loss becomes _∥_ _**v** θ_[cfg][(] _[·]_[)] _[ −]_ _**[v]**_[target] _[∥]_[2][[][16][,][ 17][].][See Alg.][ 3][.][For each training example,] we randomly sample a self-conditioning CFG scale _w ∈_ [0 _._ 5 _,_ 5 _._ 0] from a power distribution biased toward smaller values [16, 17]. Since ELF uses _**x**_ -prediction, the quantity _**v**_ is always converted from its _**x**_ prediction counterpart (conditional or unconditional). 

Our model uses a diverse set of conditions. Standard diffusion models typically implement conditioning through adaLN-Zero [50], which combines all conditioning signals through summation. This design becomes less effective when many heterogeneous conditions are present. Therefore, we adopt in-context conditioning [17] by prepending a set of _control_ tokens that encode the conditioning information. Each control-token embedding has the same dimensionality as a standard languagetoken embedding. We prepend three types of control tokens: 4 time tokens with values in [0 _,_ 1], 4 CFG-scale tokens sampled from [0 _._ 5 _,_ 5], and 4 model-mode tokens indicating either denoising or decoding. These tokens are jointly trained with the model. All continuous values, _i.e._ , time and CFG scale, are encoded with positional embeddings. 

For conditional generation, we place the clean embeddings of the conditioning sequence immediately after the control tokens and before the target sequence to be generated. The model then performs bidirectional self-attention over the concatenated sequence of conditioning and target tokens. The conditioning embeddings are kept uncorrupted during training. To enable CFG for conditional generation, we randomly drop the condition with 10% probability by zeroing out the embeddings of the conditioning sequence. This allows the model to learn both conditional and unconditional generation under the same framework. 

18 

**Algorithm 5** ELF inference with conditioning and guidance. 

```
#net(z,t,c,w,mode):ELFnetworkwithin-contextconditioning
#self_cond_proj(z):Self-conditioningprojectionlayerthatconvertsconcatenated
embeddingsbacktotheoriginalembeddingdimension
#shape:embeddingsshape
#ts:discretizedtimegridover[0,1]withNintervals
#c:condition(onlyforconditionalgeneration)
#w:self-conditioningCFGscale
```

```
z=randn(shape)
x_pred=zeros(shape)
foriinrange(len(ts)-1):
t=ts[i]
dt=ts[i+1]-ts[i]
#Self-conditiononthepreviousprediction
z_sc=self_cond_proj(concat([z,x_pred],dim=-1))
x_pred=net(z_sc,t,c,w,mode="denoise")
#convertxpredictiontovelocity
v=(x_pred-z)/(1-t)
z=z+dt*v
#decoding
z=self_cond_proj(concat([z,zeros_like(z)],dim=-1))
h=net(z,t=1,c,w,mode="decode")
#unembedding
token_logits=unembed(h)
tokens=argmax(token_logits)
```

## **B.2 Inference** 

We show the full inference algorithm in Alg. 5. Since the self-conditioning CFG scale is provided through in-context conditioning, changing _w_ does not require an additional inference pass. By modifying _w_ as a model input, we can flexibly control the trade-off between generation quality and diversity. 

**Time schedule.** We discretize the continuous time interval _t ∈_ [0 _,_ 1] into _T_ intervals using a logit-normal time schedule. Specifically, we sample _T −_ 1 time steps from the same logit-normal distribution used for the denoising branch during training and sort them to form the intermediate points. We use _P_ mean = _−_ 1 _._ 5 and _P_ std = 0 _._ 8 to match the training-time logit-normal distribution. We ensure that the first interval starts at _t_ = 0 and the last interval ends at _t_ = 1. This schedule produces smaller intervals when _t_ is close to 0 and larger intervals as _t_ approaches 1. It shows strong empirical performance, likely because the noisier regime requires finer discretization and the schedule better matches the noise distribution used during training. 

**Samplers.** Our method supports both deterministic ODE sampling and an SDE-inspired stochastic sampler. The main algorithm in Alg. 2 uses the ODE sampler for simplicity, while Alg. 6 summarizes one-step updates for both samplers. 

The SDE variant is motivated by the SDE associated with Flow Matching [43], whose dynamics can be interpreted as injecting infinitesimal noise at each step. In practice, we adopt a simple approximation that re-injects Gaussian noise at each sampling step while shifting the time variable slightly toward the noise regime. We introduce a noise re-injection scale _γ_ to control the amount of stochasticity added at each step. The denoiser is then evaluated on this perturbed state, and its clean-embedding prediction is used to update the original state. When _γ_ = 0, no stochastic perturbation is applied, and the update reduces to deterministic ODE sampling. 

19 

**Algorithm 6** ELF inference with different samplers. 

```
#z:noisyembeddingsofcurrenttimestep
#t:currenttimestep
#dt:timeinterval,t_next-t
#gamma:controlstheamountofnoiseaddedbackintheSDEsampler
defode_step(z,t,dt):
x_hat=net(z,t,mode="denoise")
v=(x_hat-z)/(1-t)
z=z+dt*v
returnz
defsde_step(z,t,dt,gamma):
#Re-injectnoiseandmovebacktothecorrespondingtimestep
#Thejumpsizeisdefinedrelativetothetime-stepinterval
e=randn_like(z)
alpha=1-gamma*dt
t_back=alpha*t
z_back=alpha*z+(1-alpha)*e
x_hat=net(z_back,t_back,mode="denoise")
v=(x_hat-z)/(1-t)
z=z+dt*v
returnz
```

**CFG for conditional generation.** We apply standard CFG by combining the conditional and unconditional predictions. Similarly, we use the CFG scale to control the guidance strength. 

## **C Additional Ablations** 

In this section, we present additional ablations of our design choices. Unless otherwise specified, all experiments use time schedule with either a 64-step ODE sampler or a 64-step SDE sampler with _γ_ = 1. As before, we evaluate the generative perplexity–entropy trade-off by varying the self-conditioning CFG scale. We use red to indicate regions with poor generation quality, _i.e._ , entropy below 5.0, which often corresponds to repetitive or degenerate sentences, or generative perplexity above 300, which often corresponds to semantically meaningless or ungrammatical sentences. All models are trained for the same number of steps, with all other configurations kept the same as the default setting. 

## **C.1 Prediction Targets** 

Our model directly predicts the clean embeddings _**x**_ ( _**x**_ -prediction). This allows us to use a unified denoiser and decoder through weight sharing and jointly optimize the model with both the denoising objective _L_ MSE and the token-level objective _L_ CE. Prior work has also suggested that _**x**_ -prediction is essential, as high-dimensional clean data tends to lie on a low-dimensional manifold [32]. 

Here, we further study the effect of prediction targets. Specifically, since there are three quantities and two constraints: linear interpolation _**z** t_ = _t_ _**x**_ + (1 _− t_ ) _**ϵ**_ and flow velocity _**v**_ = _**x** −_ _**ϵ**_ , the network can be trained to predict one of these quantities, _i.e._ , _**x**_ -, _**v**_ -, or _**ϵ**_ -prediction. To study this in a controlled setting, we use a two-stage pretrained encoder-decoder setup: a pretrained T5 encoder maps tokens into continuous embeddings, and a decoder is trained to reconstruct masked and noisy embeddings (See Sec. D.3 for details). We train only the denoising model while keeping both the encoder and decoder fixed. We use adaLN-Zero conditioning and a 64-step ODE sampler to plot the generative perplexity–entropy trade-off curve. 

To study how prediction targets behave as the embedding dimension increases, we consider T5-small, T5-base, and T5-large encoders, corresponding to embedding dimensions of 512, 768, and 1024, respectively. We set the bottleneck dimension equal to the corresponding input embedding dimension. 

20 

**==> picture [396 x 112] intentionally omitted <==**

**----- Start of picture text -----**<br>
Dimension=512 Dimension=768 Dimension=1024<br>800<br>500 xv--predictionprediction 500 x v -prediction -prediction 700 xv-prediction-prediction<br>-prediction -prediction -prediction<br>400 400 600<br>500<br>300 300<br>400<br>200 200 300<br>100 200<br>100<br>100<br>0<br>5.2 5.3 5.4 5.5 5.6 5.7 5.8 2.5 3.0 3.5 4.0 4.5 5.0 5.5 5.1 5.2 5.3 5.4 5.5 5.6 5.7<br>Entropy  Entropy  Entropy<br>(a) (b) (c)<br>Gen. PPL  Gen. PPL  Gen. PPL<br>**----- End of picture text -----**<br>


Figure 10: **Effects of prediction targets.** We vary the input dimension from 512 to 768 and 1024 by using T5-small, T5-base, and T5-large encoders, respectively. Across all input dimensions, _**x**_ -prediction remains stable and performs well. In contrast, _**v**_ -prediction performs well at 512 dimensions but degrades at higher dimensions, while _**ϵ**_ -prediction collapses across all dimensions from 512 to 1024. The red region indicates poor-quality generations, where entropy falls below 5 ( _e.g._ , repetitive sentences) or generative perplexity exceeds 300 ( _e.g._ , meaningless or ungrammatical sentences). This aligns with the hypothesis from prior work that high-dimensional clean data often lies on a low-dimensional manifold [32]. 

**==> picture [298 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
ODE SDE<br>140 32 60 32<br>128 128<br>120 512 512<br>50<br>100<br>40<br>80<br>60 30<br>40 20<br>20<br>10<br>4.9 5.0 5.1 5.2 5.3 5.4 5.5 5.6 4.7 4.8 4.9 5.0 5.1 5.2 5.3<br>Entropy  Entropy<br>(a) (b)<br>Gen. PPL  Gen. PPL<br>**----- End of picture text -----**<br>


Figure 11: **Effect of bottleneck dimension.** We compare bottleneck dimensions of 32, 128, and 512 under ODE and SDE sampling. A moderate bottleneck dimension of 128 provides the best generative perplexity–entropy trade-off, while overly small or large bottlenecks either reduce diversity or hurt generative perplexity. Red indicates regions with poor generation quality, _i.e._ , entropy below 5. 

As shown in Fig. 10, _**x**_ -prediction remains the most stable across all dimensions, maintaining a reasonable generative perplexity–entropy trade-off even at 1024 dimensions. In contrast, _**v**_ -prediction is competitive at 512 dimensions but degrades as the dimension increases, with substantially higher generative perplexity at 768 and 1024 dimensions. _**ϵ**_ -prediction collapses across all dimensions, either achieving extremely low entropy or high generative perplexity, indicating repetitive, degenerate, or ungrammatical generations. These results support the hypothesis that clean-data prediction is better suited to high-dimensional language representations, consistent with findings from prior work [32]. 

## **C.2 Bottleneck** 

Our model uses a bottleneck design that projects encoder representations into a lower-dimensional space before mapping them back to the model hidden size. This design is motivated by the hypothesis that natural data may lie on a low-dimensional manifold within the high-dimensional embedding space. We compare bottleneck dimensions of 32, 128, and 512, and show the results in Fig. 11. The bottleneck dimension has a clear effect on the generative perplexity–entropy trade-off. Under ODE sampling, all three bottleneck sizes follow a similar frontier, but smaller bottlenecks tend to reach lower generative perplexity at the cost of lower entropy. Under SDE sampling, the differences become more significant: the 32-dimensional bottleneck achieves the lowest generative perplexity but often lies in the low-entropy region, indicating reduced diversity, whereas the 512-dimensional bottleneck maintains higher entropy but suffers from substantially worse generative perplexity. The 128-dimensional bottleneck provides the best overall balance, achieving strong generative perplexity while preserving reasonable entropy. We therefore use a bottleneck dimension of 128 as the default 

21 

**==> picture [298 x 118] intentionally omitted <==**

**----- Start of picture text -----**<br>
ODE SDE<br>120 Denoising mode prob = 0.8 45 Denoising mode prob = 0.8<br>Denoising mode prob = 0.5 Denoising mode prob = 0.5<br>100 Denoising mode prob = 0.2 40 Denoising mode prob = 0.2<br>80 35<br>60 30<br>40 25<br>20 20<br>5.0 5.1 5.2 5.3 5.4 5.5 5.0 5.0 5.1 5.2 5.2 5.2 5.3<br>Entropy  Entropy<br>(a) (b)<br>Gen. PPL<br>**----- End of picture text -----**<br>


Figure 12: **Effect of the denoising mode probability during training.** This probability controls the allocation between denoising and decoding updates in the shared-weight denoiser-decoder model. A denoising mode probability of 0 _._ 8 provides the best generative perplexity–entropy trade-off across both ODE and SDE samplers. 

**==> picture [397 x 119] intentionally omitted <==**

**----- Start of picture text -----**<br>
ODE SDE ODE SDE<br>80 80 80 80<br>In-context In-context Muon Muon<br>70 AdaLN-Zero 70 AdaLN-Zero 70 AdamW 70 AdamW<br>60 60 60 60<br>50 50 50 50<br>40 40 40 40<br>30 30 30 30<br>20 20 20 20<br>10 10 10 10<br>5.2 5.3 5.4 5.1 5.2 5.2 5.3 5.3 5.1 5.2 5.3 5.4 5.1 5.2 5.2 5.3<br>Entropy  Entropy  Entropy  Entropy<br>(a) (b) (a) (b)<br>Gen. PPL  Gen. PPL<br>**----- End of picture text -----**<br>


Figure 13: **Effect of conditioning strategies.** We compare in-context conditioning with adaLNZero conditioning. In-context conditioning slightly improves performance while substantially reducing the number of model parameters. 

Figure 14: **Effect of optimizers.** We compare generation quality under different optimizers using Muon and AdamW. Muon achieves lower generative perplexity at comparable entropy under both ODE and SDE sampling methods. 

setting. This finding is also consistent with prior work [32], which observes that an appropriate bottleneck can improve performance. 

## **C.3 Denoising Mode Probability** 

Since ELF is trained with both MSE and CE losses through a shared-weight denoiser-decoder, each training step is assigned to either denoising mode or decoding mode. The denoising-mode probability controls this allocation: a higher probability emphasizes learning the continuous denoising dynamics, while a lower probability provides more supervision for mapping embeddings back to tokens. We study this trade-off by varying the denoising-mode probability during training. 

As shown in Fig. 12, assigning a low probability to the denoising mode consistently degrades the generative perplexity–entropy trade-off, especially under SDE sampling. This suggests that the model requires sufficient training on the denoising process. Among the configurations tested, a denoising mode probability of 0 _._ 8 achieves the best overall trade-off across both ODE and SDE samplers. We therefore use 0 _._ 8 as the default denoising mode probability in our main experiments. 

## **C.4 Conditioning Strategies** 

As discussed in Sec. 3.3, our model is conditioned on the time step, CFG scale, and model mode. We use in-context conditioning for these signals by prepending them as condition tokens to the input sequence, allowing the model to attend to them through full attention. This differs from the conventional adaLN-Zero conditioning design, which typically introduces additional model components to process the conditioning inputs. We compare these two designs in Fig. 13. In-context conditioning performs slightly better while avoiding the substantial parameter overhead introduced by 

22 

**==> picture [298 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
500 Logit-normal time schedule Uniform time schedule 70<br>=0<br>400 60 =0.1<br>50<br>300<br>40 =0.5<br>200<br>30 =1 =0.75<br>100 20 =1.5<br>H=5.22 =2<br>0 H=5.11 10<br>8 16 32 64 128 256 512 1024 5.0 5.1 5.2 5.3 5.4 5.5<br>Sampling Steps Entropy<br>(a) (b)<br>Gen. PPL  Gen. PPL<br>**----- End of picture text -----**<br>


Figure 15: **Effect of time schedule and SDE noise re-injection scale.** (a) Logit-normal time schedule consistently improves generative perplexity across different sampling budgets, especially in the few-step regime. (b) The SDE noise re-injection scale _γ_ controls the generative perplexity–entropy trade-off by adjusting the amount of stochastic noise injected during sampling. 

**==> picture [298 x 105] intentionally omitted <==**

**----- Start of picture text -----**<br>
De-En XSum<br>26.0<br>27.6<br>24.0<br>27.2<br>22.0<br>20.0 26.8<br>18.0<br>26.4<br>1 2 3 4 1 2 3 4<br>CFG CFG<br>(a) (b)<br>BLEU<br>ROUGE-L<br>**----- End of picture text -----**<br>


Figure 16: **Effect of CFG scale on conditional generation.** We sweep the CFG scale on WMT14 DeEn translation and XSum summarization. Moderate guidance substantially improves task performance, with CFG scale 2 achieving the best result on both tasks, while overly strong guidance slightly degrades performance. 

adaLN-Zero (ELF-B’s parameter count is reduced from 148M to 105M). Therefore, we use in-context conditioning as our default setting. 

## **C.5 Optimizers** 

We evaluate the impact of optimizer choice, comparing Muon [28] and AdamW [39], and show the results in Fig. 14. We tune the hyperparameters for both optimizers to obtain their best performance: for Muon, we use a learning rate of 2 _×_ 10 _[−]_[3] ; for AdamW, we use a learning rate of 1 _×_ 10 _[−]_[4] with _β_ 1 = 0 _._ 9 and _β_ 2 = 0 _._ 95. During training, Muon achieves lower loss within the same number of steps. During inference, models trained with Muon consistently achieve a better generative perplexity–entropy trade-off than those trained with AdamW under both samplers. The improvement is especially significant under SDE sampling, where Muon achieves lower generative perplexity at the same entropy level. These results highlight the importance of optimizer choice. Nevertheless, models trained with both optimizers still outperform other baselines, suggesting that the strong performance of ELF cannot be attributed to the optimizer alone. 

## **C.6 Sampling Methods** 

We study two sampling design choices that improve inference efficiency and generation quality: sampling time schedule and stochastic SDE-inspired sampling. The logit-normal time schedule improves sampling efficiency by reducing the required number of denoising steps, while the SDE noise re-injection scale provides additional control over the generative perplexity–entropy trade-off. 

**Time schedules.** By default, we use a logit-normal time schedule during inference [29]. We also evaluate an alternative uniform schedule. Fig. 15a shows the effect of the time schedule on ODE sampling across different numbers of sampling steps. Across all step counts, the logit-normal schedule consistently reduces generative perplexity compared with the uniform schedule. This improvement 

23 

|**Model**|**Depth**|**Hidden size**|**# Heads**|**Params**|**Training epochs**|
|---|---|---|---|---|---|
|ELF-B|12|768|12|105M|5|
|ELF-M|24|1056|16|342M|4|
|ELF-L|32|1280|16|652M|3|



Table 3: **ELF Model configurations** across different scales. 

is especially significant in the few-step regime. These results suggest that the logit-normal time schedule improves sampling efficiency and final sample quality, likely because it better aligns the inference-time trajectory with the training-time schedule and allocates more sampling steps to noisier time steps. 

**SDE noise re-injection scale.** For SDE sampling, we introduce a noise re-injection scale hyperparameter _γ_ that controls the amount of stochasticity injected at each sampling step, as discussed in Sec. B.2. Intuitively, increasing _γ_ introduces more stochasticity, while _γ_ = 0 reduces to deterministic ODE sampling. As shown in Fig. 15b, _γ_ controls the generative perplexity–entropy trade-off: within a moderate range, larger _γ_ leads to lower generative perplexity while slightly reducing entropy. We hypothesize that the noise re-injection process helps correct early denoising errors, rather than deterministically amplifying imperfect trajectories as in ODE sampling. We therefore choose _γ_ = 1 _._ 0 as our default setting, which provides a strong balance between generative perplexity and entropy. 

## **C.7 CFG on Conditional Generation** 

We further study the effect of CFG scale on conditional generation tasks. As shown in Fig. 16, increasing the CFG scale from 1 to 2 substantially improves performance on both WMT14 De-En and XSum, suggesting that stronger conditioning helps the model better follow the source input. However, further increasing the scale leads to a gradual decline in performance, indicating that overly strong guidance can hurt generation quality. Based on this trend, we use CFG scale 2 as the default setting for conditional generation. 

## **D Experimental Details** 

## **D.1 Model Architecture** 

Our model uses a standard Diffusion Transformer architecture [50]. We also incorporate popular general-purpose improvements, including SwiGLU [61], RMSNorm [80], RoPE [67], and qk-norm [24]. We use in-context conditioning instead of adaLN-Zero [50] conditioning, which allows us to significantly reduce the number of parameters; for example, the ELF-B model size is reduced from 148M to 105M parameters. Tab. 3 summarizes the configurations of ELF across different model sizes. We report the Transformer depth, hidden size, number of attention heads, and parameter count. We also report the number of training epochs used on the OWT dataset for each variant. Larger models tend to learn faster in our setup, and therefore require fewer training epochs. 

## **D.2 Hyperparameters** 

**ELF pipeline hyperparameters.** Tab. 4 summarizes the main hyperparameters used in the ELF pipeline, covering model architecture, diffusion settings, conditioning and guidance, and optimization details. Unless noted otherwise, all experiments in the paper follow this default configuration. We include these settings for completeness and to facilitate reproducibility. 

**Inference-time settings for system-level comparison.** For system-level comparison in Fig. 7, we use SDE sampling with time schedule enabled for all step budgets. We set the CFG scale to 3 for 8-, 16-, and 32-step generation. For SDE sampling, we use a stronger noise injection scale of _γ_ = 2 in the very few-step regimes of 8 and 16 steps, and reduce it to _γ_ = 1 _._ 5 for 32 steps, as longer denoising trajectories require less stochastic correction. For the system-level comparison in Tab. 1, we use 64-step ODE sampling with time schedule. We set the self-conditioning CFG scale to 1 and the input-condition CFG scale to 2. 

24 

|**Model Architecture**|**Denoising and Decoding Confg**|
|---|---|
|||
|Model<br>ELF-B<br>Time schedule<br>logit normal<br>Model size<br>105M<br>Denoiser(_P_mean_, P_std)<br>(_−_1_._5_,_ 0_._8)<br>Encoder backbone<br>T5-small<br>Denoiser noise scale<br>2_._0<br>Embedding dimension<br>512<br>Decoder(_P_mean_, P_std)<br>(0_._8_,_ 0_._8)<br>Bottleneck dimension<br>128<br>Decoder noise scale<br>5_._0<br>Model dimension<br>768<br>Denoiser_vs._decoder prob.<br>0_._8_vs._0_._2<br>Sequence length<br>1024||
|**Conditioning and Guidance**<br>**Optimization and Training**||
|||
|Self-conditioning probability<br>0_._5<br>Self-conditioning CFG range<br>[0_._5_,_ 5]<br>Num. of time tokens<br>4<br>Num. of model-mode tokens<br>4<br>Num. of CFG tokens<br>4<br>SDE_γ_<br>1_._0|Optimizer<br>Muon<br>Learning rate<br>0_._002<br>Weight decay<br>0<br>Training epochs<br>5<br>Global batch size<br>512<br>Learning rate schedule<br>constant<br>Warmup epochs<br>0_._5<br>EMA decay<br>0_._9999<br>Training device<br>TPU v5p_×_64<br>Training time<br>1.5 h per epoch|



Table 4: **Default training hyperparameters** and setup for ELF-B on the OpenWebText dataset. Unless noted otherwise, all experiments in the paper follow this default configuration. 

|**Method**|**Base training**|**Distillation training**|**Effective tokens**|**Ratio**|
|---|---|---|---|---|
|MDLM [56]|512_×_1M_×_1024|-|524.3B|11.6_×_|
|Duo [57]|512_×_1M_×_1024|-|524.3B|11.6_×_|
|MDLM + SDTT [56]|512_×_1M_×_1024|512_×_10K_×_5_×_1024|550.5B|12.2_×_|
|Duo + DCD [57]|512_×_1M_×_1024|512_×_10K_×_5_×_1024|550.5B|12.2_×_|
|FLM [30]|512_×_1M_×_1024|-|524.3B|11.6_×_|
|FMLM [30]|512_×_1M_×_1024|512_×_100K_×_1024|576.7B|12.8_×_|
|LangFlow [10]|512_×_1M_×_1024|-|524.3B|11.6_×_|
|**ELF (ours)**|5_×_9_._04B|-|**45.2B**|**1.0**_×_|



Table 5: **Estimated effective training tokens** for ELF and the prior DLM baselines used in our systemlevel comparison (Fig. 7c). We estimate base-training tokens as batch size _×_ steps _×_ sequence length; distillation / flow-map stages are added on top where applicable. 

**Training-token budget for system-level comparison.** Tab. 5 reports the estimated effective training tokens used by ELF and each baseline in Fig. 7c. We estimate base-training tokens as batch size _×_ steps _×_ sequence length and add distillation or flow-map stages on top where applicable. The OWT dataset contains roughly 9.04B tokens. With our default training schedule of 5 epochs, ELF therefore uses 45.2B effective training tokens. Thus, ELF requires roughly an order of magnitude fewer effective training tokens than the compared DLMs. 

## **D.3 Ablation Studies Setting** 

We evaluate several choices of embedding representations for ELF, and report the implementation details as below. We also try two-stage training with a separate decoder. Unless specified, we keep other settings the same as the default ELF configuration. 

**Scratch encoder.** We train an encoder from scratch on OpenWebText [18] by following the original T5-small training pipeline [53]. The encoder is trained for 5 epochs with a learning rate of 1 _×_ 10 _[−]_[3] , cosine learning rate schedule, 0.4 epoch warmup, and a batch size of 512. During ELF training, we apply channel-wise normalization to the encoder outputs. 

25 

|**Steps**|**SC CFG**|_γ_|**Gen. **|**PPL**_↓_|**Entropy**_↑_|
|---|---|---|---|---|---|
|8|3|2.0|67_._32|_±_2_._25|5_._14_±_0_._085|
|16|3|2.0|33_._66|_±_1_._09|5_._16_±_0_._026|
|32|3|1.5|24_._08|_±_0_._16|5_._15_±_0_._002|



Table 6: **System-level ELF performance** reported as mean _±_ standard error (SE) over 6 independent evaluation runs (seeds 0–5; _n_ = 6). 

|**Sampler**<br>**SC CFG**|**ELF-B 105M**<br>**Gen. PPL**<br>**Entropy**|**ELF-M 342M**<br>**Gen. PPL**<br>**Entropy**|**ELF-L 652M**<br>**Gen. PPL**<br>**Entropy**|
|---|---|---|---|
|SDE<br>0.5<br>1.0<br>1.5<br>2.0<br>3.0<br>3.5<br>4.0|36.77<br>5.28<br>29.50<br>5.23<br>25.25<br>5.18<br>22.53<br>5.14<br>19.72<br>5.10<br>37.56<br>5.30<br>36.50<br>5.29|39.21<br>5.35<br>33.45<br>5.30<br>28.42<br>5.26<br>25.34<br>5.23<br>21.69<br>5.18<br>36.48<br>5.34<br>34.93<br>5.33|37.50<br>5.41<br>31.82<br>5.37<br>28.72<br>5.35<br>26.47<br>5.32<br>23.31<br>5.28<br>22.28<br>5.27<br>21.37<br>5.26|
|ODE<br>0.5<br>1.0<br>1.5<br>2.0<br>3.0|104.29<br>5.51<br>65.30<br>5.40<br>44.85<br>5.31<br>34.65<br>5.23<br>26.62<br>5.15|88.51<br>5.51<br>62.47<br>5.44<br>46.71<br>5.37<br>37.66<br>5.32<br>28.80<br>5.24|68.27<br>5.52<br>49.72<br>5.45<br>39.97<br>5.40<br>33.72<br>5.36<br>26.57<br>5.29|



Table 7: **Scaling performance** of generative perplexity (Gen. PPL) and unigram entropy for ELF models of different sizes under SDE and ODE samplers with 64 sampling steps. The effect of self-conditioning (SC) CFG scaling diminishes beyond 3. 

**Pretrained embedding layer.** We use the frozen embedding table from the T5-small encoder as the token embedding layer. The embedding layer matrix is normalized, and the unembedding layer is trained separately. 

**Gaussian embedding layer.** We randomly initialize and freeze an embedding layer from a Gaussian distribution, with token-wise embedding mean 0 and standard deviation 1. The unembedding layer is trained separately using the decoder mode. 

**Learnable embedding layer.** We jointly train the embedding layer together with the denoiser and decoder modes. The unembedding layer is tied with the embedding layer: denoiser-mode updates affect the embedding layer, while decoder-mode updates affect the unembedding layer. To stabilize training, we apply normalization directly on the unembedding layer matrix at every step. 

**Separate decoder.** For the separate-decoder setting, we use a randomly initialized decoder architecture obtained by mirroring the T5-small encoder. We keep the encoder fixed, mask 20% of the input tokens, and add logit-normal noise to the latent representations with _P_ mean = 0 _._ 5 and _P_ std = 1 _._ 0. The model is trained for 3 epochs with a learning rate of 3 _×_ 10 _[−]_[4] and a cosine learning-rate schedule. The relative noise scale with respect to the normalized latent representations is set to 5 _._ 0. 

## **D.4 Reported Numbers** 

**System level comparison.** Across 6 independent evaluation seeds, ELF shows highly consistent system-level behavior, as shown in Tab. 6. As the number of sampling steps increases from 8 to 32, the standard error (SE) decreases. The small standard errors—especially at 32 steps—suggest that these gains are robust to random seed variation and that the overall trend is reliable across runs. See Tab. 6 for detailed numbers. 

**Scaling behavior with CFG scales.** The default setting for both sampling methods uses 64 sampling steps with time schedule. For the SDE sampler, we set _γ_ = 1 _._ 0. The exact numbers are reported in 

26 

|**Confg**|**AR**|**MDLM**|**E2D2**|**Duo**|**Duo**|
|---|---|---|---|---|---|
|_Architecture_||||||
|Codebase|E2D2|E2D2|E2D2|Duo|Duo|
|Tokenizer|Qwen3-0.6B|Qwen3-0.6B|Qwen3-0.6B|T5-small|T5-small|
|Hidden Size|256|256|256|768|768|
|Intermediate Size|768|768|768|–|–|
|#Layers / Blocks|28|28|enc=20, dec=8|12|12|
|Sequence Length|64|64|64|64|64|
|Max Cond Length|1024|1024|1024|1024|64|
|Cond Embed|–|–|–|T5-small|T5-small|
|_Training_||||||
|Dataset|XSum|XSum|XSum|XSum|De-En|
|Learning Rate|3e-4|3e-4|3e-4|3e-4|3e-4|
|LR Scheduler|const|const|const|const|const|
|Warmup Steps|1000|1000|1000|2500|2500|
|Global Batch Size|128|128|128|512|512|
|Optimizer|DecoupledAdamW|DecoupledAdamW|DecoupledAdamW|AdamW|AdamW|
|Loss Type|NLL|MDLM ELBO|E2D2 ELBO|Duo ELBO|Duo ELBO|
|Train Steps|500K|500K|500K|1M|1M|
|_Evaluation_||||||
|Sampling Strategy|greedy|predict_and_noise|predict_and_noise|Duo sampler|Duo sampler|
|Sampling Steps|_L_= 64(AR)|_≈L_(frst-hit)|_≈L_(frst-hit)|1000|1000|
|Block size|1|32|8|-|-|
|CFG Scale|–|–|–|1.0|1.5|
|Checkpoint|best|best|best|best|best|
|EMA|true|true|true|true|true|



Table 8: **Detailed training and evaluation configurations for conditional generation tasks** of our reproduced AR, MDLM, E2D2, and Duo baselines. AR, MDLM, and E2D2 are reproduced on XSum using the E2D2 [4] codebase and follow the configurations reported in the E2D2 paper. For Duo, we build on the original Duo [57] repository, add cross-attention conditioning and CFG, adapt the T5-small encoder to match our setting, and tune the hyperparameters to obtain the strongest reproduced results. 

Tab. 7. Larger CFG scales improve generation quality by reducing Gen. PPL within a certain range. The effect of CFG scaling reverses beyond 3. Only ELF-L benefits from increasing the CFG scale from 3 to 4. Thus, in most default ablation studies, we only consider CFG scales from 0.5 to 3. 

## **D.5 Conditional Generation** 

Specifically, the WMT14 results for AR, MDLM, and E2D2 are taken from the E2D2 [4] paper, the SeqDiffuSeq result is taken from the LD4LG [41] paper, and the CDCD result is taken from the original CDCD [13] paper. For reproduced results, Duo [57] is implemented using the Duo codebase[4] , while AR, MDLM, and E2D2 are reproduced using the E2D2 codebase[5] . 

For a fair comparison, we reproduce all baselines using settings that are as close as possible to their original implementations, as summarized in Tab. 8. For AR, MDLM, and E2D2, we use the E2D2 codebase and follow the training and evaluation configurations reported in the E2D2 paper on XSum. Note that although E2D2 is primarily designed for semi-autoregressive generation, we find that MDLM also achieves its best performance under a semi-autoregressive setting ( _i.e._ , block size 32 with two-block generation); using single-block diffusion without semi-autoregressive generation degrades performance. For Duo, we start from the official Duo repository and adapt it to our conditional generation setting by adding cross-attention conditioning and classifier-free guidance, and by using a T5-small encoder for the conditioning input. During inference, we generate without 

> 4 `https://github.com/s-sahoo/duo` 

> 5 `https://github.com/kuleshov-group/e2d2` 

27 

t = 0 strength  will        building  building         building     building   back            played  playedband bit         choiceband  bitband  played  playedband  played  bit                    bit The         results   was        ab                 disturbing  .               EFuture        after    watching     various  games          ,              I              was            pretty   fierce               withLI The         results   were       flat                striking      .               Immediately  after   watched      the         games          ,             I              was             pretty   determined      with t = 1 The         results   were       particularly   striking      .               Immediately  after   watching      the         games          ,             I             was              very      concerned      about 

Figure 17: **Denoising trajectory** of ELF-B. As _t_ increases from 0 to 1, ungrammatical sentences are progressively refined into fluent and grammatical text. 

semi-autoregressive decoding. We tune the main sampling and guidance hyperparameters and report the best reproduced results we obtain. 

## **E Qualitative Examples** 

## **E.1 Denoising Trajectory** 

Fig. 17 visualizes the intermediate predictions along ELF’s denoising process. Starting from repetitive tokens at _t_ = 0, the model gradually forms semantically meaningful phrases, improves grammar, and refines word choices as _t_ approaches 1. This trajectory illustrates how continuous diffusion generation progressively transforms noisy embeddings that decode to gibberish text into clean embeddings that decode to grammatical sentences. 

## **E.2 Unconditional Generation Examples on OpenWebText** 

We provide three unconditional samples generated by ELF-B on OpenWebText, reported with their entropy and generative perplexity (Gen. PPL). The examples illustrate that ELF produces fluent, syntactically coherent, and topically consistent long-form text across diverse domains. 

## **ELF-B OWT** 

```
entropy:5.36Gen.PPL:21.04
```

```
ThecompanyhasbeendevelopingavirtualsleepmodeforitsiPhoneandiPad
foryears.Thismeansthatuserscanimprovetheirqualityoflifewithout
turningpingofftheirfingersthankstoGoogle’snewvirtualsleeptechnology.
Tomaketheexperienceareality,virtualsleepmodewasdevelopedforGoogle,
usinganewbuilt-intechnologythatincludesreal-timephotographyandshadow
monitoring.Thistechnologyenablesuserstohaveasafe,comfortablelookat
wheretheysleep,evenifyouplacethekeyboardorabuttonunderyour
fingers.SomesourcespointtotheiPhone6andiPhone6asyetanother
exampleoftheimportanceofvirtualsleepmodeinoureverydaylives.This
technologyhasbeenshowntobeusefulwhenstayingbusyontightdays,during
difficulttimesorlyingasleeponahotnight.Thistechnologycouldalsobe
usedtoimprovesleepqualityandhelpusersimprovequalityoflife.Editor’s
note:Thisposthasbeenupdatedtoanswertorelevantquestions.Googlesays
itwilladdvirtualsleepmodetoitsiPhoneandiPadincomingweek.Google
announcedsomegoodnewsThursdaymorning,includinginstructionsonwhento
eat,checkingout,wheretosleepandtherulessurroundingwhattoeat.The
companyreportedarevenueof\$957billion--morethanathirdofthetotal
revenueduringthesameperiod.Butthecompanydoesn’tseemtohaveaslewof
othergoodnewsyet,likethefirstone...
```

## **ELF-B OWT** 

```
entropy:5.27Gen.PPL:21.29
```

```
Balinsaidthepotentialcostofstartingthereisverylow,andhetoldUSA2
Networkinaninterviewthatheisnotonlyinterestedinexpandingthe
capacityoftheuniversity,butisalsointerestedinexpandingother
```

28 

```
services,includingstudentassistance,communityassistance,youth
assistance,youthassistance,andsocialjusticeassistance.Balinsaid,\"One
ofthethingsaboutthisisthatit’sdifficulttostart,becauseifyou’re
underfunded,you’regoingtoneedalltheservicesthatyouneed,andthat’s
whatyouhavetopayfor.Andit’sgoingtodifficult,ifnothing,foryouto
getthefundingyouneedtostartrightthere.\"TheUDUhasnotmadesuch
promise.\"Wehavealotoftheguysinthedepartmentthataredoingwell,a
lotoftheguysthataren’tdoingwellattheuniversityandthey’recurrently
underfunded,\"Balinsaid.\"MostoftheotherLHSuniversitiesacrossthe
countryarecurrentlyunderfunded.So,whatdoyouwantthemtodo?Youknow,
rightnow,thecostisverylowandtherearenogreatuniversitiesinthe
restofthecountry.It’snotgoingtobeeasy.\"Inthemeantime,Balinsaid,
theUDUisn’tlookingtoattracthigh-quality...
```

## **ELF-B OWT** 

## `entropy: 5.17 Gen. PPL: 21.80` 

```
Hey,IgrewupinLyndonintheearly’90sand,aftermyfather’sdeath,began
writingabookabouthimselfcalledTheLifeofSteveO’Malse.Afterhis
secondyearattheUniversityofChicagoO’Malsedecidedtowriteabiobio
abouthisfather.HefinishedhisstudyattheUniversityofChicagointhe
fallof1999.In2009hepublishedabiobiocalled"MyDadAndDaughterWhile
heWasHome,"asuccessfulbiobiowrittenbyaformermilitaryofficer,Lt.
Gen.DavidWilde.SteveO’Malsehashadgreatnationalsecurityexperience.
Throughouthiscareerasahigh-levelnationalsecurityadviser,hehasserved
asanadvisertoGeorgeH.W.BushandanadvisertotwotopFBIofficials,
JohnJ.TillersonandMichaelE.Comey,bothofwhomhavebeeninvolvedinthe
investigationsthatledtotheresignationofAttorneyGeneralRexTillerson.
Heplayedakeyrolethroughouttheadministrationasanationalsecurity
adviser,thenasaspecialadvisertoformerPresidentGeorgeW.Bush,thenas
secretaryofHomelandSecurityunderBushandformerpresidentialcandidate
RonaldClinton.In2008,O’MalsewasnamedbyPresidentRonaldReaganasanew
nationalsecurityadviser.Inaspeechlastyear,hedetailedhisexperience
intheReaganadministration,asanewnationalsecurityadviser.O’Malsesaid
hewassurprisedbyhisabilitytoexpresshisconcernsaboutnational
security,butaddedthathewouldbespeakingmoreforyearstocome...
```

## **E.3 Conditional Generation Examples** 

**WMT14 De-En qualitative examples.** We show qualitative examples on WMT14 De-En to complement the corpus-level BLEU results. ELF generally produces fluent and globally coherent translations. 

## **ELF-B WMT-DE-EN** 

```
<Originaltext>
```

```
DiesesPhänomenhatnachdenWahlenvomNovember2010anBedeutunggewonnen,
beidenen675neuerepublikanischeVertreterin26Staatenverzeichnetwerden
konnten.
```

```
<Referencetranslation>
```

```
ThisphenomenongainedmomentumfollowingtheNovember2010elections,which
saw675newRepublicanrepresentativesaddedin26States.
```

```
<Ourtranslation>
```

```
ThisphenomenonhasincreasedinsignificanceaftertheelectionsinNovember
2010,inwhich675newRepublicananrepresentativeshavebeenrecordedin26
countries.
```

29 

## **ELF-B WMT-DE-EN** 

```
<Originaltext>
```

```
ImGegensatzzuKanadasinddieUS-BundesstaatenfürdieDurchführungder
WahlenindeneinzelnenStaatenverantwortlich.
```

```
<Referencetranslation>
```

```
UnlikeinCanada,theAmericanStatesareresponsiblefortheorganisationof
federalelectionsintheUnitedStates.
```

```
<Ourtranslation>
```

```
UnlikeCanada,theUnitedStatesarestatesresponsibleforholdingelections
ineachcountry.
```

**XSum qualitative examples.** We show qualitative examples on XSum to complement the ROUGE results. ELF generally produces fluent and concise summaries that capture the main content of the source document. 

## **ELF-B XSum** 

```
<Originaltext>
```

```
Vogeswasforcedtoretirehurton86aftersufferingtheinjurywhilebatting
duringtheCountyChampionshipdrawwithSomerseton4June.
MiddlesexhopetohavetheAustralianbackfortheirT20Blastgameagainst
HampshireatLord’son3August.
The37-year-oldhasscored230runsinfourfirst-classgamesthisseasonat
anaverageof57.50.
"LosingAdamisnaturallyablowashecontributessignificantlytoeverything
wedo,"directorofcricketAngusFrasersaid.
```

```
"Hisabsence,however,doesgiveopportunitiestootherplayerswhoare
desperatetoplayinthefirstXI.
```

```
"InthepastwehavecopedwellwithoutanoverseasplayerandIexpectusto
dosonow."
```

```
DefendingcountychampionsMiddlesexaresixthintheDivisionOnetable,
havingdrawnallfouroftheirmatchesthisseason.
```

```
VogesretiredfrominternationalcricketinFebruarywithaTestbatting
averageof61.87from31innings,secondonlytoAustraliangreatSirDonald
Bradman’scareeraverageof99.94from52Tests.
```

```
<Referencesummarization>
```

```
MiddlesexbatsmanAdamVogeswillbeoutuntilAugustaftersufferingatorn
calfmuscleinhisrightleg.
```

```
<Oursummarization>
```

```
MiddlesexcaptainAdamVogeswillnotmisstherestoftheseasonashe
struggleswithaboneinjury.
```

## **ELF-B XSum** 

```
<Originaltext>
```

```
MsKendalltoldtheBBCLabourriskedsendinga"resignationlettertothe
Britishpeopleasaseriouspartyofgovernment"byelectingMrCorbyn.
Separately,MsCooperwarnedtherewasa"seriousriskthepartywillsplit"
iftheleft-wingerbecomesitsleader.
```

```
ItcomesasLabourbeginssendingoutthefirstballotpaperstovoters.
```

30 

```
Theresultofthecontestwillbeannouncedataspecialconferenceon12
September.
```

```
Morethan600,000peoplehavesigneduptovoteinthefour-waycontestbut
Labourhassaidapplicationsarestillbeingverified.
610,753
```

```
totalelectorate,thoughthismayfallaspartyremovesthosenotentitledto
vote
Ofwhich,fullpartymembers:299,755
Affiliatedtoatradeunion:189,703
RegisteredtovotebypayingGBP3:121,295
```

```
MeanwhilevotingintheelectionforthenewScottishLabourleaderendedat
midday.
```

```
MrCorbynisduetounveila10-pointpolicyplanwhileinGlasgowlater.
Thepopularityoftheleft-wingIslingtonNorthMP,whoispromising"anew
kindofpolitics",hassparkedarowaboutthefuturedirectionoftheLabour
party.
```

```
Anotherleadershipcontender,AndyBurnham,toldtheBBCMrCorbyn’spolicies
"lackcredibility".
```

```
"It’snotpossibletopromisefreeuniversityeducation,re-nationalisingthe
utilities,withoutthatcomingatagreatcostandifyoucan’texplainhow
thatisgoingtobepaidforthenIdon’tthinkwe’llwinbackthetrustof
votersontheeconomy,"hesaid.
```

```
BBCpoliticalcorrespondentRossHawkinssaidtherehadbeen"frustration"in
rivalcampswhoaccusedMrBurnhamofbeingreluctanttotakeonMrCorbyn.
Thisappearedtobehismostdirectattackyet,headded.
```

```
ButinaninterviewwithJeremyVineonBBCRadio2,MrBurnhamdeclinedto
followMsKendallandMsCooperandadvisehissupportersnottobackMr
Corbynwiththeirsecondandthirdpreferences.
```

```
Headded:"Peoplewillsayiftheyhearthingslikethat,’hangon,whatdo
youbelieve?’"
```

```
InaninterviewwithTheIndependent,MsKendallcalledforvoterstomarkMs
CooperorMrBurnhamassecondandthirdpreferences,andavoidgivingvotes
toMrCorbyn.
```

```
"IhavesetoutveryclearlywhereIdifferwithallthecandidatesbutour
differenceswithJeremy’skindofpoliticsarefargreater,"saidMsKendall.
SpeakingonBBCRadio4’sTodayprogrammeshesaidshe"can’tpretendtobe
agnostic"aboutavictoryforMrCorbyn,sayingofthevotingprocess:"Itis
analternativevotesystemandIwanttourgepartymemberstouseallof
theirdifferentpreferences.
```

```
"IwillbeusingmysecondandthirdpreferencesandIwouldurgeotherstodo
thesamebecauseIdon’twanttoseeourpartygobacktothepoliticsofthe
’80s,justbeingapartyofprotest."
```

```
TheLeicesterWestMPalsosaidshedidnotseethepartysplitting,asitdid
inthe1980swhenLabourmembersformedtheSocialDemocraticParty.
```

```
However,MsCoopertoldBBC2’sNewsnight:"Ithinkthereisaseriousrisk
thatthepartywillsplit,willpolariseandIcannotbeartoseethathappen
becausethereistoomuchatstake."
```

```
AskedinaninterviewwithgrassrootsLabourwebsiteLabourlistwhethervoters
shouldusetheirvotestotrytopreventMrCorbynwinning,shesaid:"Ithink
peopleshouldusealloftheirpreferences.
```

```
"AndIthinkthefocushastobehowdowemakesurewecanwinthatelection,
andthat’sthemostimportantthing-andIdon’tthinkJeremycandothat."
MrCorbynhaswarnedagainst"personalabuse"inthecampaign,sayinghewants
tofocusonpolicy.
```

```
Hispolicyprogrammeincludesacommitmentto"growthnotausterity",
nationalisingtherailwaysandenergysector,andaplanfornuclear
disarmament.
```

```
InanessayfortheFabianSocietyhealsosuggestedLabour’snewincreased
followingshouldbemoreinvolvedinthepartyandproposedareviewof
membershipfeestomakethepartymore"inclusive".
FormerPrimeMinisterGordonBrownisexpectedtojointhedebateoverthe
leadershipcontestwithaspeechonSunday,called"powerforapurpose-the
futureoftheLabourParty".
```

31 

```
LancePrice,formerdirectorofcommunicationsforLabour,toldtheBBCthe
contesthadbeenan"unedifyingmess"andhad"donenothingtoreengagethe
labourpartywiththosemillionsofpeoplewhodesertedit".
TheGuardiannewspaperhasendorsedMsCooperfortheleadershipwhilethe
DailyMirrorhasgivenitsbackingtoMrBurnham,althoughthepaperurgedhim
to"findarole"inhisteamforMrCorbyn,whoitsayshas"litupthe
electioncampaign".
LabourleadershiphopefulsLizKendallandYvetteCooperhavesaidtheir
supportersshouldbackanyoneotherthanJeremyCorbyninthecontest.
<Referencesummarization>
```

```
LabourleadershiphopefulsLizKendallandYvetteCooperhavesaidtheir
supportersshouldbackanyoneotherthanJeremyCorbyninthecontest.
<Oursummarization>
```

```
LabourleadershipcontenderKendallKendallhaswarnedshedoesnotavoidan
threatofusingJeremyCorbynwithsecondandthirdpreferences.
```

32 

