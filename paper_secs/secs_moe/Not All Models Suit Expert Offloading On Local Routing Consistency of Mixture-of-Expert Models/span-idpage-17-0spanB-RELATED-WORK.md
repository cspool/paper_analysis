# <span id="page-17-0"></span>B RELATED WORK

### B.1 MOE-BASED LLM AND EXPERT ANALYSIS

Since its introduction into large neural networks, MoE has become a critical strategy to build large language models up to trillions of parameters [\(Shazeer et al., 2017;](#page-14-2) [Fedus et al., 2022;](#page-12-0) [Rajbhandari](#page-14-3) [et al., 2022\)](#page-14-3). While some early models like SwitchTransformers [\(Fedus et al., 2022\)](#page-12-0) and NLLB [\(NLLB Team et al., 2022\)](#page-14-4) employ encoder-decoder structures as their backbone, due to the success of GPT-3, the most recent popular MoE-based LLMs use decoder-only structures [\(Jiang et al., 2024;](#page-12-1) [Yang et al., 2024a;](#page-15-5) [DeepSeek-AI et al., 2024b;](#page-11-0) [Abdin et al., 2024\)](#page-10-1), replacing their original FFN layers with MoE layers containing multiple experts (other components may be replaced too, e.g., self-attention [\(Shen et al., 2023;](#page-15-6) [2024b\)](#page-15-7) and LoRA [\(Li et al., 2024a;](#page-13-3) [Feng et al., 2024\)](#page-12-5)). [Cai et al.](#page-10-3) [\(2024\)](#page-10-3) systematically introduces MoE architectures and implementation in LLMs.

The popularity of MoE LLMs has triggered interest in understanding how experts are activated in such models. Many model reports and individual studies focused on the relation between expert selection and the input context. For example, [Muennighoff et al.](#page-14-1) [\(2025\)](#page-14-1) reported that OLMoE shows a significant difference in expert activity across different domains. Contrastively, [Xue et al.](#page-15-8) [\(2024a\)](#page-15-8) found that the routing choice of OpenMoE is highly related to the input token rather than the input context. Other works investigated the similarity among expert activation patterns [\(Li et al., 2024b;](#page-13-6) [Lu](#page-13-7) [et al., 2024\)](#page-13-7), as well as the relation between expert output and routing choice [\(Pham et al., 2024;](#page-14-5) [Lo](#page-13-8) [et al., 2025\)](#page-13-8). Some further proposed methods to reinforce such patterns [\(Guo et al., 2025;](#page-12-6) [Chen et al.,](#page-10-4) [2025\)](#page-10-4). However, few of them have focused on the local activation pattern of experts. For example, [Jiang et al.](#page-12-1) [\(2024\)](#page-12-1) reported that in Mixtral, experts are more likely to be activated consecutively, compared to the random case. While their results provide fundamental support for many efficient MoE inference systems [Liu et al.](#page-13-9) [\(2026\)](#page-13-9), they only examined the case of 2 consecutive tokens, which may be insufficient to ensure the consistency of expert activation in longer segments.

### B.2 EFFICIENT MOE INFERENCE AND EXPERT OFFLOADING

The discrete nature of routers and redundant parameters has caused MoE models to infer more slowly and consume more memory than dense models with the same number of activated parameters. Many techniques have been proposed to boost the inference of MoE models, ranging from model modifications like model compression [\(Chen et al., 2022;](#page-10-5) [Huang et al., 2025;](#page-12-7) [Yang et al., 2024b;](#page-16-5) [Rajbhandari et al., 2022\)](#page-14-3) and soft routing [\(Muqeeth et al., 2024;](#page-14-6) [Zhong et al., 2024\)](#page-16-6) to system implementations like load-balanced expert parallel [\(Lepikhin et al., 2021;](#page-13-5) [Huang et al., 2023;](#page-12-8) [Li](#page-13-10) [et al., 2023\)](#page-13-10) and hardware adaptation [\(DeepSeek-AI et al., 2024b;](#page-11-0) [Yi et al., 2025\)](#page-16-0). [Liu et al.](#page-13-9) [\(2026\)](#page-13-9) provides an in-depth summary of various inference optimization strategies of MoE models.

In this paper, we mainly focus on the potential performance of expert offloading, which enables lossless inference of MoE models on memory-constrained devices by caching only some experts on (fast) memory while leaving others on slow memory or disk storage. Many such systems use pretrained external models and/or information from previous layers to prefetch experts for upcoming layers [\(Ren et al., 2024;](#page-14-7) [Du et al., 2024;](#page-11-1) [He et al., 2024;](#page-12-9) [Song et al., 2024\)](#page-15-9). Various expert offloading systems propose curated heuristics to manage the expert cache [\(Skliar et al., 2025;](#page-15-2) [Xue et al., 2024b;](#page-15-1) [Yu et al., 2025;](#page-16-7) [Fang et al., 2025\)](#page-12-10). Among them, some examine the locality of expert activations as empirical support for expert caching efficiency:

• [Jiang et al.](#page-12-1) [\(2024\)](#page-12-1) first observed that Mixtral-8x7B is likely to choose the same expert for the next token, with probabilities higher than the random expectation. [Eliseev & Mazur](#page-12-2) [\(2023\)](#page-12-2) extended the argument to 2-4 consecutive tokens through a case study, and boosted inference performance of the same model with LRU caching (plus other techniques such as prefetching and quantization). These works align with our settings, but their analyses are limited to a single model (Mixtral-8x7B), short token spans, and lack a systematic and quantitative study.

• Xue et al. (2024b) reported frequent expert reuse during decoding, and observed that the reused experts depend on the prefilled input. Zhang et al. (2025) found similar routing choices between prefilling and decoding stages. Both observations are utilized to back the effectiveness of expert caching, yet are too coarse in terms of locality (at the input level instead of the token span level).

As more MoE LLMs emerge, understanding what models are more friendly to expert offloading becomes important for the development of both MoE architectures and expert offloading methods.

### C FORMAL DEFINITIONS AND PROOFS

#### <span id="page-18-0"></span>C.1 FORMAL DEFINITION OF SRP

**Single expert** For any input sequence  $T=[t_1,\ldots,t_{|T|}]$ , we denote the activation sequence of expert e on T as  $A(e,T)=[a_1,\ldots,a_{|T|}]$ , where  $a_i\in\{0,1\}$  indicates whether e is activated on  $t_i$  ( $a_i=1$ ) or not ( $a_i=0$ ). A segment-based router  $R_e^m$  with segment length m>0 for expert e will try to mimic  $[a_p,\ldots,a_{p+m-1}]$  for any T and p. Let  $R_e^m(T,p)=[b_1,\ldots,b_m]$  be its prediction, with the segment-prediction constraint:

<span id="page-18-2"></span>
$$b_i = 0, \forall i = 1, \dots, m \quad \text{or} \quad b_i = 1, \forall i = 1, \dots, m$$
 (2)

In other words,  $R_e^m$  decides that e is either always active or always inactive on  $[t_p, \ldots, t_{p+m-1}]$ . For simplicity, we write  $R_e^m(T,p)=0$  and  $R_e^m(T,p)=1$  for the two cases respectively. By treating each segment routing attempt as a binary classification task with m samples, and considering all possible segments of all possible inputs, we can calculate the  $F_1$  score of  $R_e^m$ :

<span id="page-18-1"></span>
$$F_1(R_e^m) = \frac{2\sum_T \sum_{p=1}^{|T|-m+1} R_e^m(T, p) \cdot f(e, T, p, m)}{\sum_T \sum_{p=1}^{|T|-m+1} \left[ m \cdot R_e^m(T, p) + f(e, T, p, m) \right]}$$
(3)

where  $f(e,T,p,m) = \sum_{i=p}^{p+m-1} A(e,T)[i]$  is the active frequency of e in the segment of T with length m starting at position p. We demonstrate the detailed process to obtain this equation in Appendix C.2. Based on Equation 3, we define the segment routing best performance of e under segment length m as the maximum  $F_1$  score any  $R_e^m$  can achieve:  $\mathrm{SRP}(e,m) \triangleq \max_{e} F_1(R_e^m)$ . Furthermore, in Appendix C.3 we prove that  $F_1(R_e^m)$  is maximized if and only if  $R_e^m$  gives active predictions for all segments that activates e at least  $\alpha_e^m$  times, where  $\alpha_e^m \in [0,m]$  is only related to e and e:

<span id="page-18-3"></span>
$$SRP(e, m) = \frac{2\sum_{T}\sum_{f(e, T, p, m) \ge \alpha_e^m} f(e, T, p, m)}{\sum_{T}\sum_{p=1}^{|T|-m+1} \left[m \cdot I[f(e, T, p, m) \ge \alpha_e^m] + f(e, T, p, m)\right]}$$
(4)

Therefore, SRP(e, m) is an intrinsic property of the expert that reflects its local routing consistency, unrelated to any specific segment routing methods.

**Expert group** For a group of experts E, let  $R_E^m$  be a segment-based router that decides whether each expert  $e \in E$  should be activated in a segment of some input T with length m; more specifically,  $R_E^m(e,T,p)$  is a prediction sequence similar to  $R_e^m(T,p)$  that also follows the segment-prediction constraint (Equation 2). Following the same procedure in Appendix C.2, we have

$$F_1(R_E^m) = \frac{2\sum_T \sum_{p=1}^{|T|-m+1} \sum_{e \in E} R_E^m(e, T, p) \cdot f(e, T, p, m)}{\sum_T \sum_{p=1}^{|T|-m+1} \sum_{e \in E} [m \cdot R_E^m(e, T, p) + f(e, T, p, m)]}$$
(5)

Again,  $F_1(R_E^m)$  is maximized if and only if  $R_E^m$  gives active predictions for all expert-segment pairs where the expert is activated at least  $\alpha_e^m$  times in the segment, where  $\alpha_e^m$  is decided by E and m. Therefore we have

$$SRP(E, m) \triangleq \max_{R_E^m} F_1(R_E^m) = \frac{2\sum_{T} \sum_{f(e, T, p, m) \ge \alpha_e^m} f(e, T, p, m)}{\sum_{T} \sum_{p=1}^{|T|-m+1} \sum_{e \in E} \left[ m \cdot I[f(e, T, p, m) \ge \alpha_e^m] + f(e, T, p, m) \right]}$$
(6)

SRP(E, m) measures how well a group of experts is coordinated by the original router(s) to achieve layer-level or model-level local routing consistency.

### <span id="page-19-1"></span>C.2 PROOF OF EQUATION 3

In Appendix C.1, we consider each routing decision of  $R_e^m$  for a segment of length m as a binary classification task with m samples. If we merge all samples from all segments of all possible inputs into one global binary classification task, and define  $f(e,T,p,m) = \sum_{i=p}^{p+m-1} A(e,T)[i]$  as in Appendix C.1, we will have the following prediction statistics:

$$TP(R_{e}^{m}) = \sum_{T} \sum_{p=1}^{|T|-m+1} \sum_{i=1}^{m} I[A(e,T)[p+i-1] = 1 \land R_{e}^{m}(T,p)[i] = 1]$$

$$= \sum_{T} \sum_{p=1}^{|T|-m+1} R_{e}^{m}(T,p) \cdot f(e,T,p,m)$$

$$FP(R_{e}^{m}) = \sum_{T} \sum_{p=1}^{|T|-m+1} \sum_{i=1}^{m} I[A(e,T)[p+i-1] = 0 \land R_{e}^{m}(T,p)[i] = 1]$$

$$= \sum_{T} \sum_{p=1}^{|T|-m+1} R_{e}^{m}(T,p)[m-f(e,T,p,m)]$$

$$FN(R_{e}^{m}) = \sum_{T} \sum_{p=1}^{|T|-m+1} \sum_{i=1}^{m} I[A(e,T)[p+i-1] = 1 \land R_{e}^{m}(T,p)[i] = 0]$$

$$= \sum_{T} \sum_{p=1}^{|T|-m+1} [1 - R_{e}^{m}(T,p)]f(e,T,p,m)$$

$$(9)$$

Therefore we have

$$F_{1}(R_{e}^{m}) = \frac{1}{1/\operatorname{Precision}(R_{e}^{m}) + 1/\operatorname{Recall}(R_{e}^{m})}$$

$$= \frac{1}{[TP(R_{e}^{m}) + FP(R_{e}^{m})]/TP(R_{e}^{m}) + [TP(R_{e}^{m}) + FN(R_{e}^{m})]/TP(R_{e}^{m})}$$

$$= \frac{2TP(R_{e}^{m})}{[TP(R_{e}^{m}) + FP(R_{e}^{m})] + [TP(R_{e}^{m}) + FN(R_{e}^{m})]}$$

$$= \frac{2\sum_{T} \sum_{p=1}^{|T|-m+1} R_{e}^{m}(T,p) \cdot f(e,T,p,m)}{\left[\sum_{T} \sum_{p=1}^{|T|-m+1} m \cdot R_{e}^{m}(T,p)\right] + \left[\sum_{T} \sum_{p=1}^{|T|-m+1} f(e,T,p,m)\right]}$$

$$= \frac{2\sum_{T} \sum_{p=1}^{|T|-m+1} R_{e}^{m}(T,p) \cdot f(e,T,p,m)}{\sum_{T} \sum_{p=1}^{|T|-m+1} [m \cdot R_{e}^{m}(T,p) + f(e,T,p,m)]}$$
(10)

which gives Equation 3.  $\square$ 

### <span id="page-19-0"></span>C.3 Proof of Equation 4

Assume that  $R_e^m(T_0, p_0) = 0$  for some  $e, m > 0, R_e^m, T_0$  and  $p_0$ , then we have

$$F_{1}(R_{e}^{m}) = \frac{2\sum_{T}\sum_{p=1}^{|T|-m+1}R_{e}^{m}(T,p) \cdot f(e,T,p,m)}{\sum_{T}\sum_{p=1}^{|T|-m+1}\left[m \cdot R_{e}^{m}(T,p) + f(e,T,p,m)\right]}$$

$$= \frac{2\sum_{T}\sum_{p=1}^{|T|-m+1}R_{e}^{m}(T,p) \cdot f(e,T,p,m)}{m\sum_{T}\sum_{p=1}^{|T|-m+1}R_{e}^{m}(T,p) + \sum_{T}\sum_{p=1}^{|T|-m+1}f(e,T,p,m)}$$

$$= \frac{2\sum_{T\neq T_{0}\wedge p\neq p_{0}}R_{e}^{m}(T,p) \cdot f(e,T,p,m)}{m\sum_{T\neq T_{0}\wedge p\neq p_{0}}R_{e}^{m}(T,p) + \sum_{T}\sum_{p=1}^{|T|-m+1}f(e,T,p,m)}$$

$$= \frac{2X}{mY+Z}$$
(11)

where

$$X = \sum_{\substack{T \neq T_0 \\ p \neq p_0}} R_e^m(T, p) \cdot f(e, T, p, m), \quad Y = \sum_{\substack{T \neq T_0 \\ p \neq p_0}} R_e^m(T, p), \quad Z = \sum_{T} \sum_{p=1}^{|T| - m + 1} f(e, T, p, m)$$

Let  $\widehat{R_e^m}$  be a copy of  $R_e^m$  except that  $R_e^m(T,p)=1$ ; all other routing decisions remain the same. Then the  $F_1$  score of the new segment router will be

$$F_{1}\left(\widetilde{R_{e}^{m}}\right) = \frac{2\sum_{T}\sum_{p=1}^{|T|-m+1}\widetilde{R_{e}^{m}}(T,p)\cdot f(e,T,p,m)}{\sum_{T}\sum_{p=1}^{|T|-m+1}\left[m\cdot\widetilde{R_{e}^{m}}(T,p)+f(e,T,p,m)\right]}$$

$$= \frac{2[X+f(e,T_{0},p_{0},m)]}{m(Y+1)+Z}$$

$$= \frac{(mY+Z)\cdot F_{1}(R_{e}^{m})+2f(e,T_{0},p_{0},m)}{m(Y+1)+Z}$$

$$= \frac{(mY+Z)\cdot F_{1}(R_{e}^{m})+m\cdot[2f(e,T_{0},p_{0},m)/m]}{(mY+Z)+m}$$
(12)

which is a weighted mean of  $F_1(R_e^m)$  and  $2f(e,T_0,p_0,m)/m$  with weights mY+Z and m. Note that  $Z=\sum_T\sum_{p=1}^{|T|-m+1}f(e,T,p,m)\geq 0$ , and Z=0 if and only if f(e,T,p,m)=0 for all T and p. If Z=0, then e is inactive everywhere and  $F_1(R_e^m)=0$  for any  $R_e^{m\$}$ , thus  $\mathrm{SRP}(e,m)=0$  and we can simply let  $\alpha_e^m=0$ . Therefore, we assume that Z>0, then both m and mY+Z are positive. Hence,  $F_1\left(\widehat{R_e^m}\right)\geq F_1(R_e^m)$  if and only if  $2f(e,T_0,p_0,m)/m\geq F_1(R_e^m)$ . Equality is achieved when and only when all equalities hold.

The above result indicates that, in order to increase  $F_1(R_e^m)$ , for any segment satisfying  $R_e^m(T,p)=0$  and  $f(e,T,p,m)\geq (m/2)\cdot F_1(R_e^m)$ , we should change the routing decision to  $R_e^m(T,p)=1^{\P}$ , and for any segment satisfying  $R_e^m(T,p)=1$  and  $f(e,T,p,m)<(m/2)\cdot F_1(R_e^m)$ , we should change the routing decision to  $R_e^m(T,p)=0$ . Under the case where the number of possible inputs is finite (which is the case for most LLMs due to their limited context windows), this will eventually result in a  $\widehat{R_e^m}$  that activates and only activates all segments with  $f(e,T,p,m)\geq (m/2)\cdot F_1\left(\widehat{R_e^m}\right)$ , whose  $F_1$  cannot increase further. Such  $\widehat{R_e^m}$  must be unique and maximizing  $F_1(R_e^m)$ : Otherwise, if there exists another  $\widehat{R_e^m}'$  with  $F_1\left(\widehat{R_e^m}'\right)>F_1\left(\widehat{R_e^m}\right)$ , then the only segments where  $\widehat{R_e^m}$  and  $\widehat{R_e^m}'$  disagree are the ones satisfying  $(m/2)\cdot F_1\left(\widehat{R_e^m}\right)\leq f(e,T,p,m)<(m/2)\cdot F_1\left(\widehat{R_e^m}'\right)$ , where  $\widehat{R_e^m}(T,p)=1$  and  $\widehat{R_e^m}'(T,p)=0$ ; however, changing  $\widehat{R_e^m}$  on these segments to 0 should not increase  $F_1\left(\widehat{R_e^m}\right)$ , thus  $F_1\left(\widehat{R_e^m}'\right)\leq F_1\left(\widehat{R_e^m}\right)$ , a contradiction. Therefore, we can let  $\alpha_{e,m}=\left\lceil F_1\left(\widehat{R_e^m}\right)\right\rceil$ , which yields Equation 4.  $\square$ 

#### D EXPERIMENT SETUP DETAILS

#### <span id="page-20-0"></span>D.1 REAL MODEL ARCHITECTURE LIST

Table 5 lists the detailed architecture and configuration of all REAL models where we conduct our experiments.

A few notes:

• SwitchTransformers-Base-128 and NLLB-MoE-54B are encoder-decoder models that use the T5 architecture. SwitchTransformers-Base-128 has 12 encoder layers and 12 decoder layers. NLLB-MoE-54B has 24 encoder layers and 24 decoder layers.

<span id="page-20-1"></span><sup>§</sup> If  $Y = R_e^m(T, p) = 0$  for all T and p, then  $F_1(R_e^m)$  is undefined, which we do not concern.

<span id="page-20-2"></span>When  $f(e, T, p, m) = (m/2) \cdot F_1(R_e^m)$ , changing  $R_e^m(T, p)$  does not affect  $F_1(R_e^m)$ .

<span id="page-21-1"></span>

| Table 5: REAL Model architecture and configuration, sorted by model size. Experts: T: total; A: |  |
|-------------------------------------------------------------------------------------------------|--|
| active; S: shared (not included in total).                                                      |  |

| Model                                  |       | # Params (B) | # Layers | MoE Layer |     | # Experts |   |  |
|----------------------------------------|-------|--------------|----------|-----------|-----|-----------|---|--|
|                                        |       | Active       |          |           | T   | A         | S |  |
| PowerMoE-3B (Shen et al., 2024c)       | 3.30  | 0.88         | 32       | all       | 40  | 8         | 0 |  |
| LLaMA-MoE-v1-3.5B (Zhu et al., 2024)   | 6.74  | 3.50         | 32       | all       | 16  | 4         | 0 |  |
| OLMoE-1B-7B-0125                       | 6.92  | 1.28         | 16       | all       | 64  | 8         | 0 |  |
| (Muennighoff et al., 2025)             |       |              |          |           |     |           |   |  |
| SwitchTransformers-Base-128            | 7.42  | 0.22         | 24       | every 2   | 128 | 1         | 0 |  |
| (Fedus et al., 2022)                   |       |              |          |           |     |           |   |  |
| LLaMA-MoE-v2-3.8B (Qu et al., 2024)    | 8.03  | 3.80         | 32       | all       | 8   | 2         | 0 |  |
| JetMoE-8B (Shen et al., 2024b)         | 8.52  | 2.33         | 24       | all       | 8   | 2         | 0 |  |
| OpenMoE-8B (Xue et al., 2024a)         | 11.86 | 3.80         | 24       | every 6   | 32  | 2         | 1 |  |
| MiniCPM-MoE-8x2B (Hu et al., 2024)     | 13.87 | 4.32         | 40       | all       | 8   | 2         | 0 |  |
| Qwen1.5-MoE-A2.7B (Qwen Team, 2024)    | 14.32 | 2.69         | 24       | all       | 60  | 4         | 4 |  |
| DeepSeek-V2-Lite                       | 15.71 | 2.66         | 27       | after 1st | 64  | 6         | 2 |  |
| (DeepSeek-AI et al., 2024a)            |       |              |          |           |     |           |   |  |
| DeepSeekMoE (Dai et al., 2024)         | 16.38 | 2.83         | 28       | after 1st | 64  | 6         | 2 |  |
| XVERSE-MoE-A4.2B                       | 25.78 | 4.23         | 28       | all       | 64  | 6         | 2 |  |
| (XVERSE Technology Inc., 2024)         |       |              |          |           |     |           |   |  |
| Qwen3-30B-A3B (Yang et al., 2025)      | 30.53 | 3.35         | 48       | all       | 128 | 8         | 0 |  |
| Yuan2.0-M32 (Wu et al., 2024)          | 39.94 | 3.70         | 24       | all       | 32  | 2         | 0 |  |
| Phi-3.5-MoE (Abdin et al., 2024)       | 41.87 | 6.64         | 32       | all       | 16  | 2         | 0 |  |
| GRIN-MoE (Liu et al., 2024)            | 41.87 | 6.64         | 32       | all       | 16  | 2         | 0 |  |
| Mixtral-8x7B-v0.1 (Jiang et al., 2024) | 46.70 | 12.88        | 32       | all       | 8   | 2         | 0 |  |
| Jamba-Mini-1.6 (Lenz et al., 2025)     | 51.57 | 12.11        | 32       | every 2   | 16  | 2         | 0 |  |
| NLLB-MoE-54B (NLLB Team et al., 2022)  | 54.50 | 3.75         | 48       | every 4   | 128 | 2         | 0 |  |
| Qwen2-57B-A14B (Yang et al., 2024a)    | 57.41 | 14.25        | 28       | all       | 64  | 8         | 8 |  |

- JetMoE-8B employs mixture-of-attention [\(Shen et al., 2024a\)](#page-14-10), which we keep intact in our experiments.
- GRIN-MoE shares the same architecture with Phi-3.5-MoE, but is trained using different methods.
- Jamba-Mini-1.6 employs a hybrid SSM-Transformer structure, yet the MoE part is identical to vanilla transformer-based MoE models.

### <span id="page-21-0"></span>D.2 TOY MODEL CONFIGURATIONS

To validate the potential factors that affects local routing consistency, we modify the configuration of OLMoE [\(Muennighoff et al., 2025\)](#page-14-1) and create a series of toy MoE models which we pretrain from scratch. The baseline model, Baseline, has only 8 layers and a hidden dimension of 1,280, compared to the original 16 layers and a hidden dimension of 2,048. Other architectural hyperparameters, such as the number of experts (activate 8 out of 64 experts) and the hidden dimension ratio between attention and expert (2:1, so Baseline is 640 and the original OLMoE is 1,024), are left intact. We sample 20B tokens from OLMoE's pretraining data, and pretrained the model on it for 10,000 steps; other training configurations such as sequence length (4,096), global batch size (1,024) and learning rate (cosine decay from 4e-4 to 5e-5) all follow the original pretraining stage settings.

Starting from the configuration of Baseline, we tweak one single setting once to create the following TOY models (including Baseline); all models have around 1.43B parameters, although they may activate a different number of parameters:

- FewerExp: Use 32 experts instead of 64, with doubled expert hidden dimension (1,280) and halved activated experts (4).
- ActMore: Activate 16 experts instead of 8, under the same total number of experts;
- ActFewer: Activate 2 experts instead of 8, under the same total number of experts;
- 1ShrExp: Replace an expert with a shared expert, so the router selects 7 out of 63 experts;
- 2ShrExp: Replace 2 experts with shared experts, so the router selects 6 out of 62 experts;
- DenseFst: Replace the first layer with a dense MLP layer, whose hidden dimension is the sum of *all* experts' hidden dimensions (40,960);

- DenseHlf: Replace the 1st, 3rd, 5th and 7th layers with dense MLP layers same as of DenseFst;
- NoLB: Adjust the load balance auxiliary loss coefficient from 0.01 to 0 (no regularization);
- OverLB: Adjust the load balance auxiliary loss coefficient from 0.01 to 0.1 (over regularization).

### <span id="page-22-1"></span>D.3 DATA PROCESSING AND INPUT GENERATION

We first extract samples from RedPajama and downstream application datasets in plain text format. For RedPajama, this is already done. For LMArena, each of the original instances consists of two human-LLM conversations and a preference vote. We keep the instances where one of the conversations is preferred and concatenate all rounds from the preferred conversation (each round with its role and content) into one document. For OpenMath, OpenCode, and OpenScience, we simply concatenate the input and output of each instance.

After collecting samples from each RedPajama category and the downstream application dataset, we concatenate the samples within the domain, cutting them into input sequences of 512 tokens (the context window size of SwitchTransformers). We sample 2,048 input sequences for each domain, resulting in 22,528 input samples in total.

For SwitchTransformers, since the model is trained for masked language modeling, we randomly select 64 tokens from each input sequence, masking them in the original sequence as the encoder input and constructing the corresponding label sequence as the decoder input. For NLLB-MoE, as the model is trained for machine translation, we use the same sequence (with the English language token prepended) as both the encoder input and the decoder input. All other models do not need further data preprocessing, as they are decoder-only and trained for next token prediction.

