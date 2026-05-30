# 5 Conclusion

In this paper, we propose a low overhead structure named LocMoE to relieve the performance bottleneck of existing MoE. The modifications mainly revolve around the mechanism of token assignment. The locality loss, which can be delineated as the distribution difference of token assignments, is proposed to promote locality computation on the premise of load balance. We also provide the theoretical demonstration for the lower bound of the expert capacity to achieve the same effect by training fewer tokens. To meet the assumption of orthogonal gating weight, the GrAP layer is adopted instead of the dense layer to calculate the gating values, and it can also reduce the overhead of computation. Incorporating group-wise All-to-All and communication overlapping features, the elapsed time of communication is further reduced. The experiments are performed on Ascend clusters with 64, 128, and 256 910A NPUs. Compared with current state-ofthe-art MoEs, the performance improvement of training is up to 22.24%. Evaluating multiple NLP tasks, it is detected that the interactive capability of our model is also enhanced. From the results that explore the relationship between the scale of expert capacity and the token features, we find that the dataset construction still needs to be improved. In future work, we will further organize the multilingual corpora from more fields.

### **Appendix**

#### A. Proof Sketch in 3.2

#### A.1 Proof for Lemma 1

*Proof.* According to the previous definition,  $\delta_{i^*,j} = \cos(\theta_{i^*,j})$ , where  $i^*$  is the expert that the token j routed to.  $\theta_{i^*,j}$  is the angle between token j and the gating weight  $\omega_{i^*}$  corresponding to the expert  $i^*$ . Combined with Formula (3) and (4) in Section 3.2, we have:

$$i^* = \underset{i \in [n]}{\arg \max} (\langle \omega_i, x_m \rangle)$$
where  $\langle \omega_i, x_m \rangle = \|\omega_i\| \cdot \|x_m\| \cdot \cos(\theta_{i^*,j})$ 

$$i^* = \underset{i \in [n]}{\arg \max} (\langle \omega_i, x_m \rangle)$$

$$= \underset{i \in [n]}{\arg \max} (\delta_{i^*,j})$$
(10)

#### A.2 Proof for Lemma 3

*Proof.* The area of a hyperspherical cap in a n-sphere of radius r can be obtained by integrating the surface area of an (n-1)-sphere of radius  $r\sin\theta$  with arc element  $r\mathrm{d}\theta$  over a great circle arc, that is:

$$A_{n}^{\text{cap}}(r) = \int_{0}^{\phi} A_{n-1}(r\sin\theta)rd\theta$$

$$= \frac{2\pi^{(n-1)/2}}{\Gamma\left(\frac{n-1}{2}\right)}r^{n-1}\int_{0}^{\phi}\sin^{n-2}\theta d\theta$$

$$= \frac{2\pi^{(n-1)/2}}{\Gamma\left(\frac{n-1}{2}\right)}r^{n-1}J_{n-2}(\phi)$$

$$= \frac{2\pi^{(n-1)/2}}{\Gamma\left(\frac{n-1}{2}\right)}r^{n-1}\frac{1}{2}B\left(\frac{n-1}{2},\frac{1}{2}\right)I_{\sin^{2}\phi}\left(\frac{n-1}{2},\frac{1}{2}\right)$$

$$= \frac{1}{2}\frac{2\pi^{(n-1)/2}}{\Gamma\left(\frac{n-1}{2}\right)}r^{n-1}\frac{\Gamma\left(\frac{n-1}{2}\right)\Gamma\left(\frac{1}{2}\right)}{\Gamma\left(\frac{n}{2}\right)}I_{\sin^{2}\phi}\left(\frac{n-1}{2},\frac{1}{2}\right)$$

$$= \frac{1}{2}\frac{2\pi^{n/2}}{\Gamma\left(\frac{n}{2}\right)}r^{n-1}I_{\sin^{2}\phi}\left(\frac{n-1}{2},\frac{1}{2}\right)$$

$$= \frac{1}{2}A_{n}(r)I_{\sin^{2}\phi}\left(\frac{n-1}{2},\frac{1}{2}\right)$$
(11)

where  $A_n(r)$  denotes the area of the high-dimensional sphere.  $p_\delta$  can be viewed as the proportion of the symmetrical areas formed by  $\theta$  to that of the entire sphere, shown as Figure 15:

$$p_{\delta} = \frac{2A_n^{\text{cap}}(r,\theta)}{A_n(r)}$$

$$= I_{1-\delta^2}\left(\frac{d-1}{2}, \frac{1}{2}\right)$$

$$= 1 - I_{\delta^2}\left(\frac{1}{2}, \frac{d-1}{2}\right)$$
(12)

<span id="page-7-0"></span>![](_page_7_Picture_10.jpeg)

Figure 15: The schematic of  $p_{\delta}$ 

Suppose  $\delta=\sqrt{\frac{1}{d-\frac{3}{2}}}$ , when d is large,  $\delta$  approximates to  $\sqrt{\frac{1}{d}}$ , then:

$$\begin{split} I_{\delta^{2}}(\frac{1}{2}, \frac{d-1}{2}) &\approx I(\frac{\delta^{2}(d-1+\frac{1}{2}-1)}{2-\delta^{2}}, \frac{1}{2}) + \Theta[(\frac{d-1}{2})^{-2}] \\ &\approx I(\frac{\frac{1}{d-\frac{3}{2}}(d-\frac{3}{2})}{2-\frac{1}{d-\frac{3}{2}}}, \frac{1}{2}) \\ &= I(\frac{1}{2}(\frac{d-\frac{3}{2}}{d-2}), \frac{1}{2}) \\ &= \frac{1}{\Gamma(\frac{1}{2})} \int_{0}^{\frac{1}{2}(\frac{d-\frac{3}{2}}{d-2})} \exp(-t)t^{\frac{1}{2}} dt \\ &\approx \frac{1}{\Gamma(\frac{1}{2})} \int_{0}^{\frac{1}{2}} e^{-t}t^{-\frac{1}{2}} dt \\ &= \frac{1}{\Gamma(\frac{1}{2})} \gamma(\frac{1}{2}, \frac{1}{2}) \\ &= \operatorname{erf}(\frac{\sqrt{2}}{2}) \end{split}$$

where  $\gamma$  is the incomplete gamma function. Combined with Formula (2) in Section 3.2,  $\operatorname{erf}(\frac{\sqrt{2}}{2}) \approx 0.68$ , then:

$$p_{\delta} = 1 - I_{\delta^2}(\frac{1}{2}, \frac{d-1}{2}) \approx 0.3$$
 (14)

### A.3 Proof for Theorem 1

*Proof.* Refer to the assumption about distributions of class-discriminative and class-irrelevant patterns in pMoE [Chowdhury *et al.*, 2023], with analogy, the tokens satisfy  $\delta_{i,j} \geq \delta$  can be regarded as the class-discriminative token. Then, the problem we need to explore can be converted to find the minimum amount of tokens that make at least one class-discriminative token routed to expert i.

Suppose  $p_i$  is the probability that the token routed to the expert i is a class-discriminative token; we have:

<span id="page-8-0"></span>

| Hyperparameter              | Description                                                                                     | Value |
|-----------------------------|-------------------------------------------------------------------------------------------------|-------|
| adam eps                    | Terms to increase the stability of numerical calculations                                       | 1e-6  |
| batch size                  | The size of data input to the model for training each time, related<br>to the number of devices | 32    |
| expert num per dp dim       | Number of experts per communication group                                                       | 1     |
| expert parallel             | Number of experts in parallel                                                                   | 16    |
| moe layer num               | Number of MoE layers                                                                            | 8     |
| num heads                   | Number of parallel heads                                                                        | 40    |
| op level model parallel num | Number of parallel models                                                                       | 8     |
| sink size                   | The size of data executed per sink                                                              | 16    |

Table 1: The critical hyperparameters in configuration of PanGu-Σ.

$$p_i \le \frac{p_\delta}{\frac{1}{n}} = n[1 - I_{\delta^2}(\frac{1}{2}, \frac{d-1}{2})]$$
 (15)

where the first inequality holds since the token satisfies δi,j ≥ δ may not always be routed to expert i. Then, the minimum value of expert capacity under the circumstance of at least one class-discriminative token routed to expert i can be written as:

$$ec_{\min} = \frac{1}{p_{s}} = \frac{1}{n[1 - I_{\delta^{2}}(\frac{1}{2}, \frac{d-1}{2})]}$$
For large  $d$ ,  $I_{\delta^{2}}(\frac{1}{2}, \frac{d-1}{2}) \approx I(\frac{\delta^{2}(d - \frac{3}{2})}{2 - \delta^{2}}, \frac{1}{2})$ 

$$\approx \frac{1}{\Gamma(\frac{1}{2})}\gamma(\frac{1}{2}, \frac{\delta^{2}d}{2 - \delta^{2}})$$

$$= 1 - \operatorname{erfc}(\sqrt{\frac{\delta^{2}d}{2 - \delta^{2}}})$$
thus,  $ec_{\min} \ge \frac{1}{n \cdot \operatorname{erfc}(\sqrt{\frac{\delta^{2}d}{2 - \delta^{2}}})}$ 

$$> \frac{1}{n} \exp(\frac{\delta^{2}d}{2 - \delta^{2}})$$

### B. Datasets

PanGu-Σ has already demonstrated its ability to learn efficiently and independently from text corpus in various domains. In this work, we will evaluate the performance of PanGu-Σ in detailed knowledge of a specific area. The materials connected to mobile network operators' services are chosen as input corpora. Concretely, blogs and technical documents in the form of *iCase*, *Wiki*, core network/Man-Machine language (MML), configuration translations, feature documents, etc., are collected. These corpora are in Chinese, English, or bilingual (Chinese-English).

Among them, *iCase* indicates the technology case, which records procedures of problem handling and contributes to problem delimitation and localization. *iCase* contents include the wireless network, optical, carrier IT, cloud core network, network energy, etc. It contains code of Java, SQL, Shell, other programming languages or commands, and the

related logs, totaling 591,972 documents (368,282 Chinese, 223,690 English, 1.7GB) and 387,223,874 tokens. *Wiki* is the document extracted from 3ms (Huawei's internal knowledge management platform). Topics of Wiki include insight reports, R&D tool guides, training summaries, industry standards, configuration manuals, etc., totaling 1,146,755 documents (1,118,669 in Chinese, 27,632 in English, and 454 bilingual, 4.1 GB) and 116,152,3537 tokens. The corpora in the field of core network and MML are mainly derived from the product information from mobile network operators or public platforms, such as 3GPP protocols, customized specifications, high-quality MO Support Processes (MOP), engineering solutions, and MML scripts for existing networks, totaling 223,898 documents (all in Chinese, 0.476GB) and 136908105 tokens. Configuration translation data come from product documents for data communication equipment of Huawei or Cisco involving switches, firewalls, and routers, totaling 1460680 documents (all in Chinese, 2.2 GB) and 559716720 tokens. Feature documents include product design documents for data communication, IT and other business lines, 4G/5G feature documents, the frequently asked question (FAQ) of machine question and answering (Q&A), fault trees, fault location guides, etc., totaling 86,913 documents (52,677 in Chinese, 34,236 in English, 0.29GB).

The above corpora are in different formats: Word, PDF, HDX, and HTML. First of all, the original corpora need to be parsed. For instance, The text of a PDF document is extracted with the pattern recognition technique, and the machine Q&A corpus is manually entered by iCare engineers. After that, the fine-grained corpora are merged and organized into a complete sample to ensure a complete thought chain. Taking MML scripts as an example, their structuredness is divided into three levels from global to local: (1) Features composed of medium features; (2) Medium features composed of multiple ordered MMLs; (3) MML instances. Product documents can uniquely identify medium features, and the diversity of MML instances can be constructed from the present network's MMLs. The corpora are refined; that is, after removing meaningless symbols and descriptions, duplication elimination is performed on the corpora based on text similarity and semantics to avoid overlapping data. The next step is to regularize the data, including removing private data and unifying the specification of forms and process symbols. Finally, a customized tokenizer based on the domain dictionary is applied to the participle, and the cleaned corpora are obtained for training.

### C. Experimental Environment

The experiments are conducted on Ascend clusters, and the environment falls into three groups: 64, 128, and 256 Ascend 910A NPUs. The Ascend 910A series NPU has 32 AI Cores, with a maximum memory capacity of 2TB and a maximum memory bandwidth of 1.07TB/s. The collective communication function on high-speed links such as PCI-E, HCCS, and RoCE is realized by HCCL, a high-performance collective communication library based on the Ascend. It provides communication primitives on single-node-multi-card and multinode-multi-card, and it also supports various communication algorithms such as ring, mesh, HD, ring + HD, and mesh + HD.

The versions of the Compute Architecture for Neural Networks (CANN) suite (toolkit, CANN, driver) are 5.1.RC2.1, 1.84, and 23.0.rc2, respectively. The CANN is the heterogeneous computing architecture developed by Huawei, and it supports multiple AI frameworks, including MindSpore, Py-Torch, TensorFlow, etc., providing interfaces to build AI applications on the Ascend platform. Our model runs on the MindSpore framework with version 2.0.0.

### D. Model Configuration

The hyperparameter configuration of our model is listed in Table [1.](#page-8-0) Thereinto, *batch size* and *sink size* are relevant to the number of devices, and the values in the table are under 128N. The total number of experts can be obtained by *expert num per dp dim* \* *expert parallel*.

