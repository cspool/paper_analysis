# <span id="page-13-0"></span>A Proofs

<span id="page-13-1"></span>We use the following lemma (simplified from [\[27\]](#page-11-3), Lemma 14) without proof.

Lemma A.1. *For every non-negative sequence* {rt}t≥<sup>0</sup> *and any parameters* a ≥ 0*,* c ≥ 0*,* T ≥ 0*, there exists a constant* η ≤ a *, such that*

$$\frac{1}{T+1} \sum_{t=0}^{T} \left( \frac{r_t - r_{t+1}}{\eta} + c\eta \right) = \frac{1}{T+1} \frac{r_0 - r_{T+1}}{\eta} + c\eta \le \frac{ar_0}{T+1} + \frac{2\sqrt{cr_0}}{\sqrt{T+1}}.$$

Theorem 4.1 (Convergence error bound). *For arbitrary non-convex function under Assumption [4.1](#page-6-0) and Assumption [4.2,](#page-6-1) taking learning rate* η ≤ 1 10L( 2 <sup>δ</sup> <sup>+</sup>ρκ+ρ+κ) *, Algorithm [4](#page-5-2) converges to a critical point with the following error bound:*

$$\frac{\sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_{t})\|^{2}]}{T+1} \leq \frac{80L\left(\frac{2}{\delta} + \rho\kappa + \rho + \kappa\right)\left(f(w_{0}) - f^{*}\right)}{T+1} + 4\sigma\sqrt{\frac{(11-\delta)(\kappa+1)L(f(w_{0}) - f^{*})}{T+1}}.$$

*Proof.* By using smoothness (Assumption [4.1\)](#page-6-0), we have

$$f(w_{t+1}) \le f(w_t) - \eta \langle \nabla f(w_t), \mathcal{U}_g(g_t) \rangle + \frac{\eta^2 L}{2} \|\mathcal{U}_g(g_t)\|^2.$$

Taking expectation w.r.t. the random compressor Ug, we have

$$\begin{split} & \mathbb{E}_{gc}[f(w_{t+1})] \\ & \leq f(w_t) - \eta \left\langle \nabla f(w_t), g_t \right\rangle + \frac{\eta^2 L}{2} \mathbb{E}_{gc} \|\mathcal{U}_g(g_t)\|^2 \\ & = f(w_t) - \eta \left\langle \nabla f(w_t), g_t \right\rangle + \frac{\eta^2 L}{2} [\|g_t\|^2 + \mathbb{E}_{gc} \|\mathcal{U}_g(g_t) - g_t\|^2] \\ & \leq f(w_t) - \eta \left\langle \nabla f(w_t), g_t \right\rangle + \frac{\eta^2 L(\kappa + 1)}{2} \|g_t\|^2. \end{split}$$

Conditional on wt, taking expectation on the random sample ζt, we have

$$\begin{split} &\mathbb{E}_{\zeta}[\mathbb{E}_{gc}[f(w_{t+1})]] \\ &\leq f(w_{t}) - \eta \left\langle \nabla f(w_{t}), \nabla f(\tilde{w}_{t}) \right\rangle + \frac{\eta^{2}L(\kappa+1)}{2} \mathbb{E}_{\zeta} \|g_{t}\|^{2} \\ &= f(w_{t}) - \eta \left\langle \nabla f(w_{t}), \nabla f(\tilde{w}_{t}) \right\rangle + \frac{\eta^{2}L(\kappa+1)}{2} \mathbb{E}_{\zeta} \|g_{t} - \nabla f(\tilde{w}_{t}) + \nabla f(\tilde{w}_{t}) \|^{2} \\ &= f(w_{t}) - \eta \left\langle \nabla f(w_{t}), \nabla f(\tilde{w}_{t}) \right\rangle + \frac{\eta^{2}L(\kappa+1)}{2} \mathbb{E}_{\zeta} [\|g_{t} - \nabla f(\tilde{w}_{t}) + \nabla f(\tilde{w}_{t}) \|^{2} ] \\ &\leq f(w_{t}) - \eta \left\langle \nabla f(w_{t}), \nabla f(\tilde{w}_{t}) \right\rangle + \frac{\eta^{2}L(\kappa+1)(\rho+1)}{2} \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &\leq f(w_{t}) - \eta \left\langle \nabla f(w_{t}), \nabla f(\tilde{w}_{t}) \right\rangle + \frac{\eta^{2}L(\kappa+1)(\rho+1)}{2} \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &= f(w_{t}) - \eta \left\langle \nabla f(w_{t}) - \nabla f(\tilde{w}_{t}) + \nabla f(\tilde{w}_{t}), \nabla f(\tilde{w}_{t}) \right\rangle + \frac{\eta^{2}L(\kappa+1)(\rho+1)}{2} \|\nabla f(\tilde{w}_{t})\|^{2} \\ &+ \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &= f(w_{t}) - \eta \left\langle \nabla f(w_{t}) - \nabla f(\tilde{w}_{t}), \nabla f(\tilde{w}_{t}) \right\rangle - \eta \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)(\rho+1)}{2} \|\nabla f(\tilde{w}_{t})\|^{2} \\ &+ \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &= f(w_{t}) - \eta \left(1 - \frac{\eta L(\kappa+1)(\rho+1)}{2}\right) \|\nabla f(\tilde{w}_{t})\|^{2} - \eta \left\langle \nabla f(w_{t}) - \nabla f(\tilde{w}_{t}), \nabla f(\tilde{w}_{t}) \right\rangle \\ &+ \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &\leq f(w_{t}) - \eta \left(1 - \frac{\eta L(\kappa+1)(\rho+1)}{2}\right) \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta}{2} \|\nabla f(w_{t}) - \nabla f(\tilde{w}_{t})\|^{2} \\ &+ \frac{\eta}{2} \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &\leq f(w_{t}) - \eta \left(1 - \frac{\eta L(\kappa+1)(\rho+1)}{2}\right) \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta}{2} \|\nabla f(w_{t}) - \nabla f(\tilde{w}_{t})\|^{2} \\ &+ \frac{\eta}{2} \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &\leq f(w_{t}) - \eta \left(1 - \frac{\eta L(\kappa+1)(\rho+1)}{2}\right) \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta}{2} \|\nabla f(w_{t}) - \nabla f(\tilde{w}_{t})\|^{2} \\ &+ \frac{\eta}{2} \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2} \\ &\leq f(w_{t}) - \frac{\eta}{2} [1 - \eta L(\kappa+1)(\rho+1)] \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta}{2} \|\nabla f(w_{t}) - \nabla f(\tilde{w}_{t})\|^{2} + \frac{\eta^{2}L(\kappa+1)\sigma^{2}}{2}. \end{cases}$$

Again using smoothness, and taking η ≤ <sup>2</sup>L(ρ+1)(κ+1) , we have − η 2 [1 − ηL(κ + 1)(ρ + 1)] ≤ −<sup>η</sup> 4 , and, we have

$$\begin{split} &\mathbb{E}_{\zeta}[\mathbb{E}_{gc}[f(w_{t+1})]] \\ &\leq f(w_t) - \frac{\eta}{2} \left[ 1 - \eta L(\kappa + 1)(\rho + 1) \right] \|\nabla f(\tilde{w}_t)\|^2 + \frac{\eta L^2}{2} \|w_t - \tilde{w}_t\|^2 + \frac{\eta^2 L(\kappa + 1)\sigma^2}{2} \\ &\leq f(w_t) - \frac{\eta}{2} \left[ 1 - \eta L(\kappa + 1)(\rho + 1) \right] \|\nabla f(\tilde{w}_t)\|^2 + \frac{\eta L^2}{2} \|e_t\|^2 + \frac{\eta^2 L(\kappa + 1)\sigma^2}{2} \\ &\leq f(w_t) - \frac{\eta}{4} \|\nabla f(\tilde{w}_t)\|^2 + \frac{\eta L^2}{2} \|e_t\|^2 + \frac{\eta^2 L(\kappa + 1)\sigma^2}{2}, \end{split}$$

where we define the sequence

$$e_t = w_t - \tilde{w}_t, e_0 = 0.$$

Now we establish the upper bound of the sequence ∥et∥ 2 as follows.

First, using wt+1 = wt−ηUg(gt) and w˜t+1 = ˜wt+Cw(wt+1−w˜t), we have the following equations:

$$w_{t+1} - \tilde{w}_{t+1} = e_{t+1} = w_t - \tilde{w}_t - \eta \mathcal{U}_g(g_t) - \mathcal{C}_w(w_{t+1} - \tilde{w}_t) = e_t - \eta \mathcal{U}_g(g_t) - \mathcal{C}_w(e_t - \eta \mathcal{U}_g(g_t))$$

Taking expectation w.r.t. the random compressor Cw, we have

$$\begin{split} & \mathbb{E}_{wc}[\|e_{t+1}\|^2] \\ & = \mathbb{E}_{wc}[\|e_t - \eta \mathcal{U}_g(g_t) - \mathcal{C}_w(e_t - \eta \mathcal{U}_g(g_t))\|^2] \\ & \leq (1 - \delta)\|e_t - \eta \mathcal{U}_g(g_t)\|^2. \end{split}$$

Taking expectation w.r.t. the random compressor Ug, we have

$$\begin{split} & \mathbb{E}_{gc}[\mathbb{E}_{wc}[\|e_{t+1}\|^{2}]] \\ & \leq (1 - \delta)\mathbb{E}_{gc}[\|e_{t} - \eta \mathcal{U}_{g}(g_{t})\|^{2}] \\ & = (1 - \delta)\mathbb{E}_{gc}[\|e_{t} - \eta g_{t} + \eta g_{t} - \eta \mathcal{U}_{g}(g_{t})\|^{2}] \\ & = (1 - \delta)\|e_{t} - \eta g_{t}\|^{2} + (1 - \delta)\eta^{2}\mathbb{E}_{gc}[\|g_{t} - \mathcal{U}_{g}(g_{t})\|^{2}] \\ & \leq (1 - \delta)\|e_{t} - \eta g_{t}\|^{2} + (1 - \delta)\eta^{2}\kappa\|g_{t}\|^{2}. \end{split}$$

Conditional on wt, taking expectation on the random sample ζt, we have

$$\mathbb{E}_{\zeta}[\mathbb{E}_{gc}[\mathbb{E}_{wc}[\|e_{t+1}\|^{2}]]] \\
\leq (1 - \delta)\mathbb{E}_{\zeta}[\|e_{t} - \eta \nabla f(\tilde{w}_{t}) + \eta \nabla f(\tilde{w}_{t}) - \eta g_{t}\|^{2}] + (1 - \delta)\eta^{2}\kappa\mathbb{E}_{\zeta}[\|g_{t} - \nabla f(\tilde{w}_{t}) + \nabla f(\tilde{w}_{t})\|^{2}] \\
= (1 - \delta)\|e_{t} - \eta \nabla f(\tilde{w}_{t})\|^{2} + (1 - \delta)(\kappa + 1)\eta^{2}\mathbb{E}_{\zeta}[\|g_{t} - \nabla f(\tilde{w}_{t})\|^{2}] + (1 - \delta)\eta^{2}\kappa\|\nabla f(\tilde{w}_{t})\|^{2} \\
\leq (1 - \delta)\|e_{t} - \eta \nabla f(\tilde{w}_{t})\|^{2} + (1 - \delta)(\kappa + 1)\eta^{2}(\rho\|\nabla f(\tilde{w}_{t})\|^{2} + \sigma^{2}) + (1 - \delta)\eta^{2}\kappa\|\nabla f(\tilde{w}_{t})\|^{2} \\
= (1 - \delta)\|e_{t} - \eta \nabla f(\tilde{w}_{t})\|^{2} + (1 - \delta)\eta^{2}(\rho\kappa + \rho + \kappa)\|\nabla f(\tilde{w}_{t})\|^{2} + (1 - \delta)(\kappa + 1)\eta^{2}\sigma^{2}.$$

With ∀b > 0, we have

$$\mathbb{E}_{\zeta}[\mathbb{E}_{gc}[\mathbb{E}_{wc}[\|e_{t+1}\|^{2}]]] \\
\leq (1-\delta)(1+b)\|e_{t}\|^{2} + (1-\delta)(1+b^{-1})\|\eta\nabla f(\tilde{w}_{t})\|^{2} + (1-\delta)\eta^{2}(\rho\kappa + \rho + \kappa)\|\nabla f(\tilde{w}_{t})\|^{2} \\
+ (1-\delta)(\kappa+1)\eta^{2}\sigma^{2} \\
= (1-\delta)(1+b)\|e_{t}\|^{2} + (1-\delta)\eta^{2}[1+b^{-1} + (\rho\kappa + \rho + \kappa)]\|\nabla f(\tilde{w}_{t})\|^{2} + (1-\delta)(\kappa+1)\eta^{2}\sigma^{2}.$$

Then, by taking b = δ 2(1−δ) , we have (1 − δ)(1 + b) = 1 − δ 2 , 1 + b <sup>−</sup><sup>1</sup> = 2−δ <sup>δ</sup> ≤ 2 δ , and

$$\mathbb{E}_{\zeta}[\mathbb{E}_{gc}[\mathbb{E}_{wc}[\|e_{t+1}\|^{2}]]]$$

$$\leq (1 - \frac{\delta}{2})\|e_{t}\|^{2} + (1 - \delta)\eta^{2}\left(\frac{2}{\delta} + \rho\kappa + \rho + \kappa\right)\|\nabla f(\tilde{w}_{t})\|^{2} + (1 - \delta)(\kappa + 1)\eta^{2}\sigma^{2}.$$

We simplify the notation by denoting E[∥et+1∥ 2 ] = E<sup>ζ</sup> [Egc[Ewc[∥et+1∥ 2 ]]], and then unroll the sequence of e<sup>t</sup> back to t = 0.

$$\mathbb{E}[\|e_{t+1}\|^{2}]$$

$$\leq \sum_{\tau=0}^{t} (1 - \frac{\delta}{2})^{t-\tau} \left[ (1 - \delta)\eta^{2} \left( \frac{2}{\delta} + \rho\kappa + \rho + \kappa \right) \|\nabla f(\tilde{w}_{\tau})\|^{2} + (1 - \delta)(\kappa + 1)\eta^{2}\sigma^{2} \right]$$

$$\leq (1 - \delta)\eta^{2} \left( \frac{2}{\delta} + \rho\kappa + \rho + \kappa \right) \sum_{\tau=0}^{t} (1 - \frac{\delta}{2})^{t-\tau} \|\nabla f(\tilde{w}_{\tau})\|^{2} + (1 - \delta)(\kappa + 1)\eta^{2}\sigma^{2} \sum_{\tau=0}^{t} (1 - \frac{\delta}{2})^{t-\tau}$$

$$\leq (1 - \delta)\eta^{2} \left( \frac{2}{\delta} + \rho\kappa + \rho + \kappa \right) \sum_{\tau=0}^{t} (1 - \frac{\delta}{2})^{t-\tau} \|\nabla f(\tilde{w}_{\tau})\|^{2} + \frac{2(1 - \delta)(\kappa + 1)\eta^{2}\sigma^{2}}{\delta}.$$

$$\geq \sum_{\tau=0}^{t} (1 - \frac{\delta}{2})^{t-\tau} \leq \frac{1}{1 - (1 - \frac{\delta}{2})}$$

$$\begin{split} & \text{Taking } \eta \leq \frac{1}{10L\left(\frac{2}{\delta} + \rho \kappa + \rho + \kappa\right)}, \text{ we have} \\ & \mathbb{E}[\|e_{t+1}\|^2] \\ & \leq \frac{1-\delta}{100L^2\left(\frac{2}{\delta} + \rho \kappa + \rho + \kappa\right)} \sum_{\tau=0}^t (1 - \frac{\delta}{2})^{t-\tau} \|\nabla f(\tilde{w}_\tau)\|^2 + \frac{2(1-\delta)(\kappa+1)\eta\sigma^2}{\delta 10L\left(\frac{2}{\delta} + \rho \kappa + \rho + \kappa\right)} \\ & \leq \frac{1-\delta}{100L^2\frac{2}{\delta}} \sum_{\tau=0}^t (1 - \frac{\delta}{2})^{t-\tau} \|\nabla f(\tilde{w}_\tau)\|^2 + \frac{2(1-\delta)(\kappa+1)\eta\sigma^2}{\delta 10L\frac{2}{\delta}} \\ & \leq \frac{(1-\delta)\delta}{200L^2} \sum_{\tau=0}^t (1 - \frac{\delta}{2})^{t-\tau} \|\nabla f(\tilde{w}_\tau)\|^2 + \frac{(1-\delta)(\kappa+1)\eta\sigma^2}{10L}. \end{split}$$

Then, stacking  $\mathbb{E}[\|e_t\|^2]$  and taking total expectation, we have

$$\sum_{t=0}^{T} \mathbb{E}[\|e_{t+1}\|^{2}]$$

$$\leq \frac{(1-\delta)\delta}{200L^{2}} \sum_{t=0}^{T} \sum_{\tau=0}^{t} (1-\frac{\delta}{2})^{t-\tau} \|\nabla f(\tilde{w}_{\tau})\|^{2} + \frac{(T+1)(1-\delta)(\kappa+1)\eta\sigma^{2}}{10L}$$

$$\leq \frac{(1-\delta)\delta}{200L^{2}} \sum_{t=0}^{T} \left[ \sum_{\tau=0}^{+\infty} (1-\frac{\delta}{2})^{\tau} \right] \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{(T+1)(1-\delta)(\kappa+1)\eta\sigma^{2}}{10L}$$

$$\leq \frac{1-\delta}{100L^{2}} \sum_{t=0}^{T} \|\nabla f(\tilde{w}_{t})\|^{2} + \frac{(T+1)(1-\delta)(\kappa+1)\eta\sigma^{2}}{10L}.$$

Putting all the ingredients together and taking total expectation, we have

$$\begin{split} \sum_{t=0}^{T} \mathbb{E}[f(w_{t+1})] \\ &\leq \sum_{t=0}^{T} \mathbb{E}[f(w_t)] - \frac{\eta}{4} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2] + \frac{\eta L^2}{2} \sum_{t=0}^{T} \mathbb{E}[\|e_t\|^2] + \frac{(T+1)\eta^2 L(\kappa+1)\sigma^2}{2} \\ \Rightarrow \quad \mathbb{E}[f(w_{T+1})] \\ &\leq \mathbb{E}[f(w_0)] - \frac{\eta}{4} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2] + \frac{\eta L^2}{2} \sum_{t=0}^{T} \mathbb{E}[\|e_t\|^2] + \frac{(T+1)\eta^2 L(\kappa+1)\sigma^2}{2} \\ \Rightarrow \quad \mathbb{E}[f(w_{T+1})] \\ &\leq \mathbb{E}[f(w_0)] - \frac{\eta}{4} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2] + \frac{\eta L^2}{2} \sum_{t=0}^{T} \mathbb{E}[\|e_t\|^2] + \frac{(T+1)\eta^2 L(\kappa+1)\sigma^2}{2} \\ \Rightarrow \quad \mathbb{E}[f(w_{T+1})] \\ &\leq \mathbb{E}[f(w_0)] - \frac{\eta}{4} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2] + \frac{(1-\delta)\eta}{200} \sum_{t=0}^{T} \|\nabla f(\tilde{w}_t)\|^2 \\ &+ \frac{(T+1)(1-\delta)(\kappa+1)L\eta^2\sigma^2}{20} + \frac{(T+1)\eta^2 L(\kappa+1)\sigma^2}{2} \\ \Rightarrow \quad \mathbb{E}[f(w_{T+1})] \leq \mathbb{E}[f(w_0)] - \frac{\eta}{8} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2] + \frac{(T+1)(11-\delta)(\kappa+1)L\eta^2\sigma^2}{20} \\ \Rightarrow \quad \frac{1}{8(T+1)} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_t)\|^2] \leq \frac{1}{T+1} \frac{\mathbb{E}[f(w_0)] - \mathbb{E}[f(w_{T+1})]}{\eta} + \frac{(11-\delta)(\kappa+1)L\eta\sigma^2}{20} \end{split}$$

Finally, using Lemma A.1, we have

$$\begin{split} &\frac{1}{T+1} \sum_{t=0}^{T} \mathbb{E}[\|\nabla f(\tilde{w}_{t})\|^{2}] \\ &\leq \frac{8}{T+1} \frac{\mathbb{E}[f(w_{0})] - f^{*} + f^{*} - \mathbb{E}[f(w_{T+1})]}{\eta} + \frac{8(11-\delta)(\kappa+1)L\eta\sigma^{2}}{20} \\ &\leq \frac{80L\left(\frac{2}{\delta} + \rho\kappa + \rho + \kappa\right)(f(w_{0}) - f^{*})}{T+1} + 4\sigma\sqrt{\frac{(11-\delta)(\kappa+1)L(f(w_{0}) - f^{*})}{T+1}}. \end{split}$$

## **B** Other Evaluation Results

To further demonstrate the effectiveness of SDP4Bit in enhancing training efficiency, we present the relationship between wall clock time and training loss in Figure 9.

<span id="page-17-1"></span>![](_page_17_Figure_4.jpeg)

Figure 9: Comparison of validation loss versus wall-clock time for Baseline, ZeRO++ and SDP4Bit on the GPT-6.7B model.

<span id="page-17-0"></span>To further illustrate the impact of the Hadamard transformation on (de)quantization performance, we provide (de)quantization throughput experiment in Table 5, which is tested on an A100 GPU.

| Input/Output Size |                 | ization          | Dequantization  |                 |  |
|-------------------|-----------------|------------------|-----------------|-----------------|--|
|                   | w/o Had.        | w/ Had.          | w/o Had.        | w/ Had.         |  |
| 8 MB              | 305.6±10.9      | $301.8 \pm 10.6$ | 367.7±10.6      | $359.6 \pm 9.6$ |  |
| 16 MB             | 389.0±12.8      | $387.1 \pm 8.2$  | 428.0±10.6      | $428.6 \pm 7.6$ |  |
| 64 MB             | 494.8± 3.7      | $493.7 \pm 2.6$  | $505.3 \pm 2.1$ | $505.6 \pm 2.2$ |  |
| 512 MB            | $682.1 \pm 0.8$ | $681.6 \pm 1.2$  | $685.1 \pm 0.8$ | $685.2 \pm 0.6$ |  |
| 1024 MB           | $686.5 \pm 1.2$ | $686.3 \pm 0.4$  | $688.0 \pm 0.3$ | $688.0 \pm 0.3$ |  |
| 2048 MB           | $688.6 \pm 0.2$ | $688.6 \pm 0.2$  | $689.5 \pm 0.2$ | $689.4 \pm 0.2$ |  |

Table 5: (De)quantization Throughput with/without Hadamard, including std. dev.

## **C** Notations in Training

| qWD weight difference int4 quantization  ULq original ZeRO++ uniform-level Int4-Int4 all-to-all gradient quantization  TLq two-level Int8-Int4 all-to-all gradient quantization  TI a-HS two-level Int8-Int4 all-to-all gradient quantization with Hadamard Smoother | qW     | original ZeRO++ int4 weight quantization                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|-----------------------------------------------------------------------------|
| TLq two-level Int8-Int4 all-to-all gradient quantization                                                                                                                                                                                                             | qWD    | weight difference int4 quantization                                         |
| <u> </u>                                                                                                                                                                                                                                                             | ULq    | original ZeRO++ uniform-level Int4-Int4 all-to-all gradient quantization    |
| TI a-HS two-level Int8-Int4 all-to-all gradient quantization with Hadamard Smoother                                                                                                                                                                                  | TLq    | two-level Int8-Int4 all-to-all gradient quantization                        |
| 12d 115 two level into int i air to air graaicht quantization with Hadamard Shioother                                                                                                                                                                                | TLq-HS | two-level Int8-Int4 all-to-all gradient quantization with Hadamard Smoother |

Table 6: Notations in experiments.

## <span id="page-18-0"></span>**D** Detailed Training Settings

In the experimental section, we utilize a total of six different sizes of GPT models. Their model configurations are detailed in Table 7.

For the accuracy experiments, we standardize the batch size to 256, and set sequence length to 2048. We use AdamW [17] optimizer in all the experiments. The detailed training parameters are listed in Table 9.

In the throughput experiments, to more clearly study the communication bottleneck and ensure consistency across different GPU counts, we set the accumulation step to 1. The batch size is adjusted according to the number of GPUs, and the sequence length (micro batch) is uniformly set to 2048. Due to the different number of GPUs per node in the two architectures, we adjusted the tensor parallel size (TP) and pipeline parallel size (PP) accordingly, referencing [26], to achieve the highest throughput. Specifically, the maximum tensor parallel size is 4 for the 4xA100 environment and 8 for the 8xH800 environment. See detailed parameters in Table 8.

Table 7: Model Size Parameters

<span id="page-18-1"></span>

| Model Size | Sequence Length | Hidden Size | Layers |
|------------|-----------------|-------------|--------|
| 125M       | 2048            | 768         | 12     |
| 350M       | 2048            | 1024        | 24     |
| 1.3B       | 2048            | 2048        | 24     |
| 6.7B       | 2048            | 4096        | 32     |
| 13B        | 2048            | 5120        | 40     |
| 18B        | 2048            | 6144        | 40     |

Table 8: Parallel Configuration for Throughput Test

| Model Size | TP  | PP  | Accumulation<br>Step |
|------------|-----|-----|----------------------|
| 1.3B       | 1   | 1   | 1                    |
| 2.7B       | 1   | 1   | 1                    |
| 6.7B       | 4   | 1   | 1                    |
| 13B        | 4/8 | 2/1 | 1                    |
| 18B        | 4/8 | 2/1 | 1                    |

Table 9: E2E Convergence Training Parameters

<span id="page-18-2"></span>

| Model Size | Learning Rate | Betas     | Epsilon | Weight Decay | Batch Size |
|------------|---------------|-----------|---------|--------------|------------|
| 125M       | 6e-4          | 0.9, 0.95 | 1e-8    | 0.1          | 256        |
| 350M       | 3e-4          | 0.9, 0.95 | 1e-8    | 0.1          | 256        |
| 1.3B       | 2e-4          | 0.9, 0.95 | 1e-8    | 0.1          | 256        |
| 6.7B       | 12e-5         | 0.9, 0.95 | 1e-8    | 0.1          | 256        |