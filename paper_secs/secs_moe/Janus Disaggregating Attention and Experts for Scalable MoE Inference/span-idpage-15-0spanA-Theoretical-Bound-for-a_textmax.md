# <span id="page-15-0"></span>**A** Theoretical Bound for $a_{\text{max}}$

This appendix derives the closed-form upper bound on  $a_{\text{max}}(n_e, B)$  used in §3.5. We model expert activation as a balls-into-bins process and take an adversarial view of AEBS: every activation of a replicated expert is assumed to land on the instance being analyzed, so the bound is independent of the scheduler's routing decisions.

Consider a batch of *B* tokens, each selecting *K* experts with activation probabilities  $\sum_e p_e = K$ , and let  $X_e \in \{0,1\}$  denote the event that expert *e* is hit by at least one token in the batch; then  $\Pr(X_e = 1) = 1 - (1 - p_e)^B$ . Under the adversarial view, the load on instance *g* satisfies  $a_g \leq \sum_{e \in P(g)} X_e$ , and expectation gives:

$$\mathbb{E}[a_g] \le \sum_{e \in P(g)} [1 - (1 - p_e)^B]. \tag{4}$$

Under uniform activation ( $p_e = K/E$ ), this simplifies to  $C \cdot [1 - (1 - K/E)^B]$ , which grows with B and saturates at C as every hosted expert becomes almost surely activated. The bottleneck instance is the one that attains  $\bar{a}_{max} := \max_g \mathbb{E}[a_g]$ .

Although top-K gating couples the indicators  $\{X_e\}$ , the resulting coupling is *negatively* associated, which preserves  $Var(a_g) \leq \mathbb{E}[a_g]$ ; applying a Bernstein-type tail bound on each instance and a union bound over  $n_e$  instances yields:

<span id="page-15-1"></span>
$$a_{\max}(n_e, B) \le \left[\min\left(C, \ \bar{a}_{\max} + \sqrt{2\bar{a}_{\max}\ln n_e}\right) + 1\right].$$
 (5)

 $a_{\rm max}$  counts distinct experts and is integer-valued; the +1 slack absorbs the replication-induced overflow that lets the bottleneck instance occasionally host  $\lceil E/n_e \rceil + 1$  distinct experts.

Two regimes follow. When B is small,  $\bar{a}_{\max} \ll C$  and  $a_{\max}$  grows with B, driving  $T_{\text{moe}}^{(\ell)}$  upward. When B is large,  $\bar{a}_{\max} \to C$  and  $a_{\max}$  plateaus, so  $T_{\text{moe}}^{(\ell)}$  is effectively capped while  $T_{\text{attn}}^{(\ell)}$  continues to grow with  $B/n_a$ ; this explains the diminishing returns of batch size on end-to-end throughput. Eq. (5) is conservative because it treats activations adversarially and ignores the variance-flattening effect of replication and placement (§3.5) as well as the peak-reduction effect of AEBS (§3.4); the Monte Carlo estimator  $\widehat{a}_{\max}$  used by the scaling solver absorbs these effects.

Empirical validation and high-leverage regime. Figure 17 overlays the analytical bound against the layer-averaged Monte Carlo estimate  $\widehat{a}_{max}$  on ShareGPT across  $n_e \in \{6, 8, 12, 16\}$ , with three batch-size regimes shaded. The bound holds on all cells: in the saturation regime ( $B \ge 64$ ) the gap is within one or two experts, and even at small B it stays below  $\sim 2\times$ . Two observations justify treating this as an acceptable, usefully conservative bound rather than a loose one. First, the gap is one-sided—the bound never under-predicts, so using it in the scaling solver can only err on the side of over-provisioning, which is the safe direction under SLO constraints. Second, the predicted values remain within the range

<span id="page-15-3"></span>![](_page_15_Figure_10.jpeg)

Figure 17: Analytical bound (dashed) vs. Monte Carlo estimate (solid) on ShareGPT across  $n_e \in \{6, 8, 12, 16\}$ , with three batch-size regimes shaded. The high-leverage window  $B \in [10, 100]$  is where  $a_{\text{max}}$  is simultaneously most sensitive to placement (steepest slope) and already at 30-60% of C, and it coincides with the per-instance batch sizes reported in online decode traces.

of activated-expert counts actually observed in practice: the MoE latency measurements in Fig. 3 span roughly 8–22 activated experts, and all bound values in Fig. 17 fall inside or below this envelope, so the bound does not drive the solver into regions that contradict measured behavior. The residual looseness reflects the adversarial assumption that ignores the variance-flattening effect of replication and placement and the peak-reduction effect of AEBS;  $\hat{a}_{max}$  closes this gap at decision time by incorporating both effects.

Beyond validating the bound, Fig. 17 reveals three regimes with sharply different scheduling leverage. (i) Sparse ( $B \lesssim$ 10, gray band):  $\hat{a}_{max}$  is small ( $\leq 4$  across all  $n_e$ ) and largely insensitive to placement—there are too few tokens for policy to matter. (ii) Saturation ( $B \ge 100$ , blue band):  $\widehat{a}_{max}$  plateaus near min $(C, E/n_e)$  (e.g.,  $n_e$ =6: 19.98  $\rightarrow$  20.47 from B=256 to B=512;  $n_e=16$ :  $10.78 \rightarrow 10.95$ ); the ceiling is structural and no scheduling policy can push  $\hat{a}_{max}$  below it. (iii) Highleverage ( $B \in [10, 100]$ , orange band): the curve exhibits its steepest slope—each  $4\times$  increase in B raises  $\hat{a}_{max}$  by 4-7experts (e.g.,  $n_e$ =6: B=16  $\rightarrow$  64 moves  $\hat{a}_{max}$  from 8.46 to 15.66;  $n_e$ =12: 5.84  $\rightarrow$  10.82), and absolute values already reach 30-60% of *C*=27. Because  $T_{\text{moe}}^{(\ell)} = \beta^{(\ell)} \cdot a_{\text{max}}^{(\ell)} + c_e^{(\ell)}$ dominates per-layer latency once  $a_{\text{max}}$  is in this range, a 2–3expert shift in placement suffices to move TPOT across the SLO. Per-instance batch sizes in online decode traces [22,29] sit inside exactly this window, which motivates concentrating replication, placement, and AEBS on  $B \in [10, 100]$  rather than on sparse or saturated regimes.

