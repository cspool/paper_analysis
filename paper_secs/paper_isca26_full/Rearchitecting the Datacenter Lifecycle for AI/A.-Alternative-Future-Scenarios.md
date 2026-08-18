# *A. Alternative Future Scenarios*

To avoid conclusions that are overly driven by current trends, we evaluate non-trend scenarios. We introduce structural breaks in demand, model scaling, hardware capability, or pricing, and re-run the full lifecycle optimization pipeline.

Scenario 1: Demand Shock and Plateau. To stress-test procurement under a breakthrough-driven demand surge, we model a regime where aggregate AI demand jumps by a factor α > 1 at year t<sup>s</sup> and then plateaus. In this setting, trendextrapolating policies systematically overestimate demand and persistently overprovision. Formally, we model demand with a zero post-shock growth as:

$$D(t) = \begin{cases} D_{\text{trend}}(t), & t < t_s \\ \alpha \cdot D_{\text{trend}}(t_s), & t \ge t_s \end{cases}$$

*Results.* Once our framework detects a demand jump, it rapidly scales capacity to absorb the shock. As growth approaches zero, it curtails further expansion and extends hardware refresh lifecycle to reflect market saturation. Relative to the baseline, our policy avoids a second wave of unnecessary expansion and reduces stranded capital. In a representative scenario (α = 3, t<sup>s</sup> = 5 years), it reduces TCO by *31%*.

Scenario 2: Model Size Contraction. While recent years have been dominated by model scaling, efficiency gains could reverse this trend. To capture this, we decouple demand growth from per-task compute intensity and introduce a breakpoint at year tr, after which efficiency improves (β < 1):

$$C_{\text{per-task}}(t) = \begin{cases} C_{\text{trend}}(t), & t < t_r \\ \beta \cdot C_{\text{trend}}(t_r), & t \ge t_r \end{cases}$$

*Results.* Our framework shifts from aggressive scale-out to selective refresh, prioritizing energy-efficient nodes while deferring peak-performance upgrades. As per-task compute intensity declines, accelerator demand decreases even as request rates remain stable or continue to grow. In a representative scenario (β = 0.8, t<sup>r</sup> = 5 years), this policy reduces TCO by *43%* relative to static amortization-based replacement.

Scenario 3: Hardware Capability Shock. To model regime shifts (*e.g.*, non-linear scaling gains, alternative interconnect stacks), we model a discontinuous jump in hardware capabilities, followed by slower improvements, where γ ≫ 1 represents a generational breakthrough (*e.g.*, architectural or interconnect redesign):

$$P(t) = \begin{cases} P_{\text{trend}}(t), & t < t_h \\ \gamma \cdot P_{\text{trend}}(t_h), & t = t_h \\ \gamma \cdot P_{\text{trend}}(t_h) \cdot (1 + \epsilon(t - t_h)), & t > t_h \end{cases}$$

*Results.* Delaying refresh past t<sup>h</sup> becomes strictly suboptimal, even for relatively new clusters. Our framework automatically accelerates refresh cycles immediately after the capability jump (reducing the lifecycle), accepting temporary write-down of partially depreciated hardware when the performance-perdollar delta exceeds the residual value penalty.

These discontinuities amplify the benefit of lifecycle optimization. Fixed refresh intervals fail catastrophically under capability shocks; our framework adapts refresh with marginal TCO improvement. Under a representative scenario (γ = 3, t<sup>h</sup> = 5 years, ϵ = 0.05), our framework reduces TCO by *38%*. Scenario 4: Hardware Price Shock. To capture supply-chain corrections, competitive shifts (*e.g.*, non-dominant vendor), or architectural commoditization, we model a sudden hardware price drop (δ < 1) followed by gradual recovery:

$$\text{Price}(t) = \begin{cases} \text{Price}_{\text{trend}}(t), & t < t_p \\ \delta \cdot \text{Price}_{\text{trend}}(t_p), & t = t_p \\ \delta \cdot \text{Price}_{\text{trend}}(t_p) \cdot (1 + \eta(t - t_p)), & t > t_p \end{cases}$$

*Result.* The framework advances procurement to exploit temporary price advantages, increasing short-term capex but reducing long-term TCO. The refresh policy becomes pricesensitive rather than purely capability-sensitive, showing the importance of jointly modeling performance and cost trajectories. Under a representative scenario (δ = 0.6, t<sup>p</sup> = 5 years, η = 0.1), our framework reduces TCO by *36%*.

Summary. Across scenarios, the numeric optima vary, but the conclusions are consistent. First, optimal datacenter lifecycle policies are sensitive to structural breaks, making fixed refresh heuristics suboptimal. Second, avoiding stranded capital and energy waste requires jointly modeling demand, hardware capability, and pricing trajectories. Third, hardware–software co-design effects strengthen under regime shifts.

Overall, our framework does not assume smooth trends. It provides a decision methodology that remains robust to nonlinear, non-monotonic, and regime-shift futures, improving the credibility of our conclusions beyond baseline projections.

