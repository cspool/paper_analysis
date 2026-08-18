# *A. Total Cost of Ownership (TCO) Model*

Existing TCO Models. TCO modeling is a long-standing tool in datacenter planning. Classical frameworks [\[49\]](#page-14-19), [\[58\]](#page-14-20) decompose facility costs into capital and operational components with straight-line amortization. Recent work extends TCO to sustainability-driven hardware choices [\[41\]](#page-13-5), [\[111\]](#page-15-22), carbon-aware AI inference [\[64\]](#page-14-21), and platform-aware [\[78\]](#page-14-22) and performance-cost modeling for LLM systems [\[40\]](#page-13-26). Finally, industry analyses emphasize that effective TCO depends on platforms, reliability, and utilization dynamics [\[78\]](#page-14-22). Our framework builds on this foundation and extends it in two ways: (1) it specializes TCO for AI workloads by integrating roofline-based performance modeling, acceleratorroadmap projections, and LLM workload evolution; and (2) it enables *cross-stage* lifecycle analysis, capturing how buildtime, provisioning, and operational decisions interact over a 15-year horizon. The contribution is a systematic use of TCO as a unifying lens for evaluating and optimizing architectural choices across the full AI datacenter lifecycle.

Our TCO Model. [Table III](#page-4-0) summarizes the components of our TCO model, breaking them down into *CapEx* and *OpEx*:

$$CapEx_F + CapEx_{Pow.} + CapEx_{Cool} + CapEx_{Net} + CapEx_{IT}$$

$$OpEx_{energy} + OpEx_{M\&R} + OpEx_{network} + OpEx_{other}$$

For the annualized TCO, CapEx amortizes long-term infrastructure and IT over their useful lives, OpEx captures variable and recurring operational costs over a full year.

<span id="page-3-0"></span>![](_page_3_Figure_12.jpeg)

Fig. 3: TCO breakdown for a 10MW AI datacenter.

[Figure 3](#page-3-0) shows the annual datacenter TCO breakdown for a representative user demand and model size projected for 2025. At 75% average utilization, this 10MW datacenter [\[120\]](#page-15-10) hosts roughly 500 H100 servers, consuming 70 GWh of energy per year. GPU servers drive IT CapEx and dominate costs, followed by energy-related OpEx. Building construction and maintenance contribute the least.

Capital Expenses (CapEx). The upfront costs for acquiring, building, or upgrading long-term datacenter assets, including facility, IT equipment (such as servers and racks), and networking infrastructure. Facility, network, and IT assets are amortized over 15–30, 7–10, and 3–5 years respectively, using straight-line or declining-balance depreciation [\[111\]](#page-15-22). We normalize CapEx per delivered kW to enable design comparisons. *Facility.* It includes the physical infrastructure to support a datacenter, providing the foundational environment needed to house equipment safely and efficiently.

*Power Infrastructure.* Electrical systems deliver reliable power to racks and IT equipment, including all power devices and connections to ensure stable and redundant power distribution. *Cooling Infrastructure.* Mechanical systems cool datacenters to maintain safe operating temperatures, with core infrastructure such as chillers and pumps. Together, these systems prevent overheating and enable high-performance operation. *Networking Infrastructure.* It connects servers, storage, and other datacenter resources. Fabric switches route rack-torack traffic, while optical transceivers and structured cabling provide high-bandwidth, low-latency links across the facility. *IT Infrastructure.* Compute servers equipped with CPUs, memory, and accelerators (*e.g.*, GPUs, TPUs, or NPUs) along with racks and storage devices like NVMe.

Operational Expenses (OpEx). Ongoing costs of running and maintaining a datacenter. We model utilization-sensitive costs (*e.g.*, energy) based on workload mix and scheduling policies, since they scale with activity. Utilization-insensitive costs (*e.g.*, maintenance, software contracts, leases) are treated as per-rack/site constants regardless of workload intensity.

*Energy.* Electricity costs include power for IT equipment and supporting infrastructure (*e.g.*, cooling, power distribution). Billed monthly, these costs reflect IT utilization and *power usage effectiveness* (PUE), the ratio of total facility energy to IT energy. A higher PUE indicates more energy spent on overheads like cooling and power conversion.

*Maintenance & Repairs.* Preventive and corrective maintenance of mechanical, electrical, and IT systems. Costs are driven by component failure rates (cooling, power, servers,

<span id="page-4-0"></span>

| Category | Component                                       | Description                                                                                                                                                                                                                                                                                             | Example Cost (\$)                                                                                                           |
|----------|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| CapEx    | IT Networking Building Power Cooling            | Servers, racks, accelerators, storage Fabric switches, optics, structured cabling Site preparation, building shell, land, electrical and mechanical base infrastructure Switchgear, transformers, UPS, PDUs, busbars, rack distribution Chillers, CRAH/CRAC units, pumps, piping, liquid loops, airflow | \$375k/server [18]<br>\$2000/server [61], [90]<br>\$0.5/ft <sup>2</sup> [29]<br>\$7.0/W [26], [121]<br>\$2.5/W [108], [121] |
| OpEx     | Networking<br>Energy<br>Maintenance<br>Software | Port licenses, optics replacement, networking component power IT load scaled by PUE, utility tariffs, demand charges Spares, repairs, monitoring, water/treatment, field-replaceable units, failure-rate Licenses, support contracts                                                                    | \$600/server [61], [87], [89]<br>\$20–40/MWh [35]<br>\$5000/server [107]<br>\$200/server                                    |

TABLE III: TCO components for an example datacenter with DGX H100 [18] servers.

storage, networking) and include all activities required to keep the datacenter operational and reliable.

Network Operations. Ongoing costs of maintaining the datacenter network, including licensing fees for switch ports and the replacement of failed components. Costs are influenced by network size, redundancy, and utilization patterns, as higher traffic and denser topologies can increase wear and require more frequent upgrades or maintenance.

*Other.* Recurring expenses such as software licenses, support contracts, and land or lease payments. These costs are largely fixed and do not vary with workload or utilization.

## B. Modeling Assumptions

**Timeline.** We model a 15-year lifecycle starting from 2015 to 2030, fitting outputs to current trends (2025) and forecasting costs and fleet composition for the next 5 years. Our methodology generalizes to longer horizons, but uncertainty grows.

Workload. We focus on LLM inference as the dominant AI datacenter workload [11]. Training workloads follow a similar methodology but differ in modeling, as they are heavier in both computation and communication. We use input traces from DynamoLLM [105], which exhibit diurnal patterns, and assume a baseline of 100K requests per second (RPS). Following Section II-A, we apply a 15% annual growth rate [57], [96], [113], which implies over 200K RPS after five years.

**AI Models.** Based on 2015–2025 parameter-scaling trends (Section II-A), we assume linear growth in model size through 2030, with alternative scenarios for accelerated (exponential) or slowed (sub-linear) scaling. Providers adopt new models via *smooth migration*, gradually transitioning workloads [45], [85]. We assume future LLMs follow the LLaMA design [70] (*i.e.*, decoder-only transformers with consistent layer organization, attention mechanisms, MoE, and parameterization).

Hardware. Hardware projections include FLOPS, memory bandwidth, TDP, and cost, with linear growth trends [44]. We also model delays between the announcement of a new GPU [83] and its actual mass availability in cloud providers [6], [15], [99] (e.g., B200 had a delay 6–12 months). Performance. We develop a roofline model for LLM inference across diverse hardware. Our validation against known model/hardware pairings and profiling results confirms alignment with prior work [40], [119]. The model captures interactions between hardware (compute throughput, memory bandwidth) and workload (arithmetic intensity, memory footprint). Rather than relying on aggregate performance trends, the model

<span id="page-4-1"></span>![](_page_4_Figure_10.jpeg)

Fig. 4: Server count by GPU type over time in an AI fleet following the traditional baseline in Table II.

explicitly incorporates architectural parameters, such as peak FLOPS, memory bandwidth and capacity, interconnect bandwidth and latency, and power envelopes. By modeling resource ceilings and bottleneck transitions, technology advances translate directly into shifts of the roofline surface. This enables us to propagate microarchitectural changes into lifecycle TCO.

We analytically derive the arithmetic intensity and memory footprint from the LLM architecture and parameter count. Extending our models to new GPUs requires only peak FLOPs and memory bandwidth; for new LLMs, theoretical compute and memory requirements are recomputed from the architecture. The roofline model predicts time-to-first-token (TTFT) and time-between-tokens (TBT) latencies for a given hardware, model, and request load.

We then increase the load until requests exceed an SLO of 400 ms for TTFT and 100 ms for TBT [105]. The resulting *goodput* defines the maximum RPS sustainable without violating latency targets, identifying the utilization point where performance degrades. Using this SLO, we provision the minimal GPUs needed and compute corresponding utilization. **Cost.** We combine CapEx (IT hardware, networking, power, cooling) and OpEx (networking, energy, maintenance). CapEx is amortized over asset lifetimes, while OpEx captures recurring operational costs. This allows evaluating trade-offs such as upfront investment in cooling *vs.* deferred savings in refresh.

## <span id="page-4-2"></span>C. Lifecycle Evaluation

Baseline Timeline. Figure 4 shows the simulated deployment timeline of an AI fleet under the baseline traditional datacenter lifecycle approach (Table II). Model release cycles and hardware availability shape the fleet composition over time, with the release dates of notable large models marked for reference. The simulation starts in 2015 with 50 P100 servers supporting 100K RPS, at a total annual TCO of  $\approx $0.2M$ .

<span id="page-5-0"></span>

| Variable                   | Distribution | Parameters / Bounds                        | Notes / Correlations                                       |
|----------------------------|--------------|--------------------------------------------|------------------------------------------------------------|
| Workload growth factor     | Log-normal   | µ = log(1.05), σ = 0.05                    | Positive-only; correlated with model size growth (ρ = 0.4) |
| Model size annual growth   | Log-normal   | Fit to historical P50 trend; capped at ±2σ | Captures uncertainty in scaling-law extrapolation          |
| GPU Perf/W improvement     | Normal       | Mean from regression, σ = 0.1µ             | Correlated with GPU cost improvement (ρ = −0.5)            |
| GPU price per generation   | Triangular   | min = −15%, mode = 0%, max = +20%          | Reflects supply-chain variability                          |
| Release interval           | Discrete     | {1, 1.5, 2} years                          | Uniform sampling                                           |
| Electricity price (\$/kWh) | Log-normal   | Mean = regional avg., σ = 15%              | Independent across trials                                  |
| Cooling efficiency (PUE)   | Normal       | Mean = baseline, σ = 0.05                  | Affects total energy cost                                  |
| Server lifetime            | Discrete     | {4, 5, 6} years                            | Uniform sampling                                           |

TABLE IV: Stochastic variables used in our Monte Carlo simulations.

As user demand and model size grow (*i.e.*, 15% year-overyear), the fleet scales gradually. By 2024, traffic reaches 350K RPS, coinciding with DeepSeek V3 (671B-parameters) [\[21\]](#page-13-32), [\[67\]](#page-14-28), prompting a major H200 GPU refresh. Server count rises to 25K to meet performance targets, and annual TCO climbs to \$0.3B, reflecting additional hardware, expanded infrastructure, and higher OpEx. Deployment peaks align with major LLM releases, highlighting the strong link between AI model roadmaps and datacenter economics.

Monte Carlo Methodology. To account for TCO uncertainty, we run Monte Carlo simulations [\[73\]](#page-14-29) where we model inputs as random variables. Each trial represents a plausible future trajectory of workload growth, model scaling, hardware evolution, pricing, and energy costs. The simulator deterministically computes capacity planning, server acquisitions/decommissions, and annual CapEx/OpEx, producing a single TCO. Repeating this yields a distribution over lifecycle costs.

[Table IV](#page-5-0) summarizes the stochastic inputs, their distributions, parameterization, and rationale. We guide the distribution choices by historical AI model scaling trends, GPU release data, and publicly reported datacenter cost variability. In most experiments, we chose conservative bounds and separately study a few extreme cases.

We draw samples from a multivariate normal distribution with covariance matrix Σ, whose entries encode empirically derived pairwise correlations [\(Table IV\)](#page-5-0). We then map these samples to the desired marginals via inverse CDF transforms (*e.g.*, log-normal, triangular). All remaining variables are sampled independently. Unless otherwise stated, results are based on 10,000 independent trials. We verified that increasing to 20,000 trials changes the mean and 95% confidence interval of total TCO by less than 1%, indicating statistical stability.

We validate convergence using three checks: (1) running mean stabilization (change < 1% over final 2,000 samples), (2) stabilization of 5th/95th percentile estimates, and (3) bootstrap confidence intervals over batches of 1,000 samples. All reported figures use the full converged sample set.

For each policy (*e.g.*, aggressive vs. delayed refresh), our framework reports: (1) expected lifecycle TCO, (2) variance and 95% confidence intervals, (3) probability that one policy outperforms another, and (4) sensitivity (Sobol-style first-order effects via regression-based decomposition).

By modeling full distributions instead of point estimates, our approach provides distributional robustness and explicitly quantifies the option value associated with flexible hardware refresh timing under uncertainty.

