# A. The Classical 5-Minute Rule and Its Limitation

The rule makes a page-level decision: keep a page in DRAM when doing so is cheaper than fetching it from storage. This is expressed by the *break-even interval*: if the expected interreference time is below this interval, cache the page; otherwise, leave it on storage. Balancing the "rent" on DRAM against the cost of repeated fetches yields:

<span id="page-1-1"></span>
$$\tau_{\mathrm{break-even}} = \left(\frac{\text{\# Pages per MB}}{\text{Storage Drive IOPS}}\right) \times \left(\frac{\text{Storage Drive Cost}}{\text{Cost of 1MB DRAM}}\right)$$

The classical rule can be written in a simpler, unit-consistent form. Let  $C_{\mathrm{DRAM}}^{\mathrm{page}}$  denote the amortized capital cost of storing one page in DRAM, and  $C_{\mathrm{SSD}}^{\mathrm{IO}}$  denote the amortized capital cost per storage access (i.e., SSD cost divided by its peak IOPS). Break-even occurs when keeping a page in DRAM over a reuse interval T costs the same as repeatedly fetching it from storage:

$$T \cdot C_{\mathrm{DRAM}}^{\mathrm{page}} = C_{\mathrm{SSD}}^{\mathrm{IO}}.$$

Solving for T yields:

$$T_{\text{break-even}} = \frac{C_{\text{SSD}}^{\text{IO}}}{C_{\text{DRAM}}^{\text{page}}}.$$

When host and bandwidth costs are ignored and full peak SSD IOPS is assumed, the calibrated formulation that will be presented later in Section III reduces to this classical expression. This expression makes explicit that the break-even interval is simply the ratio between per-access storage cost and per-page DRAM capital cost. Under HDD-era parameters in [19], this ratio yielded a break-even point in minutes, providing the historical context for our seconds-scale findings. However, this *economics-only* view has the following key limits.

(A) Insufficient realism. The classical formulation treats host resources as free. In practice, issuing and completing I/O consumes CPU cycles, interrupts, and DRAM bandwidth, which are negligible for HDDs (100~200 IOPS) but significant for modern SSDs. Prior revisits [5], [17]–[19] also relied on vendor peak specs, overlooking architectural effects such as NAND physics, internal parallelism, and block-size scaling.

(B) Missing feasibility. Optimizing only device prices (e.g.,

\$/GB or \$/IOPS) cannot ensure deployability. Real feasibility depends on host-side constraints (e.g., submission/completion rate, latency and throughput targets, and DRAM bandwidth and capacity). Ignoring these factors can yield configurations unable to meet workload demands or service-level objectives. Summary. These gaps make the classical rule non-actionable for system design and provisioning. We therefore develop a feasibility-aware framework that models host resource usage and enforces practical system constraints, yielding accurate, actionable guidance. Unless stated otherwise, all cost terms

