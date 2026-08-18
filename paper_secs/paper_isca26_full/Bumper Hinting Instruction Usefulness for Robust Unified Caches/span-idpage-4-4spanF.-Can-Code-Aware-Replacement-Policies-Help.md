# <span id="page-4-4"></span>F. Can Code-Aware Replacement Policies Help?

Emissary [14] is a recent *code-aware* replacement policy targeting instruction cache misses that cause the pipeline to starve; once the corresponding lines are identified, Emissary pins them into the L2C, so as to retain them and ensure a cache hit for future accesses to the same line. In a nutshell, Emissary focuses on retaining code lines it identifies as high-priority, not on evicting *useless* lines faster; as such, Emissary does not discriminate between *useful* and *useless* code lines.

Figure 6a reports performance of the best Emissary configuration (max 25% of ways can be pinned, pinning probability 1/16) determined through extensive parameter exploration. We observe that Emissary harms the performance of modern mobile applications. The reason for Emissary's ineffectiveness lies in the behavior of mobile workloads: As Figure 5a reports,  $\sim 30\%$  of the L2C contains code lines and more than half of those code lines are *useless*. As Figure 6b shows, Emissary identifies as *high-priority*, and thus tries to pin, the majority of code lines in the L2C, because they correspond to instructions that cause pipeline starvation. However, it cannot discriminate between *useful* and *useless* code lines. Indeed, 56.7% of the

<span id="page-4-3"></span>![](_page_4_Figure_10.jpeg)

Fig. 7: Impact of inserting code lines as low priority (RRPV=3) in L2C on performance (top), fraction of L2C occupied by *useful* and *useless* code lines (bottom-left), and lifetime of *useful* and *useless* code lines in L2C (bottom-right).

lines identified as *high-priority* by Emissary are *useless* and pinning them increases L2C pressure, which explains why Emissary hurts performance on the studied mobile workloads.

#### IV. How to Deal with Useless Code Lines?

<span id="page-4-0"></span>Our analysis in Section III highlights that over 20% of the L2C is polluted by *useless* code lines that tend to reside there for a long time (see Figure 5). As noted in Section III-E, how to filter IFU requests that insert *useless* lines into the L2C is an open problem. While a predictor [29]–[35] could, potentially, be trained to identify *useful* code lines and serve as a filter, this approach would require a large storage budget for the code footprint of mobile workloads. It would also harm performance if *useful* lines were mispredicted and filtered out of the prefetch stream (Section VIII).

A more promising approach, from a storage overhead and complexity perspective, is to reduce the lifetime of *useless* code lines in the L2C by evicting them more quickly. A naive way of doing so is to simply insert all code lines into L2C at low priority (RRPV=3),<sup>3</sup> under the assumption that *useful* lines will experience hits, which will promote them in the cache and extend their lifetime, while *useless* lines will be evicted quickly in the absence of hits.

Figure 7 presents the evaluation of this approach. We observe that this naive policy severely harms performance and reduces the baseline IPC by 13.9% (Figure 7, top). The bottom half of Figure 7 explains the reason for this poor performance. While inserting code lines at low priority does indeed reduce the L2C occupancy and lifetime of *useless* code lines, it also adversely affects *useful* code lines, which are frequently evicted before they experience reuse. Because all code lines are aggressively evicted under this policy, we observe that other line types (data, MMU) enjoy longer lifetimes in L2C.

Why does this policy fail? In other words, why do *useful* code lines fail to get promoted quickly, which would prevent

<span id="page-4-2"></span><sup>&</sup>lt;sup>3</sup>The baseline L2C replacement policy is DRRIP [36] which relies on Re-Reference Prediction Values (RRPVs) to approximate the reuse of cache lines; RRPVs range between 0 (high chance of reuse) and 3 (low chance of reuse).

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Fig. 8: Upper bound potential of discriminating the *usefulness* of code lines on L2C lifetime (left) and performance (right).

them from being evicted? We find that the answer lies in the fact that *useful* lines tend to experience reuse in the L1I, which absorbs hits and prevents the L2C from observing that these lines are, in fact, *useful*. Thus, while an insertion policy can help reduce the lifetime of *useless* code lines, there needs to be a way to identify the *useful* code lines and keep them in the L2C. Next, we quantify this opportunity.

1) Opportunity Upper Bound: To evaluate the opportunity of a better-informed policy, we extend the approach in the previous study that inserts code lines into the L2C at low priority (RRPV=3) by adding a discriminator that identifies useful code lines and changes their insertion priority to high (RRPV=0). The discriminator is an oracle with perfect knowledge of whether an incoming code line will be useful or not.

Figure 8 quantifies the impact of this idealized policy lifetime of L2C lines and application performance. Figure 8 (left) reveals that, under this scenario, the lifetime of *useless* code lines is significantly reduced, as in the naive approach, but without harming the *useful* code lines. In fact, the lifetime of *useful* code lines in the L2C increases by ~50%, on average, compared to the baseline – a highly desirable outcome that provides more opportunity for reuse. Moreover, because *useless* code lines are quickly evicted, other line types (*e.g.*, data, MMU) also tend to stay longer in the cache and potentially experience more reuse. The increased lifetime of both *useful* code lines and other lines in L2C, stemming from quicker eviction of *useless* code lines, has a direct positive impact on application performance. As Figure 8 (right) shows, all considered workloads experience a performance improvement.

2) Seizing the Opportunity: Exploiting the opportunity shown in Figure 8 requires discriminating useful from useless code lines in the L2C. However, useful lines tend to see high reuse in L1I, which prevents the L2C from observing their usefulness. Intuitively, what is needed is a mechanism for informing the L2C about the usefulness of its code lines.

A naive version of such mechanism would send a message from the L1I cache to L2C on L1I cache hits, indicating that a specific line has been used. However, as discussed in Section III-B, a large fraction of code lines is accessed on a misspeculated execution path and are not actually useful per Definition 1. To quantify this behavior, we added instrumentation code to track the lifetime of all code lines in L2C. Then, for each L2C code line, we record (i) whether it experiences at least one hit in the L1I cache while it is resident in L2C

and (ii) whether any instruction belonging to that line commits before the line is evicted from L2C. With this instrumentation we compute, for each workload, the fraction of *useless* L2C code lines that experienced at least one L1I cache hit during their lifetime in the L2C. Averaged across all workloads, 47% of useless L2C code lines experience an L1I hit (min: 43%, max: 49%). For this reason, L1I hits are not a good proxy for *usefulness* of a code line.

Therefore, we need to more strictly follow Definition 1 and use commit information to drive the detection of useful code lines in the L2C. While simple in principle, this idea presents multiple challenges in a real-world design that must contend with wiring, bandwidth, and energy costs. Specific challenges include (1) minimizing signaling throughout the core pipeline (wiring, bandwidth, energy); (2) reducing MMU lookups for address translation and L1I cache accesses (bandwidth, energy); (3) minimizing signaling between the L1I and L2C (bandwidth, energy) since a line may see frequent reuse in the L1I cache, which could trigger many needless accesses to the L2C if every committed instruction triggered an RRPV update signal to the L2C. In short, an ideal scheme would minimize signaling throughout the pipeline and avoid redundant accesses to the MMU, L1I cache, and L2C. We next present a low-cost microarchitectural scheme that meets these constraints and makes commit-driven management of unified caches feasible in a real-world CPU design.

## V. Bumper Design

<span id="page-5-0"></span>Bumper is a microarchitectural scheme that capitalizes on the insights from our previous analysis. Bumper allows the L2C to distinguish between *useful* and *useless* code lines by carefully orchestrating the propagation of *usefulness* information across the CPU pipeline and cache hierarchy at a low implementation cost and with negligible signaling overhead.

At a high level, Bumper inserts all L2C code lines in the vulnerable position (RRPV=3) and subsequently promotes only the code lines identified as *useful*, thereby increasing their likelihood of remaining in the cache long enough to be reused. Bumper extends a typical microarchitecture with the ability to *track useful code lines* (Section V-A) and exploits this knowledge to *inform L2C management* (Section V-B). Figure 9 shows how Bumper extends a microarchitecture to correlate L2C lines with committed instructions; this design minimizes the impact on the microarchitecture in terms of both added storage and signaling. The rest of this section presents how Bumper operates, referring to the circled markers in Figure 9.

# <span id="page-4-4"></span>F. Can Code-Aware Replacement Policies Help?

Emissary [14] is a recent *code-aware* replacement policy targeting instruction cache misses that cause the pipeline to starve; once the corresponding lines are identified, Emissary pins them into the L2C, so as to retain them and ensure a cache hit for future accesses to the same line. In a nutshell, Emissary focuses on retaining code lines it identifies as high-priority, not on evicting *useless* lines faster; as such, Emissary does not discriminate between *useful* and *useless* code lines.

Figure 6a reports performance of the best Emissary configuration (max 25% of ways can be pinned, pinning probability 1/16) determined through extensive parameter exploration. We observe that Emissary harms the performance of modern mobile applications. The reason for Emissary's ineffectiveness lies in the behavior of mobile workloads: As Figure 5a reports,  $\sim 30\%$  of the L2C contains code lines and more than half of those code lines are *useless*. As Figure 6b shows, Emissary identifies as *high-priority*, and thus tries to pin, the majority of code lines in the L2C, because they correspond to instructions that cause pipeline starvation. However, it cannot discriminate between *useful* and *useless* code lines. Indeed, 56.7% of the

<span id="page-4-3"></span>![](_page_4_Figure_10.jpeg)

Fig. 7: Impact of inserting code lines as low priority (RRPV=3) in L2C on performance (top), fraction of L2C occupied by *useful* and *useless* code lines (bottom-left), and lifetime of *useful* and *useless* code lines in L2C (bottom-right).

lines identified as *high-priority* by Emissary are *useless* and pinning them increases L2C pressure, which explains why Emissary hurts performance on the studied mobile workloads.

#### IV. How to Deal with Useless Code Lines?

<span id="page-4-0"></span>Our analysis in Section III highlights that over 20% of the L2C is polluted by *useless* code lines that tend to reside there for a long time (see Figure 5). As noted in Section III-E, how to filter IFU requests that insert *useless* lines into the L2C is an open problem. While a predictor [29]–[35] could, potentially, be trained to identify *useful* code lines and serve as a filter, this approach would require a large storage budget for the code footprint of mobile workloads. It would also harm performance if *useful* lines were mispredicted and filtered out of the prefetch stream (Section VIII).

A more promising approach, from a storage overhead and complexity perspective, is to reduce the lifetime of *useless* code lines in the L2C by evicting them more quickly. A naive way of doing so is to simply insert all code lines into L2C at low priority (RRPV=3),<sup>3</sup> under the assumption that *useful* lines will experience hits, which will promote them in the cache and extend their lifetime, while *useless* lines will be evicted quickly in the absence of hits.

Figure 7 presents the evaluation of this approach. We observe that this naive policy severely harms performance and reduces the baseline IPC by 13.9% (Figure 7, top). The bottom half of Figure 7 explains the reason for this poor performance. While inserting code lines at low priority does indeed reduce the L2C occupancy and lifetime of *useless* code lines, it also adversely affects *useful* code lines, which are frequently evicted before they experience reuse. Because all code lines are aggressively evicted under this policy, we observe that other line types (data, MMU) enjoy longer lifetimes in L2C.

Why does this policy fail? In other words, why do *useful* code lines fail to get promoted quickly, which would prevent

<span id="page-4-2"></span><sup>&</sup>lt;sup>3</sup>The baseline L2C replacement policy is DRRIP [36] which relies on Re-Reference Prediction Values (RRPVs) to approximate the reuse of cache lines; RRPVs range between 0 (high chance of reuse) and 3 (low chance of reuse).

<span id="page-5-1"></span>![](_page_5_Figure_0.jpeg)

Fig. 8: Upper bound potential of discriminating the *usefulness* of code lines on L2C lifetime (left) and performance (right).

them from being evicted? We find that the answer lies in the fact that *useful* lines tend to experience reuse in the L1I, which absorbs hits and prevents the L2C from observing that these lines are, in fact, *useful*. Thus, while an insertion policy can help reduce the lifetime of *useless* code lines, there needs to be a way to identify the *useful* code lines and keep them in the L2C. Next, we quantify this opportunity.

1) Opportunity Upper Bound: To evaluate the opportunity of a better-informed policy, we extend the approach in the previous study that inserts code lines into the L2C at low priority (RRPV=3) by adding a discriminator that identifies useful code lines and changes their insertion priority to high (RRPV=0). The discriminator is an oracle with perfect knowledge of whether an incoming code line will be useful or not.

Figure 8 quantifies the impact of this idealized policy lifetime of L2C lines and application performance. Figure 8 (left) reveals that, under this scenario, the lifetime of *useless* code lines is significantly reduced, as in the naive approach, but without harming the *useful* code lines. In fact, the lifetime of *useful* code lines in the L2C increases by ~50%, on average, compared to the baseline – a highly desirable outcome that provides more opportunity for reuse. Moreover, because *useless* code lines are quickly evicted, other line types (*e.g.*, data, MMU) also tend to stay longer in the cache and potentially experience more reuse. The increased lifetime of both *useful* code lines and other lines in L2C, stemming from quicker eviction of *useless* code lines, has a direct positive impact on application performance. As Figure 8 (right) shows, all considered workloads experience a performance improvement.

2) Seizing the Opportunity: Exploiting the opportunity shown in Figure 8 requires discriminating useful from useless code lines in the L2C. However, useful lines tend to see high reuse in L1I, which prevents the L2C from observing their usefulness. Intuitively, what is needed is a mechanism for informing the L2C about the usefulness of its code lines.

A naive version of such mechanism would send a message from the L1I cache to L2C on L1I cache hits, indicating that a specific line has been used. However, as discussed in Section III-B, a large fraction of code lines is accessed on a misspeculated execution path and are not actually useful per Definition 1. To quantify this behavior, we added instrumentation code to track the lifetime of all code lines in L2C. Then, for each L2C code line, we record (i) whether it experiences at least one hit in the L1I cache while it is resident in L2C

and (ii) whether any instruction belonging to that line commits before the line is evicted from L2C. With this instrumentation we compute, for each workload, the fraction of *useless* L2C code lines that experienced at least one L1I cache hit during their lifetime in the L2C. Averaged across all workloads, 47% of useless L2C code lines experience an L1I hit (min: 43%, max: 49%). For this reason, L1I hits are not a good proxy for *usefulness* of a code line.

Therefore, we need to more strictly follow Definition 1 and use commit information to drive the detection of useful code lines in the L2C. While simple in principle, this idea presents multiple challenges in a real-world design that must contend with wiring, bandwidth, and energy costs. Specific challenges include (1) minimizing signaling throughout the core pipeline (wiring, bandwidth, energy); (2) reducing MMU lookups for address translation and L1I cache accesses (bandwidth, energy); (3) minimizing signaling between the L1I and L2C (bandwidth, energy) since a line may see frequent reuse in the L1I cache, which could trigger many needless accesses to the L2C if every committed instruction triggered an RRPV update signal to the L2C. In short, an ideal scheme would minimize signaling throughout the pipeline and avoid redundant accesses to the MMU, L1I cache, and L2C. We next present a low-cost microarchitectural scheme that meets these constraints and makes commit-driven management of unified caches feasible in a real-world CPU design.

## V. Bumper Design

<span id="page-5-0"></span>Bumper is a microarchitectural scheme that capitalizes on the insights from our previous analysis. Bumper allows the L2C to distinguish between *useful* and *useless* code lines by carefully orchestrating the propagation of *usefulness* information across the CPU pipeline and cache hierarchy at a low implementation cost and with negligible signaling overhead.

At a high level, Bumper inserts all L2C code lines in the vulnerable position (RRPV=3) and subsequently promotes only the code lines identified as *useful*, thereby increasing their likelihood of remaining in the cache long enough to be reused. Bumper extends a typical microarchitecture with the ability to *track useful code lines* (Section V-A) and exploits this knowledge to *inform L2C management* (Section V-B). Figure 9 shows how Bumper extends a microarchitecture to correlate L2C lines with committed instructions; this design minimizes the impact on the microarchitecture in terms of both added storage and signaling. The rest of this section presents how Bumper operates, referring to the circled markers in Figure 9.

