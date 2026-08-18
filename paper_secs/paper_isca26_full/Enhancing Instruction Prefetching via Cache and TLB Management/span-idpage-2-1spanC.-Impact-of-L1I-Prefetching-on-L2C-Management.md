# <span id="page-2-1"></span>*C. Impact of L1I Prefetching on L2C Management*

This section evaluates the potential performance gains of optimizing the management of lines inserted in lower-level caches by L1I prefetches. This study targets the L2C and not the LLC since we found larger headroom to optimize the management of prefetched code lines in L2C. Regarding the L1I prefetchers, we consider the EPI, FNL+MMA, and Barc¸a prefetchers, similar to [Section III-B,](#page-2-0) configured to freely prefetch across page boundaries since [Section III-B](#page-2-0) shows that permitting L1I prefetchers to cross page boundaries improves IPC over the conservative scenario that discards them.

[Figure 3](#page-3-1) evaluates the impact of inserting code lines fetched by L1I prefetches in the L2C by considering two idealized

<span id="page-3-1"></span>![](_page_3_Figure_0.jpeg)

Fig. 3: Geomean speedups of two ideal scenarios forcing L2C hits for lines fetched by i) L1I page-cross prefetches (Ideal L2C (PGC Pref)) and ii) L1I prefetches (Ideal L2C (All Pref)).

scenarios: i) *Ideal L2C (PGC Pref)*, where code lines fetched by page-cross prefetches are not inserted in L2C until a demand L2C access requests them. These entries are instead placed in an infinite buffer located alongside the L2C. When a demand access misses in L2C, we look up that buffer. If we have a hit, we magically insert that line in L2C. This ideal scenario quantifies how much performance can be extracted if code lines fetched by L1I page-cross prefetches incur no L2C pressure. ii) *Ideal L2C (All Pref)*, where code lines fetched by L1I prefetches, both in-page and page-cross, are not inserted in L2C until a demand L2C access requests them. This scenario quantifies the performance if code lines fetched by L1I prefetches incur no L2C pressure. The speedups in Figure 3 are computed over the *Permit Page Cross* version of each considered prefetcher in Figure 2.

Figure 3 shows that the L2C pressure placed by L1I prefetches undermines their potential to improve performance. For example, EPI combined with *Ideal L2C (All Pref)* scenario delivers a 9.5% speedup over the *Permit Page Cross* baseline. The main reason for the performance gap between *Permit Page Cross* and *Ideal L2C (All Pref)* stems from the fact that a large number of prefetched code lines do not serve any demand L2C access. The *Ideal L2C (All Pref)* speedups are higher than the ones delivered by the *Ideal L2C (PGC Pref)* scenario, highlighting that judiciously managing prefetched code lines in L2C can bring higher performance improvements than only improving the management of L2C lines fetched by L1I prefetches that cross page boundaries.

1) Reuse of Prefetched Code Lines: To support our claim that many prefetched code lines exhibit low reuse in L2C—and therefore harm performance—we measure how many demand L2C accesses are served by lines brought into L2C by L1I prefetches across the evaluated server workloads. Figure 4 reports, for all L1I prefetchers as well as the baseline FDiP prefetcher [13], the number of demand L2C accesses served by these prefetched lines. The x-axis groups prefetched L2C code lines by the number of demand accesses they serve, while the y-axis shows the fraction of lines in each group.

Figure 4 shows that all considered L1I prefetchers behave similarly when it comes to serving demand L2C accesses. Focusing on EPI, we observe that, on average, 36.1% of prefetched code lines in the L2C remain unused, *i.e.*, serve no accesses, while 51.6% serve between one and eight accesses. Additionally, 11.5% of these lines handle more than eight accesses while 0.8% of them serve more than 128 accesses during their time in L2C. The main takeaways of this study

<span id="page-3-2"></span>![](_page_3_Figure_6.jpeg)

Fig. 4: Breakdown of the number of demand L2C accesses served by code lines fetched in L2C by L1I prefetch requests.

are that i) in many cases, L1I prefetch requests insert deadon-arrival lines in L2C causing pollution (on average 36.1%) and ii) a non-negligible fraction of the L1I prefetch requests bring code lines in L2C that serve a large number of demand L2C accesses, thus being very valuable for performance.

Finding 2: Cache lines fetched in L2C by L1I prefetches show variable behavior, calling for a smart policy that anticipates the reuse of these prefetched code lines.

We do not target the management of code lines fetched in L2C by demand accesses, as our analysis reveals low potential over prior art [6]. Section VI-C shows that exclusively applying our proposal to prefetched code lines yields higher IPC than applying it to both demand and prefetch code lines.

## <span id="page-3-0"></span>D. Putting Everything Together

Sections III-B and III-C highlight that contemporary L1I prefetchers deliver good performance gains. However, two factors undermine their potential for delivering outstanding benefits: i) the address translation latency of L1I page-cross prefetches and ii) the low reuse of a large fraction of code lines fetched in L2C by L1I prefetches. Section IV proposes a novel scheme that addresses our analysis findings and improves the performance of any L1I prefetching scheme.

