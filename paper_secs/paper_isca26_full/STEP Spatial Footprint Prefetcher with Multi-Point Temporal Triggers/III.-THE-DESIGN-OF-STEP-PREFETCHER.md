# III. THE DESIGN OF STEP PREFETCHER

#### A. Design Overview

STEP treats spatial footprint prefetching as a sequence of trigger-time decisions within a region's lifetime. Rather than binding prefetching to one fixed event, STEP progressively observes each region at three trigger points. And because it uses offset as the event key, we call these three points: the first-offset event (FOE), the second-offset event (SOE), and the third-offset event (TOE). As the region is observed over time, the amount of context available to the prefetcher increases: FOE provides only the first offset, SOE provides the first two offsets, and TOE provides all three. At each point, STEP uses

![](_page_3_Figure_0.jpeg)

Fig. 3: Illustration of three scenarios with different prefetching confidence based on footprint pattern convergence.

the currently available event information to query historical footprints and then decides whether the evidence is already strong enough to issue prefetches. If the matched histories are sufficiently consistent, STEP issues prefetches immediately; otherwise, it waits for the next trigger point to obtain additional evidence.

This staged behavior is realized with two main components. First, a lightweight prefetch-confidence evaluator determines whether matched historical footprints have sufficiently converged at the current trigger point. Second, a unified hardware organization retains the information needed to support FOE, SOE, and TOE without replicating separate history tables for each event. FOE requires additional support because it is the earliest and least constrained trigger; STEP therefore augments FOE matching with hashed PC information to improve disambiguation at low cost. In the remaining subsections, we first present the prefetch-confidence evaluator that determines whether a trigger point is mature enough for prefetching. Next, we discuss the special handling required for FOE, which is the earliest and most under-constrained trigger. Finally, we describe the hardware structures, dataflow, and overhead of the complete design.

