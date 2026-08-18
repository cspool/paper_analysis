# B. Prefetch-Confidence Evaluation

At each trigger point, STEP may observe multiple historical footprints that match the currently available event information. The role of the confidence evaluator is to decide whether these matches are already consistent enough to support issuance or whether the prefetcher should defer to a later trigger point.

When historical footprints aligned by the same event and key largely agree on the remaining lines to be accessed, issuing a full footprint captures most of the remaining misses at low risk. When they diverge beyond the early lines, issuing early often fetches many wrong lines and causes pollution. Figure 3 illustrates these representative cases. We therefore issue prefetches only when the matched histories are sufficiently convergent; otherwise, we defer to the next trigger point.

To quantify confidence, STEP applies a simple similarity test over the last N matched footprints returned by the PHT (default N=3). We compare the most recent footprint with the remaining N-1 footprints using Jaccard similarity:

$$J(A,B) = \frac{|A \cap B|}{|A \cup B|}.$$

If all N-1 similarities exceed a threshold T (default T=0.75), we call the set convergent and issue prefetches at the current event; otherwise, we defer to the next event. We issue the intersection of the matched footprints rather than a union, prioritizing accuracy over coverage under bandwidth-constrained conditions. Bounding the comparison to the most recent N entries keeps the hardware small and avoids relying on stale patterns from old phases.

### C. First-Offset Event Support

Handling the first-offset event (FOE) is inherently difficult because a single touch provides little context, making naive full-footprint issuance prone to overfetch and pollution. FOE is therefore the most under-constrained trigger point in STEP. To improve FOE reliability, STEP adds two FOE-specific supports. First, it augments FOE matching with hashed PC information to partially disambiguate calling contexts at low cost. Second, it adds a maturity check for the single-match cold-start case, preventing an early learned pattern from being treated as already stable before recurrence has been observed.

The hashed PC change is minimal in cost: it is not used as a tag and is stored once per PHT entry and per FT/AT entry. With this PC information, we can bring part of the prefetching at SOE/TOE forward.

Another challenge is that FOE remains vulnerable to the cold start when the lookup returns exactly one matched history entry. Unlike the multi-match case, no cross-pattern disagreement can be observed, so a single early learned footprint may be falsely treated as mature even though other divergent patterns have not yet been learned. STEP therefore adds a one-bit maturity flag per learned pattern, checked only in this single-match FOE case. Newly inserted entries start as immature; if the sole matched entry is immature, STEP defers issuance to later trigger points. An entry becomes mature only after recurrence is observed. In our implementation, we approximate recurrence by marking a newly inserted pattern as mature when its hashed PC matches that of the displaced entry in the same history position, indicating that this calling context has appeared more than once rather than only during cold start.

