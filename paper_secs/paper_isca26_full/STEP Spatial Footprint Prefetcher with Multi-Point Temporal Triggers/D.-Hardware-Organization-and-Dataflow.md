# D. Hardware Organization and Dataflow

**Architecture:** Figure 4 illustrates the architecture of the STEP prefetcher. The design consists of four main structures

![](_page_4_Figure_0.jpeg)

Fig. 4: Structure of the STEP prefetcher, including the Filter Table (FT), Accumulation Table (AT), Pattern History Table (PHT), Prefetch Buffer (PB), and the Prefetch-Confidence Evaluator. Solid arrows indicate the learning/update flow and dashed arrows the prefetch flow; circled numerals annotate the order of operations.

commonly used in footprint prefetchers—a Filter Table (FT), an Accumulation Table (AT), a Pattern History Table (PHT), and a Prefetch Buffer (PB)—together with the prefetch-confidence evaluator introduced in Section III-B. FT, AT, and PHT are all set-associative structures.

Instead of handling only the first access, the FT filters out pages with fewer than three accesses. It is indexed by the hashed page number and stores the first and second access offsets. The AT tracks all active pages and accumulates their footprint patterns, indexed by the hashed page number and stored as a 64-bit bit vector (one bit per cache line). For PHT, we observe that TOE subsumes SOE and FOE, so a single PHT suffices: store TOE-associated footprints once and derive earlier-event matches from the same entry. Concretely, the PHT is indexed by the first offset and tagged with the second and third offsets. At lookup, the tag is truncated according to the event: SOE checks the upper 6 bits, and TOE checks the whole tag. The PB stores footprints indexed by page number. When the prefetch queue is full, pending prefetch requests are temporarily buffered in the PB instead of being discarded, allowing them to be reissued upon future access triggers.

**Process:** The prefetcher operates in two flows—Learning and Prefetching—both triggered by new page accesses. In Fig. 4, solid lines represent the learning flow and dotted lines represent the prefetch flow.

- 1) Learning Flow: When a demand access arrives, we first probe the Accumulation Table (AT). If an entry exists—meaning the page's footprint is currently being learned—we set the bit corresponding to the access offset in the footprint vector (1). If no entry exists, the page is treated as new and the access is forwarded to the Filter Table (FT) (2) to screen for pages with negligible activity. There are three cases for FT looking up:
  - FT miss: allocate a new entry; extract the page offset and store it as the first offset; mark the second offset

- invalid (e.g., 64).
- FT hit with invalid second offset: this access provides the second offset; update the field.
- 3) FT hit with valid second offset: this is the third access; the page passed the FT. Then, send the new offset, along with the first two offsets, to the AT and allocate a new AT entry to begin accumulating the full footprint (3); clear the FT entry.

When a new AT entry is allocated, it evicts an old one. This eviction indicates the old page's pattern is stable and complete. The evicted footprint, along with its first three offsets, is written to the PHT (4), making it available for future prefetching.

- 2) Prefetching Flow: Prefetching can be triggered by FOE, SOE, or TOE. As shown in Figure 4, the FT forwards the request containing the offsets, PC and event ID to the PHT (1). The PHT recognizes the event type according to the event ID and extracts the corresponding data, and operates as follows:
  - 1) First Offset Event (FOE): PHT extracts the first offset and PC and fetches the most recent N (default N = 3) entries matching FO+PC. If only one entry matches, STEP additionally checks the entry's maturity flag; if the entry is still immature, FOE issuance is suppressed, and the request is deferred to later trigger points. Otherwise, the matched entries are then passed to the Prefetch-Confidence Evaluator introduced in Section III-B (2). The evaluator will return the confidence (3). If confidence is high, push prefetch candidates according to the intersection of the most recent matched footprints to the prefetch buffer and issue, and notify the FT that prefetches for this page have been issued by FOE (4-5). If confidence is low, take no action and wait for the SOE.
  - 2) Second Offset Event (SOE): When the FT entry re-

ceives the second offset, it first checks the issued field. If 0, it issues a PHT request. Run prefetch confidence evaluation on patterns matching FO+SO ( 2 - 3 ). If confidence is still low, wait for TOE; otherwise, prefetch the footprints intersection and notify FT ( 4 - 5 ).

3) Third Offset Event (TOE): With three offsets available, a PHT lookup using the full tag is highly specific. On hit, issue the full prefetch ( 4 - 5 ). On miss, no prefetch action for this page entry.

TABLE I: Storage Overhead

| Component              | Entry Contents                                                                 | bits/<br>entry | Total<br>entries | Total<br>(KB)<br>2.08 |
|------------------------|--------------------------------------------------------------------------------|----------------|------------------|-----------------------|
| FT                     | Tag (36b), LRU (3b), Hashed<br>PC (12b), Offsets (6+7=13b),<br>Issued (1b)     | 65             | 256              |                       |
| AT                     | Tag (36b), LRU (3b), Hashed<br>PC (12b), Offsets (6×3=18b),<br>Footprint (64b) | 133            | 128              | 2.12                  |
| PHT                    | Tag (2×6=12b), LRU (3b),<br>Footprint (64b), Hashed PC<br>(12b), maturity (1b) | 92             | 512              | 5.88                  |
| PB                     | Tag (36b), LRU (3b),<br>Footprint (64b)                                        | 103            | 32               | 0.41                  |
| DPCT                   | Hashed PC (12b), LRU (3b)                                                      | 15             | 8                | 0.015                 |
| Total Storage Overhead |                                                                                |                |                  | 10.50                 |

