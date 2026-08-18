# *A. DRAM Faults and Scaling Errors*

Physical defects (*faults*) in DRAM can cause *errors* that deviate memory cells from their intended values. If uncorrected, these errors may escalate into system *failures*, as either *detectable uncorrectable errors* (DUEs) or *silent data corruptions* (SDCs). We group DRAM faults into three categories: inherent faults, operational faults, and rowhammer faults.

Inherent faults originate during manufacturing and are traditionally mitigated through testing and row/column remapping. However, continued DRAM scaling has introduced new fault modes that cannot be fully screened before deployment. One prominent example is *variable retention time* (VRT) cells. VRT originates from leakage-induced charge loss in DRAM cell transistors [46]. When a VRT cell enters a sub-nominal retention state, it may lose charge before the scheduled refresh and produce retention errors [8], [63], [67].

VRT cells pose a non-negligible reliability concern and are difficult to tolerate in the field. In modern technology, the probability that a cell enters a sub-nominal state exceeds 10<sup>−</sup><sup>8</sup> , 1 and this is expected to increase with further scaling [47], [61]. Because VRT cells are variable in nature and DRAM chips contain billions of cells, exhaustive pre-deployment screening is impractical [19], [29].

Measured DDR4 retention failures are strongly asymmetric, with only 0.005% corresponding to 0→1 flips at 60°C under a 1s refresh interval [41], [42]. Thus, VRT-induced retention errors are *unidirectional* [66]: they arise when a charged cell loses charge prematurely. Many prior works exploit this unidirectionality to reverse-engineer on-die ECC mechanisms [66], improve power efficiency [38], and enhance reliability [16].

Operational faults arise during DRAM operation due to particle strikes, device aging, or other physical fault modes. Depending on their manifestation, they may affect a single bit or multiple bits. Systems mitigate these faults using ECC and fault-removal mechanisms such as post-package repair (spare rows), OS-level page retirement, or module replacement. Faults that corrupt many bits require stronger ECC to prevent DUEs or SDCs.

Rowhammer faults result from repeated row activations that induce charge disturbance in adjacent rows [39], [50], and remain an active area of concern for both reliability and security [37], [45], [55], [62], [70], [79], [80]. Rowhammer mitigation is orthogonal to the mechanisms proposed in this paper: row activation commands remain under host control in the all-bank PIM architecture, and our design does not modify existing activation-driven mitigation such as PRAC [26].

