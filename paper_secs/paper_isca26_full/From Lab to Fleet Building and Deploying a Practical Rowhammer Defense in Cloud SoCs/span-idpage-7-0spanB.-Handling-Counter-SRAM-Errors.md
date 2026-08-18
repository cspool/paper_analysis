# <span id="page-7-0"></span>B. Handling Counter SRAM Errors

Within the memory controller, data and critical state are protected from soft errors via an Error Correcting Code (ECC), while other items are protected by an error detecting code (parity). Regardless of the method chosen to protect an item, the potential exists for an error to be uncorrectable, or for an error to not be detected and thus potentially cause Silent Data Corruption (SDC). While ECC provides better protection than parity, it requires extra storage and correction logic. Since this increases the design's area and power, the decision as to the appropriate means of protecting a given item is based on a

<span id="page-8-2"></span>

| Sub-Bank Mode Errant Item |                                                                         | Remediation                                                                                                                         |  |  |
|---------------------------|-------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|--|--|
| Light Mode                | Way Address<br>Way Count<br>Spill Count                                 | Switch to Heavy Mode<br>Issue DRFM, set Way Count to max value<br>Switch to Heavy Mode                                              |  |  |
| Heavy Mode                | Way Address<br>Way Count<br>Spillover Count<br>Spill Overflow Countdown | Decrement Spill Overflow Countdown<br>Set Way Count to maximum value<br>Decrement Spill Overflow Countdown<br>Restart Sampling Mode |  |  |
| Unknown                   | Heavy Mode Countdown                                                    | Restart Heavy Mode                                                                                                                  |  |  |

TABLE IV: Handling parity errors

reliability analysis of not only the memory controller but the SoC into which the controller is incorporated.

Given the amount of state stored within the SRAMs, along with the state's expected lifetime, parity provides sufficient protection from soft errors. To enhance protection, the SRAMs employ column multiplexing such that items protected by parity are bit-interleaved with each other: if a soft error flips multiple physically adjacent bits it impacts multiple parity protected items. This reduces the possibility of two bit flips occurring in the same item and thus causing SDC.

As detailed in Table IV, the algorithm handles detected parity errors by prioritizing Rowhammer protection over a temporary loss of MC performance. If multiple items simultaneously encounter parity errors, the most conservative remediation is applied. Table IV provides a simplified version of the complete implementation, detailing how the algorithm handles single-item errors. We also examined the impact of SDC errors; Figure 5 shows a small sample of our analysis.

#### <span id="page-8-0"></span>C. Sigries Parameters Configuration

To implement an effective Rowhammer protection, Sigries's parameters must be configured correctly.

Size of Misra-Gries counter tables. In Sigries, the counter tables are significantly under-provisioned relative to a full Misra-Gries implementation, such as the one used by Graphene [62]. We chose the size of the counter tables to accommodate benign workloads while minimizing their area overhead. Sigries's counter tables do not have more than a few dozen entries per sub-bank.

**Rowhammer Threshold:** DRAM customers with large deployments, such as Microsoft, routinely run a DRAM qualification process. The goal is to ensure that DRAM parts at a new process node are safe, reliable, and fit for production use in a given system and workload *before* it is deployed at scale. As part of this process, the Rowhammer threshold is characterized across DIMM samples from multiple DRAM vendors. Sigries is then configured using a threshold derived from these qualification results.

Some prior work on Rowhammer defenses assumes thresholds in the low hundreds or even lower, based on projected trends. While thresholds have indeed decreased significantly over the past decade, we do not expect server DRAM to reach such extreme values. Although the exact thresholds for Sigries are sensitive and cannot be disclosed, Sigries is not designed to operate at thresholds in the low hundreds or below. If such thresholds were to materialize, Sigries would remain functionally correct, but it would spend most of its time in

heavy mode, which lies outside its intended design point and would result in higher overhead.

**Sampling rate:** To set the sampling rate value used in heavy mode, we rely on our prior work that describes in depth how to derive a sampling rate given a target Rowhammer threshold and a probability of escaping sampling, set to a very low value [72].

**Minimum duration of heavy mode:** While in heavy mode, Sigries never transitions back to light mode until a minimum number of refresh windows (tREFW) have elapsed. As described in Section IV-B, Sigries suffers from a small exposure time upon mode switching, and thus Sigries's configuration ensures that under a *worst-case* attack strategy (see Section IV-C), Sigries reduces its cumulative exposure to less than an hour per year.

Threshold on the number of refresh windows in heavy mode in which the shadow counters are overwhelmed. In each refresh window spent in heavy mode, Sigries uses shadow counters to determine whether they remain overwhelmed, that is, whether the spillover counter reaches the Rowhammer threshold. Sigries transitions back to light mode only if the fraction of such windows falls below a threshold. We set this threshold to be very low so that the fraction of refresh windows in which the shadow counters are overwhelmed is much lower than 1%.

Sigries does not necessarily transition back to light mode once it reaches the minimum duration in heavy mode. While in heavy mode, Sigries continues to track the behavior of the sub-bank's (underprovisioned) Misra-Gries counter table. If the table is overwhelmed (i.e., its spillover counter reached the Rowhammer threshold), Sigries treats this as an indication that the attack may still be ongoing.

