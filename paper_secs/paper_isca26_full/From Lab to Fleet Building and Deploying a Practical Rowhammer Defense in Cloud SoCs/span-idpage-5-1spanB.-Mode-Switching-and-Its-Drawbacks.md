# <span id="page-5-1"></span>*B. Mode Switching and Its Drawbacks*

In each refresh window (32ms for DDR5), Sigries offers full Rowhammer protection as long as there is no transition from light to heavy mode. If such a transition occurs, Sigries cannot guarantee that a row's counter that reaches the Rowhammer threshold will be sampled and mitigated, but this reduced guarantee lasts for only one tREFW. This property is critical to the security analysis of our hybrid scheme, as it enables precise bounds on the duration over which it may provide less than full protection. In practice, we configure Sigries to reduce the cumulative exposure time to *less than an hour per year* assuming a worst-case attacker with full control. We chose this configuration based on how long today's best Rowhammer attack fuzzers, such as those in [\[27\]](#page-13-4), [\[28\]](#page-13-5), take to identify suitable attack patterns. Because Sigries is reconfigurable at runtime, fleet-wide telemetry can further tighten these limits if needed.

While our guarantees assume worst-case conditions, three practical challenges make Rowhammer attacks harder to execute. First, precise control is difficult in the cloud: complex SoC designs, unpredictable memory controller behavior, and heavy DRAM traffic all add noise that disrupts row activation timing. Second, Rowhammer fuzzers have been evaluated only on simpler hardware (e.g., few sockets, channels, and DIMMs), where researchers have far more control than attackers would on a cloud server. These challenges make our one-hour-per-year bound a conservative upper limit. Third, Sigries introduces randomness in the thresholds used to switch between modes. This adds a layer of complexity for attackers. Our security analysis does not depend on randomness to guarantee protection across the fleet.

Sigries transitions from light to heavy mode whenever the spillover counter is one less than the Rowhammer threshold. This transition occurs during a row activation command. Figure [1](#page-5-0) shows a timeline of Sigries switching from light mode (Misra-Gries) to heavy mode (row-sampling). The switching occurs in the third refresh window, and the exposure time is limited to one refresh window.

<span id="page-6-1"></span>![](_page_6_Figure_0.jpeg)

Fig. 2: Transitioning between modes.

Unlike the transition from light to heavy mode, which introduces a brief window of vulnerability, transitioning back to light mode is completely safe. In Sigries, this transition always occurs at the end of a refresh window (32ms in DDR5). However, an important design choice was deciding for how many refresh windows Sigries should stay in heavy mode. Ideally, Sigries should remain in heavy mode *as long as* the light mode is inadequate to handle the ongoing workload (i.e., it would not need to transition back to heavy mode).

In heavy mode, Sigries uses the counter table for *shadow counting*, where it follows the Misra-Gries logic but does not use it to issue DRFMs. Instead, shadow counting is used to monitor whether the spillover counter continues to reach the Rowhammer threshold during each refresh window. After spending a fixed number of refresh windows in heavy mode, Sigries evaluates the fraction of these windows in which the spillover counter was saturated. If this fraction is too high, Sigries remains in heavy mode, effectively restarting the shadow counting process. Figure 2 illustrates the state diagram of Sigries's transition process.

Table II summarizes how Sigries's hybrid design meets its design requirements.

#### <span id="page-6-3"></span>C. Worst-case Attack Strategy

A worst-case attack strategy seeks to maximize the probability of inducing Rowhammer bit flips. For Sigries, such an attack combines the worst-case patterns targeting both components: Misra–Gries and sampling. For Misra–Gries, the worst-case scenario is a k-sided attack, where k is one greater than the size of the counter table. For sampling, the worst case is a single-sided attack. An optimal adversary would alternate between these two patterns, precisely synchronized with Sigries's mode transitions.

To make such synchronization difficult, Sigries randomizes its transitions. Specifically, the minimum duration spent in heavy mode is drawn from a small randomized range, with each sub-bank independently selecting a value for each transition *each time*. Nevertheless, our security analysis does not rely on the difficulty of synchronization; instead we assume a worst-case adversary capable of perfectly synchronizing with Sigries and adapting its attack patterns accordingly.

#### V. IMPLEMENTATION

Sigries uses several small counter tables per bank, where each counter table tracks a *sub-bank*. Each memory controller is equipped with SRAM organized in a set-associative way, where each set stores the counter table of a sub-bank. The entries in the counter tables correspond to the row address, the

<span id="page-6-2"></span>

| Requirement       | Sigries                                                               |
|-------------------|-----------------------------------------------------------------------|
| Min. overhead     | Minimal overhead in common case; heavy-mode under attack only         |
| No perf. outliers | Both modes have no arbitrary perf. spikes                             |
| Liveness          | Both modes offer liveness even when under attack                      |
| Low cost          | Under-provisioned counter tables meets hardware cost reqs             |
| Flexibility       | Under-provisioning works with various configs                         |
| Config. security  | Config. mode transitions with analyzable security vs. perf. trade-off |

TABLE II: Sigries meets design requirements.

counter value, and a *lock bit*. When a counter value reaches the Rowhammer threshold, Sigries issues a DRFM, resets the counter value, and sets the lock bit to ensure that the entry is no longer swapped with the spillover count.

Misra-Gries clears all counter tables at the end of each refresh window. Because counters are stored in SRAM, clearing them requires a read followed by a write. This wastes power when the memory controller is idle or lightly loaded as the counters may already be zero. In addition, these SRAM operations would, in some cases, limit the rate at which the memory controller could issue activates and thus lower its bandwidth. Instead, Sigries clears a sub-bank's counters by setting its "clear" bit; upon the next row activation, the controller clears all counters in that sub-bank and the clear bit.

Sigries also stores a spill overflow countdown (not to be confused with the spillover counter) and a heavy mode countdown. When in heavy mode, the spill overflow countdown is decremented if not zero in each refresh window in which the shadow spillover count reaches the Rowhammer threshold. The heavy mode countdown indicates the number of refresh windows a sub-bank stays in heavy mode. When this countdown reaches zero, Sigries checks whether the spill overflow counter value is *non-zero* to switch to light mode. If not, Sigries remains in heavy mode re-initializing its state.

The DDR5 specification mandates that, on average, no two DRFM commands should target the same bank/row address within a duration of 7.8 µs [31]. This serves as a ratelimiting measure to address concerns that DRFM commands themselves could be exploited to hammer memory. To comply, Sigries maintains a per-bank table to track the row addresses of previously issued DRFMs. The table is sized to accommodate the maximum number of DRFM commands that may be issued within a 7.8 µs interval.

#### <span id="page-6-0"></span>A. Algorithm Verification

We verify our algorithm's correctness using Dafny [47], a language and tool for writing code, expressing specifications, and mechanically checking properties with formal methods. In Dafny, we model the hardware and our algorithm and prove key properties, shown in Table III. The most critical invariant is that each row's access count—a value reset at the start of a refresh window or after issuing a DRFM—remains below the threshold.

The algorithm modeled in Dafny makes two simplifications from the hardware implementation. First, the proof assumes no parity errors in the memory controller hardware (see Section V-B). Second, the proof models never exiting light

<span id="page-7-1"></span>

| Invariant name               | Description                                                                                       |
|------------------------------|---------------------------------------------------------------------------------------------------|
| RowAccessesLessOrEqualThresh | Any row's access count < Rowhammer threshold.                                                     |
| SpillCountLessThanThresh     | Spill count < Rowhammer threshold.                                                                |
| EntriesBtwSpillCount&Thresh  | Each entry's count ≤ Rowhammer threshold +                                                        |
| 1                            | each unlocked entry s count ≥ spill count.                                                        |
| AddressesDistinct            | No two entries refer to the same row address.                                                     |
| StateCorrespondsToHistory    | For each entry, # of address accesses ≤ its count + for each address with no corresponding entry, |
|                              | # of accesses < spill count.                                                                      |
| AccessesSinceLastDRFM        | Every address accessed since DRFM has an entry +                                                  |
|                              | that entry is locked +                                                                            |
|                              | its count is equal to # of accesses since last DRFM.                                              |

TABLE III: Invariants proved in Dafny. The proof demonstrates that the conjunction of all these is an inductive invariant. (We also prove two properties not listed here but discussed in the text.)

```
accessesByAddr: address accesses since the last interval
     start or issued DRFM
// accessesSinceLastDRFM: \# of address accesses since last
datatype History =
 | History(accessesByAddr: map<Address, nat>,
          accessesSinceLastDRFM: map<Address, nat>)
function InitializeHistory () : History
 History (map [], map [])
function UpdateHistoryToReflectDRFM (h: History, addr:
     Address) : History
       ory(h.accessesByAddr[addr := 0], haccessesSinceLastDRFM[addr := 0])
 History(h.accessesByAddr[addr
function UpdateHistoryToReflectAccess (h: History, addr:
     Address) : History
 var newAccesses := if addr in h.accessesByAddr then h.
    accessesByAddr[addr] + 1 else 1;
History(h.accessesByAddr[addr := newAccesses],
        if addr in h.accessesSinceLastDRFM then
        accessesSinceLastDRFM[addr] + 1] else
          h.accessesSinceLastDRFM)
```

Fig. 3: Hardware specification written in Dafny.

mode (i.e., never using sampling), so it only proves invariants about what happens before heavy mode is entered.

The hardware specification, shown in Figure 3, is written in terms of a History that maintains every address's access count and updates those counts when various events occur. The algorithm's specification, shown in Figure 4, is written as imperative code that models the Misra-Gries algorithm.

We prove correctness by induction. First, we prove that the algorithm to initialize the state satisfies all our invariants. Next, we prove that the hardware algorithm run on every access preserves those invariants. This induction proof is performed on a functional specification for the algorithm. To verify that the untrusted functional specification models the trusted imperative specification faithfully, we make the functional specification a post-condition of the imperative specification. For example, UpdateStateOnAccess in Figure 4 has a post-condition that ensures it satisfies UpdateStateOn-AccessSpec. In a further proof, we prove two other useful properties of our algorithm: (1) it issues at most one DRFM per access, and (2) any two DRFMs to the same address within the same interval are separated by at least the threshold number of accesses to that address.

Verifying the algorithm gives us confidence in its correct-

```
method InitializeState (old_s: State) returns (s: State)
  ensures InitializeStateSpec(s) {
  s := old_s.(clear := true);
method UpdateStateOnAccess (old_s: State, addr: Address)
    returns (s: State, drfmsIssued: nat)\nensures_UpdateStateOnAccessSpec(old_s, addr, s,
          drfmsIssued) {
  s := old_s;
drfmsIssued := 0;
  if s.clear
     s := s.(spillCount := 0);
     s:= s.(entries := seq(|s.entries|, (i requires 0 <= i <

       := s.(clear := false);
  if s.spillCount == Threshold() - 1 {
    s := s.(spillCount := s.spillCount + 1);
    s := s.(sampling := true);
  else if n: nat : | n < |s.entries| && (s.entries[n].count
          .....................................
     if s.entries[n].count == Threshold() - 1
  var newEntry := s.entries[n].(count :=
                                  s.entries[n].(count := 0, locked :=
         := s.(entries := s.entries[n := newEntry]);
f!s.sampling {
drfmsIssued := drfmsIssued + 1;
     élse {
       var oldEntry := s.entries[n];
var newEntry := oldEntry.(count := oldEntry.count + 1);
       s := s.(entries := s.entries[n := newEntry]);
  else if n: nat :| n < |s.entries| && s.entries[n].count ==

     s := s.(entries := s.entries[n := newEntry]);
  selse if n: nat :| n < |s.entries| && s.entries[n].count ==
    s.spillCount && !s.entries[n].locked {
    var newEntry := s.entries[n].(addr := addr, count := s.
        spillCount + 1);
    s := s.(entries := s.entries[n := newEntry]);</pre>
  élse {
     s := s.(spillCount := s.spillCount + 1);
```

Fig. 4: Specification of our algorithm in Dafny.

ness and helps identify bugs prior to the hardware implementation. During this process, we identified and fixed three issues: (1) the clear bit was never cleared, (2) the lock bit was incorrectly set to 0 instead of 1 on a DRFM, and (3) finding an empty entry in a counter table involved checking if the entry's count was zero; however, the correct check should have also verified that the entry was unlocked (i.e., *lock* bit set to 0). Catching and fixing these issues shows the value of algorithm verification; RTL verification would not have caught them since it does not check algorithm invariants.

