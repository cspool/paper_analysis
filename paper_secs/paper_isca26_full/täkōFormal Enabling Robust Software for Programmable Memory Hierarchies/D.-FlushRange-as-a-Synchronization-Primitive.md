# *D. FlushRange as a Synchronization Primitive*

While the use of an OnMiss-generated value (Me) by a Rcb or Wcb induces an hb edge between the events (§IV-F2), an OnEvict or OnWB for a phantom address can execute anytime after the address is brought into the cache (for OnEvict) or after the address is written to in the cache (for OnWB). This freedom allows races between accesses in these eviction callbacks and those in core program threads.

Consider the wbr litmus test from Figure 11a, where the bolded FlushRange (i2) is omitted. This program distills a use case of phantom memory as a write-combining buffer for scatter-updates [55], which are published back to regular memory via OnWBs. In this test, Core 0 updates the buffer at phantom address [x] in (i1), and (i5) publishes the update to address [y] in regular memory on a writeback. Core 0 also reads the published update in (i3).

Figure 11b shows that since the OnWB of [x] can execute anytime after (i1), (i5) and (i3) are unordered by hb, causing a race. If we add the FlushRange (i2) from Figure 11a to the wbr test to give us the wbf test, the race is eliminated. Specifically, this FlushRange must either commit before the OnMiss of [x] or after the OnWB of [x], as enforced by EbWf (Figure 6). Committing the FlushRange before the OnMiss causes a forbidden cycle in cbo (CboWf1). Meanwhile, Figure 11c depicts the execution where it commits after the OnWB of [x]. Here, an hb edge is added between

| Core 0                                                     | [x].OnMiss   | [x].OnWB     |  |  |
|------------------------------------------------------------|--------------|--------------|--|--|
| (i1) [x] ← 1                                               | (i4) [x] ← 0 | (i5) [y] ← 1 |  |  |
| (i2) FlushRange[x]                                         |              |              |  |  |
| (i3) r1 ← [y]                                              |              |              |  |  |
| wbr<br>(without (i2)): program racy under our tak¨ o MCM ¯ |              |              |  |  |

wbr (without (i2)): program racy under our tak¨ o MCM ¯ wbf (with (i2)): no race, r1 = 0 forbidden by our MCM

![](_page_8_Figure_10.jpeg)

Fig. 11: (a) The wbr (Writeback Race) and wbf (Writeback Flush) litmus tests. (b) wbr execution where the accesses to [y] race because the OnWB is non-blocking. (c) wbf eliminates the race using a FlushRange, and forbids r1=0.

the E<sup>e</sup> and the Fl as per Figure 6's definitions of eb and hb. Combining this edge transitively with the two yellow sb edges gives us an hb edge between (i5) and (i3), eliminating the race. We now also have a cycle in hb and fr between (i3) and (i5), which violates the Vis axiom, forbidding this execution's outcome of r1=0. Our MCM thus formalizes how FlushRange synchronization can be used to eliminate races.

The race in wbr illustrates a key difference between tak¨ o¯ and prior works like IMO [20] and EcMon [45] that allow userspace traps for cache events. In these works, traps effectively have *function call* semantics: they interrupt a core thread (either immediately after a cache event or at a predetermined execution point), execute a handler, and then return control. Thus, the ability of traps to concurrently execute with core threads is greatly reduced if not eliminated. In contrast, in tak¨ o, callbacks have ¯ *thread* semantics [55]: they execute on dedicated engines in parallel with core program threads. As a result, conflicting accesses across tak¨ o callbacks and core ¯ program threads can be races. This would not be the case in IMO and EcMon where callbacks are not separate threads.

