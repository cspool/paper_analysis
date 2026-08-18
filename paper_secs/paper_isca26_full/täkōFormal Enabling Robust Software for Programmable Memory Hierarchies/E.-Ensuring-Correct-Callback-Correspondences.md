# *E. Ensuring Correct Callback Correspondences*

As Figure 7c shows, we have now ensured correct values for the phantom reads in our execution. However, note that Figure 7c contains two OnMiss callbacks for [x] without [x] being evicted from the cache in between. This is impossible, and so an eviction callback (OnEvict or OnWB) for [x] must run in between the two OnMiss callbacks.

To enforce this constraint, we require that there must be an OnEvict or OnWB (i.e. an Es/E<sup>e</sup> pair) between the end of an OnMiss and the beginning of another OnMiss to that same address. Graphically, this constraint requires the dotted events in the diagram below to exist between the M<sup>s</sup> and M<sup>e</sup> (thd refers to events on the same thread):

$$M_e([x],n)$$
  $C_s([x],n',d)$   $C_s([x],d)$   $C_s([x],d)$   $C_s([x],d)$   $C_s([x],d)$ 

![](_page_6_Figure_0.jpeg)

Fig. 8: A pattern forbidden by traditional happens-before (hb).

Formally, this axiom is OEInt in Figure 6. We also add a similar axiom for the existence of an OnMiss between two eviction callbacks of the same phantom address (OMInt).

The OEInt axiom only ensures that an eviction callback runs in between two OnMisses. It does not enforce whether said callback is an OnEvict or OnWB for the appropriate cases. With OEInt, an execution graph like Figure 7d is possible. Here, we have an OnEvict for [x] in between the two OnMisses (the dirty bits of the E<sup>s</sup> and E<sup>e</sup> are false). Since [x] is written to by the Wcb node in the graph before its eviction, an OnWB for [x] should run instead of the OnEvict. The OnEvict should only have run if [x] had not been written to in the cache before its eviction.

To this end, we establish a correspondence between the dirty bits of E<sup>s</sup> and E<sup>e</sup> events and the existence of a write to a phantom address after it is brought into the cache. If the dirty bit of an E<sup>s</sup> is false (OnEvict case), we outlaw the existence of a callback write event in cbo order occurring between the previous M<sup>e</sup> (i.e., the end of the most recent OnMiss) and this Es. Conversely, if the dirty bit of the E<sup>s</sup> is true (OnWB case), we necessitate the existence of a write event in between the previous M<sup>e</sup> and the Es. Both cases are depicted below:

![](_page_6_Figure_5.jpeg)

To express this formally (Figure 6), we enforce EvDirty for the OnEvict and WbDirty for the OnWB cases.

Once we enforce the correct type for each eviction callback, Figure 7d can no longer be generated. Figure 7e shows the execution of our running example with an OnWB between the two OnMiss callbacks for [x], as required.

While we have now enforced correct values for the callbacks and phantom addresses, the effects of these callbacks and phantom addresses on *regular* addresses have not been enforced. Specifically, Figure 7e shows that the write of 1 to [y] in the OnWB of [x] runs before the Rcb of [x], which in turn runs before the read of [y]. (We assume no load-load reordering in our system; the tak¨ o paper [55] uses the x86 ¯ ISA which forbids load-load reordering [49].) Thus, the read of [y] should see the write of 1 to [y], but there is currently no axiom enforcing this. To enforce this ordering, we need to augment the happens-before reasoning from traditional MCMs with callback-related orderings, which we do next.

