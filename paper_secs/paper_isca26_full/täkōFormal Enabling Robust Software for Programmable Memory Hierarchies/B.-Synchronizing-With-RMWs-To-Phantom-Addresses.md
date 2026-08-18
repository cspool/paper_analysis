# *B. Synchronizing With RMWs To Phantom Addresses*

RMWs to phantom addresses (i.e., RMWcb events (§IV-B)) can also be used to enforce ordering constraints, similar to how regular address RMWs can be used on conventional programs (§V-A). This is illustrated in the mpcb litmus test (Figure 9a), where the address [b] is a phantom address with an OnMiss that returns 0. Similar to mprmw, the outcome of r1=1,r2=0 is forbidden by our tak¨ o MCM. ¯

Figure 9c demonstrates a forbidden execution graph that shows the happens-before reasoning in mpcb, which is analogous to that presented in §V-A for mprmw. In this case, as the cbo relation between the two RMWcb events is also added to the sw relation, the same hb edge is constructed between the write and read of [a]. The Vis axiom then forbids the outcome of r1=1,r2=0 for this test, very similar to how it forbids this outcome in mprmw.

Of course, with the use of a phantom address, one must reason about intervening evictions that could occur between the RMWcb accesses. In mpcb, the VisCb axiom ensures that if (i3) reads a value of 1, it must get this value from (i2) (as the OnMiss can only produce the value 0), ensuring that no eviction occurs in between.

