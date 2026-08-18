# *C. Instruction Ordering Within Callbacks*

Our MCM does not require the preservation of program order between instructions to different addresses in callbacks, similar to how we do not require such ordering in nontak¨ o programs (§V-A). We demonstrate this using the ¯ icbsb litmus test in Figure 10a. In this test, two cores perform writes (i1) and (i2) to phantom addresses [x] and [y] respectively. As there are no restrictions in tak¨ o that prevent running callbacks ¯ for *different* addresses concurrently, the two OnWBs can run at any time with respect to each other, and perform writes and reads to addresses [a] and [b].

The OnWBs for [x] and [y] recreate the well-known sb (store buffering) litmus test [53]. As in the sb test, for both loads in the callbacks to return 0 (i.e., the outcome r1=0, r2=0), the instructions in at least one callback must be reordered. Figure 10b shows an execution with the outcome r1=0,r2=0 that is allowed under our model, thus showing that our model does not require the preservation of program order in callbacks. More specifically, the axioms in our MCM allow the fr∪sb cycle among the callback instructions in Figure 10b.

By not requiring program order to be preserved in callbacks, our MCM gives computer architects significant freedom when

| Core 0                                   | [x].OnMiss   | [x].OnWB      |  |
|------------------------------------------|--------------|---------------|--|
| (i1) [x] ← 1                             | (i3) [x] ← 0 | (i5) [a] ← 1  |  |
|                                          |              | (i6) r1 ← [b] |  |
| Core 1                                   | [y].OnMiss   | [y].OnWB      |  |
| (i2) [y] ← 1                             | (i4) [y] ← 0 | (i7) [b] ← 1  |  |
|                                          |              | (i8) r2 ← [a] |  |
| icbsb: r1 = 0, r2 = 0 allowed by our MCM |              |               |  |

![](_page_8_Figure_1.jpeg)

Fig. 10: (a) The icbsb litmus test. (b) An execution of icbsb that demonstrates the intra-callback instruction reordering that is allowed by our MCM.

designing the callback engine. In particular, tak¨ o designs that ¯ buffer and reorder memory operations in the engine can still use our MCM.

