# *B. The hierarchical loop iteration identifier is a stack*

The hierarchical loop iteration identifier (or simply, the identifier) is a stack of header block and iteration count pairs. The identifier is manipulated on three special control-flow edges. For a loop L, these edges are:

- *enter loop*(L): an edge from a block outside L to its header block.
- *continue loop*: an edge from a block inside L to its header block.
- *exit loop*(pop amount): an edge from a block inside L to a block outside L. The pop amount is the difference between the loop depth of L and that of the edge's destination block. The destination block usually belongs

to the parent loop of L or is a top-level block if L itself is a top-level loop. In both cases, pop amount is one. In other cases, the pop amount is more than one.2

Once the required edges are determined for all the loops in a function, manipulating the identifier is straightforward:

- *enter loop*(L): push({L,1}); • *continue loop*: top().iteration\_count++;
- *exit loop*(pop amount): while (pop\_amount--) pop();

*1) An example:* Figure 6 shows stack manipulations on a dynamic sequence of edges for the bfs BUStep function. The outer loop (B) is iterated twice. The inner loop (D) is not visited in the first iteration of B, and is iterated twice in the second iteration of B. The exact sequence of edges is captured in the first row. The identifier is manipulated on the special edges. The identifier transitions from {} to {*B.*1} (*enter loop*(B)), {*B.*1} to {*B.*2} (*continue loop*), {*B.*2} to {*B.*2*, D.*1} (*enter loop*(D)), {*B.*2*, D.*1} to {*B.*2*, D.*2} (*continue loop*), {*B.*2*, D.*2} to {*B.*2} (*exit loop*(1)), and {*B.*2} to {} (*exit loop*(1)).

*2) Dealing with overlapping special edges:* In the bfs BUStep function, there is no case where the same edge represents a special edge for one loop and a *different* special edge for another loop. But in general, there may be such overlapping special edges. When there are overlapping special edges, there can be at most one *enter loop* or *continue loop* among them: these two types cannot overlap for a loop due to distinct source blocks (outside *vs.* inside the loop, respectively), and they cannot overlap across loops due to distinct destination blocks (*i.e.*, the header blocks). Thus, valid overlap scenarios are when the same edge represents one or more *exit loop* edges of various pop amounts, and optionally a single *enter loop* or *continue loop* edge.

If there is an overlap, the identifier may need to be manipulated more than once. If the overlap involves just *exit loop* edges, then the identifier is manipulated once, based on the highest pop amount. If the overlap involves an *enter loop* or a *continue loop* edge, the identifier is manipulated twice: first for the *exit loop* edge(s) and then for the *enter loop*/*continue loop* edge.3

## *C. Conveying the special control-flow edges to the hardware*

The compiler needs to convey the special control-flow edges in the program's binary, so that hardware can manage its stack.4

<sup>2</sup>The discussion assumes that the destination block either belongs to a loop that is an ancestor of L or is a top-level block. Otherwise, the pop amount is one more than the difference between the loop depth of L and that of the edge's destination block.

<sup>3</sup>Special edge overlaps can be reduced or eliminated using certain loopcentric compiler passes. The techniques explained here are general and do not require any such compiler pass.

<sup>4</sup>It may be possible to infer loops, hence, special control-flow edges, using a hardware-only scheme. We leave this for future work.

| Edge sequence             | Init | A->B          | B->C  | C->G  | G->B          | B->C  | C->D          | D->E      | E->D          | D->F         | F->G  | G->H         |
|---------------------------|------|---------------|-------|-------|---------------|-------|---------------|-----------|---------------|--------------|-------|--------------|
| Edge specialty            | -    | enter_loop(B) | -     | -     | continue_loop | -     | enter_loop(D) | -         | continue_loop | exit_loop(1) | -     | exit_loop(1) |
| Stack op.                 | -    | push({B,1})   | -     | -     | top().iter++  | -     | push({D,1})   | -         | top().iter++  | pop(1)       | -     | pop(1)       |
| Identifier<br>(after op.) | {}   | {B.1}         | {B.1} | {B.1} | {B.2}         | {B.2} | {B.2,D.1}     | {B.2,D.1} | {B.2,D.2}     | {B.2}        | {B.2} | {}           |

Fig. 6: Illustrating identifier manipulations using stack operations in bfs BUStep.

1) Representing special control-flow edges: The source of an exit\_loop edge is a block that ends in a conditional branch. So an exit\_loop edge is represented by its corresponding branch's PC, the exit direction, and the pop\_amount.

However, the source of an <code>enter\_loop</code> edge or a <code>continue\_loop</code> edge may not be a branch instruction (in the case where the edge is a fall-through edge). Still, the destination of all such edges is a header block, or more precisely the first PC in the header block, or simply header PC. Moreover, an <code>enter\_loop</code> edge can be differentiated from a <code>continue\_loop</code> edge reaching the same destination header PC by comparing the destination header PC with the header PC on the top of the stack: a mismatch signifies <code>enter\_loop</code> and a match signifies <code>continue\_loop</code>. So, all <code>enter\_loop</code> and <code>continue\_loop</code> edges for a loop are succinctly represented by the loop's header PC.

- 2) Loop descriptor: Each loop descriptor has the format {header PC, number of exits, exit1, exit2, ...}, where each exit is of the form {branch PC, exit direction, pop\_amount}.
- 3) Loop Information Segment (LIS): Loop descriptors are listed in this dedicated segment of the program binary.
- 4) Profiling compiler: A compiler pass generates loop descriptors for all loops in a given program. The assembler writes the loop descriptors to the program binary's LIS (Fig. 7:1).

The number of loop descriptors included in the LIS is at the discretion of the compiler or compiler user. Moreover, the LIS size is not confined by any one CPU design. A CPU will have a hardware table for holding loop descriptors (discussed in the next subsection), but its size and the LIS size are independent.

Knowing that a given CPU's table may be smaller than the LIS, it is advantageous for the compiler to use profiling to order loop descriptors in the LIS from most to least important. In this paper, a given loop's rank is based on the profiled number of occurrences of its header PC, which reflects both number of visits to the loop (enter loop) and number of iterations of those visits (continue loop).

Note that our profiling approach does not factor-in anything about hard-to-predict (H2P) branches before, within, or after loops. While H2P branches are a possible consideration, our rationale for considering only loop header frequency is that loops are the basis for invariant identifiers, independent of what the identifiers are used for.

5) Loop Information Table: On the CPU side, loop descriptors are held in the Loop Information Table (LIT). The LIT is sub-divided into two tables, one for header PCs (LIT-H) and the other for exit branches (LIT-E). Both LIT-H and LIT-E are searched by PC.

The ISA specifies the existence and formats of the LIT-H/LIT-E but leaves their capacities up to the CPU. The LIT-H capacity (number of loop header PCs) and LIT-E capacity (number of loop exits) are in an ISA-specified CPU configuration register readable by the operating system (O/S).

The CPU's LIT-H/LIT-E are loaded by the O/S when it schedules a process on the CPU, *i.e.*, when it launches a process or resumes a suspended process. The O/S maintains, for each process, a Process Control Block (PCB) containing a memory copy of all architectural register state [6]; the PCB is extended with memory copies of the LIT-H/LIT-E. The PCB's LIT-H/LIT-E are initialized by the O/S loader from the topmost loops in the program binary's LIS (up to their capacities) (Fig. 7:2). The O/S scheduler loads the CPU's LIT-H/LIT-E from the PCB's LIT-H/LIT-E using a series of load and move instruction pairs (load to general register, move from general register to indexed array of special registers) (Fig. 7:3).

![](_page_6_Figure_13.jpeg)

Fig. 7: From LIS to CPU's LIT-H/LIT-E.

A CPU with SMT support equally divides its LIT-H/E into as many partitions as there are active threads. This reduces the number of loops that may participate in invariant identifiers, for each thread. Threads from the same process or processes launched from the same binary may share a partition. If the O/S suspends a thread and doesn't schedule another one in its place, remaining threads' partitions are upsized and the O/S may load additional loops for them.

#### D. Realizing the identifier

We discuss two ways of implementing the identifier. The first is a literal stack of {header PC, iteration count} pairs (Subsection III-D1). A practical concern with this approach is that, for each access to the SBRB, a signature must be generated by hashing all entries of the stack. This motivates the second approach (Subsection III-D2) in which an ongoing signature is maintained. There is still a stack, but its purpose is to save the signature before entering a loop and restore it after exiting a loop. We call this approach the Signature Stack (SS). In this paper, we use the SS.

1) A literal stack of {header PC, iteration count} pairs: A hit in LIT-H is relayed as hit\_header(PC). A hit in LIT-E is relayed as hit\_exit(pop\_amount) only if the prediction for that exit branch matches with the exit direction. Based on

the hit information from the LIT, the stack is manipulated as follows:

• *hit\_header*(PC):

```
if (PC == top().header_PC)
  top().iteration_count++; // cont. loop\nelse
  push({PC, 1}); // enter loop
```

hit\_exit(pop\_amount):while (pop\_amount--) pop(); // exit loop

We define a **signature** as a fixed-sized (*e.g.*, 32 bits) compressed representation of the hierarchical loop iteration identifier. To compute a signature using the literal stack described here, *all* the bits in the stack have to be suitably hashed together. The literal stack is not a pragmatic design because its hashing may be slow and may need to be performed frequently. Also, while folding and hashing of PCs with XOR preserves adequate information, it is less clear how to effectively hash iteration counts.

- 2) Signature Stack (SS): In this design, an ongoing signature is maintained, that by itself represents (in compressed form) the current hierarchical loop iteration identifier. The signature is stored in a Linear Feedback Shift Register (LFSR). The signature is paired with the header PC of the current loop, which is needed to differentiate between continuing the loop vs. entering a different loop. Thus, we have a pair of registers:
  - hpc: The header PC of the current loop. If a loop header PC is observed, it can be compared against hpc to determine whether the current loop is being continued (observed header PC matches hpc) or a new loop is being entered (observed header PC differs from hpc). To reduce cost<sup>5</sup>, hpc is an 8-bit compressed representation of the header PC: fold8 (header PC) (successively split the bits into two halves, fold the top half over the bottom half, and bitwise-XOR them, until a width of 8 bits is reached). The design space exploration in Section V-B2 shows that fold8 (header PC) is as effective as using the full header PC.
  - sig: A LFSR-based representation of the current hierarchical loop iteration identifier. We use a Galois LFSR [1], [44]. In Section V-B2, we perform a design space exploration of the number of bits for sig and find that the 32-bit Galois LFSR shown in Figure 8a generates 32-bit signatures that rarely or never collide. The LFSR is augmented with the ability to bitwise-XOR sig with an arbitrary data input (of the same bit-width) as it is shifted, as shown in Figure 8b. An incremental update of sig can be written as: sig = LFSR\_step(sig ^ data).

The overall "Signature Stack (SS)" component refers to the two registers, hpc and sig, and a stack for saving/restoring these two registers as explained below. Here is how the SS is managed:

- Entering a loop: A loop is entered when a loop header PC is observed that differs from hpc. First, the values of  $\{hpc, sig\}$  are saved by pushing them to the save/restore stack. This allows updating sig for the newly-entered loop, while still being able to revert back to the sig that existed before this loop, when it is eventually exited. Second, sig is updated as follows:  $sig = LFSR\_step(sig ^ fold32(PC))$ . This reflects adding the loop's header PC (PC, folded to the bit-width of sig) and initial iteration count (1, by virtue of the LFSR rotating by 1 bit) to the hierarchical loop iteration identifier. hpc is set to the entered-loop's header PC (folded according to the bit-width of hpc).
- Continuing a loop: A loop is continued when a loop header PC is observed that matches hpc. In this case, sig is simply updated to reflect another iteration:  $sig = LFSR\_step(sig ^ 0)$ .
- Exiting a loop: The save/restore stack is popped the requisite number of times (pop amount) and {hpc, sig} are restored from the final popped stack entry.

![](_page_7_Figure_13.jpeg)

Fig. 8: (a) The 32-bit LFSR used in this paper. (b) The LFSR with data input, d, where d is either the 32-bit folded header PC for entering a loop or 0 for continuing a loop.

In addition to LIT hits, the SS is also manipulated on call and return instructions (the existing BTB signals calls and returns). This is necessary to ensure unique and invariant signatures for branches across different potential invocations of the same function, from the same loop and iteration: (1) loop header PCs and branch PCs are not unique across function instances, and (2) the signatures at potential invocations are otherwise identical (not changing) if they are in the same loop and iteration. A call causes a save and update as was described above for entering a loop, where the update uses the call instruction's PC (PC of the call site). A return causes a restore operation. Differentiation and invariance is achieved whether there are two different call sites (save sig, augment sig with first caller's PC, restore sig; repeat with second caller's PC; signatures are invariant whether or not either call is actually fetched, as determined by surrounding control-flow) or a single call site with recursion (save sig, augment sig with caller's PC, repeat).

A comprehensive description of SS management is given in Table I. The description uses 8 bits and 32 bits for hpc and sig, respectively.

<sup>&</sup>lt;sup>5</sup>While *hpc* is only one register, it is saved in entries of the save/restore stack. Further, the overall SS component is checkpointed at branches, as explained in Section III-E.

TABLE I: Signature Stack (SS) management.

| hits from LIT and BTB | SS management                         |  |  |  |
|-----------------------|---------------------------------------|--|--|--|
| hit header(PC):       | if (fold8(PC) != hpc) { // enter loop |  |  |  |
|                       | push({hpc, sig});<br>// save          |  |  |  |
|                       | hpc = fold8(PC);                      |  |  |  |
|                       | sig = LFSR_step(sig ˆ fold32(PC));    |  |  |  |
|                       | }                                     |  |  |  |
|                       | else {<br>// continue loop            |  |  |  |
|                       | sig = LFSR_step(sig ˆ 0);             |  |  |  |
|                       | }                                     |  |  |  |
| hit exit(pop amount): | // exit loop                          |  |  |  |
|                       | while (pop_amount)                    |  |  |  |
|                       | pop({hpc, sig});<br>// restore        |  |  |  |
| hit call(PC):         | push({hpc, sig});<br>// save          |  |  |  |
|                       | hpc = 0;                              |  |  |  |
|                       | sig = LFSR_step(sig ˆ fold32(PC));    |  |  |  |
| hit return:           | pop({hpc, sig});<br>// restore        |  |  |  |

## *E. Recovering the Signature Stack*

Our baseline superscalar processor has 64 branch checkpoints to quickly recover various structures or structure pointers when a mispredicted branch resolves (*i.e.*, rename map table, free list head pointer, SQ/LQ tail pointers, *etc.*) [46]. Each branch checkpoint is augmented with storage to also checkpoint the SS. This allows recovering the SS precisely.

To support SS recovery in the case of load violations (memory dependency mispredictions) and exceptions, which initiate recovery when they reach the ROB head, a retirement SS is maintained. Our baseline superscalar processor includes a Branch Queue (BQ) in the fetch unit to train branch prediction tables at retirement. Each BQ entry is augmented with 6 bits (see Table II) to redundantly manage the retirement SS. The retirement SS enables recovery of the (fetch) SS to the precise point just prior to the offending instruction.

## *F. Squashed Branch Reuse Buffer (SBRB)*

A branch accesses the SBRB using a hash of its PC and its signature. The PC is folded once (compress from 64 to 32 bits) and XOR-ed with the signature, to form a 32-bit key into the SBRB: key = fold32(PC) ˆ sig.

The SBRB can be organized as direct-mapped, setassociative, or fully-associative. The 32-bit key, above, is divided into index and tag bits, accordingly (like any other cache structure). Each SBRB entry has a valid bit, tag, replacement counter (if set- or fully-associative), and branch outcome.

Here are the operations on the SBRB:

- *When a branch executes (execute stage)*: The executed branch searches the SBRB using its key.6 If it misses, an entry is allocated and the branch's outcome is deposited. If it hits, the outcome is updated. A hit can occur when the same dynamic branch is squashed multiple times (or when there are key collisions).
- *When a branch is predicted (fetch stage)*: The SBRB is searched using the dynamic branch's key. If it hits and the branch's confidence counter (from the BTB)

6Each BQ entry can be augmented with the corresponding branch's key, that can later be used by a resolved branch (which carries its BQ index in its payload) to access the SBRB. Alternatively, the resolved branch can regenerate its key using its BQ entry's PC and its sig from the SS checkpoint within its branch checkpoint (Sec. III-E). We use the latter approach.

indicates high confidence, the outcome from the SBRB is used as the final prediction, overriding the default branch predictor.

## *G. Training confidence counters*

Each entry in the BTB is augmented with an n-bit confidence counter. When a branch is installed in the BTB, its confidence counter is initialized to 0. The SBRB may only override the default branch predictor if *counter >* 2(n−1) − 1 (*i.e.*, deemed confident when the counter is in the upperhalf of its reachable states). The design space exploration in Section V-C results in defaults of: 3-bit confidence counter, states 0–7, confidence threshold of 3.

A branch's counter learns whether a squashed outcome supplied by the SBRB (when available) tends to be more accurate than the prediction supplied by the default branch predictor. A squashed outcome obtained from the SBRB during the fetch stage is recorded in the branch's BQ entry. So is the prediction from the default branch predictor and, eventually, the actual outcome when the branch executes. When the branch retires (hence, it is at the BQ head), its confidence counter is potentially updated based on four pieces of information: whether a squashed outcome is available, the squashed outcome (if available), the default prediction, and the actual outcome. The counter is only updated if a squashed outcome is available and it differs from the default prediction, as follows:

```
if (squashed_outcome_available &&
    (squashed_outcome != default_prediction))
{
  if (squashed_outcome == actual_outcome) {
    if (counter < max_value) counter++;
  } else {
    if (counter > 0) counter--;
  }
}
```

# *B. The hierarchical loop iteration identifier is a stack*

The hierarchical loop iteration identifier (or simply, the identifier) is a stack of header block and iteration count pairs. The identifier is manipulated on three special control-flow edges. For a loop L, these edges are:

- *enter loop*(L): an edge from a block outside L to its header block.
- *continue loop*: an edge from a block inside L to its header block.
- *exit loop*(pop amount): an edge from a block inside L to a block outside L. The pop amount is the difference between the loop depth of L and that of the edge's destination block. The destination block usually belongs

to the parent loop of L or is a top-level block if L itself is a top-level loop. In both cases, pop amount is one. In other cases, the pop amount is more than one.2

Once the required edges are determined for all the loops in a function, manipulating the identifier is straightforward:

- *enter loop*(L): push({L,1}); • *continue loop*: top().iteration\_count++;
- *exit loop*(pop amount): while (pop\_amount--) pop();

*1) An example:* Figure 6 shows stack manipulations on a dynamic sequence of edges for the bfs BUStep function. The outer loop (B) is iterated twice. The inner loop (D) is not visited in the first iteration of B, and is iterated twice in the second iteration of B. The exact sequence of edges is captured in the first row. The identifier is manipulated on the special edges. The identifier transitions from {} to {*B.*1} (*enter loop*(B)), {*B.*1} to {*B.*2} (*continue loop*), {*B.*2} to {*B.*2*, D.*1} (*enter loop*(D)), {*B.*2*, D.*1} to {*B.*2*, D.*2} (*continue loop*), {*B.*2*, D.*2} to {*B.*2} (*exit loop*(1)), and {*B.*2} to {} (*exit loop*(1)).

*2) Dealing with overlapping special edges:* In the bfs BUStep function, there is no case where the same edge represents a special edge for one loop and a *different* special edge for another loop. But in general, there may be such overlapping special edges. When there are overlapping special edges, there can be at most one *enter loop* or *continue loop* among them: these two types cannot overlap for a loop due to distinct source blocks (outside *vs.* inside the loop, respectively), and they cannot overlap across loops due to distinct destination blocks (*i.e.*, the header blocks). Thus, valid overlap scenarios are when the same edge represents one or more *exit loop* edges of various pop amounts, and optionally a single *enter loop* or *continue loop* edge.

If there is an overlap, the identifier may need to be manipulated more than once. If the overlap involves just *exit loop* edges, then the identifier is manipulated once, based on the highest pop amount. If the overlap involves an *enter loop* or a *continue loop* edge, the identifier is manipulated twice: first for the *exit loop* edge(s) and then for the *enter loop*/*continue loop* edge.3

## *C. Conveying the special control-flow edges to the hardware*

The compiler needs to convey the special control-flow edges in the program's binary, so that hardware can manage its stack.4

<sup>2</sup>The discussion assumes that the destination block either belongs to a loop that is an ancestor of L or is a top-level block. Otherwise, the pop amount is one more than the difference between the loop depth of L and that of the edge's destination block.

<sup>3</sup>Special edge overlaps can be reduced or eliminated using certain loopcentric compiler passes. The techniques explained here are general and do not require any such compiler pass.

<sup>4</sup>It may be possible to infer loops, hence, special control-flow edges, using a hardware-only scheme. We leave this for future work.

| Edge sequence             | Init | A->B          | B->C  | C->G  | G->B          | B->C  | C->D          | D->E      | E->D          | D->F         | F->G  | G->H         |
|---------------------------|------|---------------|-------|-------|---------------|-------|---------------|-----------|---------------|--------------|-------|--------------|
| Edge specialty            | -    | enter_loop(B) | -     | -     | continue_loop | -     | enter_loop(D) | -         | continue_loop | exit_loop(1) | -     | exit_loop(1) |
| Stack op.                 | -    | push({B,1})   | -     | -     | top().iter++  | -     | push({D,1})   | -         | top().iter++  | pop(1)       | -     | pop(1)       |
| Identifier<br>(after op.) | {}   | {B.1}         | {B.1} | {B.1} | {B.2}         | {B.2} | {B.2,D.1}     | {B.2,D.1} | {B.2,D.2}     | {B.2}        | {B.2} | {}           |

Fig. 6: Illustrating identifier manipulations using stack operations in bfs BUStep.

1) Representing special control-flow edges: The source of an exit\_loop edge is a block that ends in a conditional branch. So an exit\_loop edge is represented by its corresponding branch's PC, the exit direction, and the pop\_amount.

However, the source of an <code>enter\_loop</code> edge or a <code>continue\_loop</code> edge may not be a branch instruction (in the case where the edge is a fall-through edge). Still, the destination of all such edges is a header block, or more precisely the first PC in the header block, or simply header PC. Moreover, an <code>enter\_loop</code> edge can be differentiated from a <code>continue\_loop</code> edge reaching the same destination header PC by comparing the destination header PC with the header PC on the top of the stack: a mismatch signifies <code>enter\_loop</code> and a match signifies <code>continue\_loop</code>. So, all <code>enter\_loop</code> and <code>continue\_loop</code> edges for a loop are succinctly represented by the loop's header PC.

- 2) Loop descriptor: Each loop descriptor has the format {header PC, number of exits, exit1, exit2, ...}, where each exit is of the form {branch PC, exit direction, pop\_amount}.
- 3) Loop Information Segment (LIS): Loop descriptors are listed in this dedicated segment of the program binary.
- 4) Profiling compiler: A compiler pass generates loop descriptors for all loops in a given program. The assembler writes the loop descriptors to the program binary's LIS (Fig. 7:1).

The number of loop descriptors included in the LIS is at the discretion of the compiler or compiler user. Moreover, the LIS size is not confined by any one CPU design. A CPU will have a hardware table for holding loop descriptors (discussed in the next subsection), but its size and the LIS size are independent.

Knowing that a given CPU's table may be smaller than the LIS, it is advantageous for the compiler to use profiling to order loop descriptors in the LIS from most to least important. In this paper, a given loop's rank is based on the profiled number of occurrences of its header PC, which reflects both number of visits to the loop (enter loop) and number of iterations of those visits (continue loop).

Note that our profiling approach does not factor-in anything about hard-to-predict (H2P) branches before, within, or after loops. While H2P branches are a possible consideration, our rationale for considering only loop header frequency is that loops are the basis for invariant identifiers, independent of what the identifiers are used for.

5) Loop Information Table: On the CPU side, loop descriptors are held in the Loop Information Table (LIT). The LIT is sub-divided into two tables, one for header PCs (LIT-H) and the other for exit branches (LIT-E). Both LIT-H and LIT-E are searched by PC.

The ISA specifies the existence and formats of the LIT-H/LIT-E but leaves their capacities up to the CPU. The LIT-H capacity (number of loop header PCs) and LIT-E capacity (number of loop exits) are in an ISA-specified CPU configuration register readable by the operating system (O/S).

The CPU's LIT-H/LIT-E are loaded by the O/S when it schedules a process on the CPU, *i.e.*, when it launches a process or resumes a suspended process. The O/S maintains, for each process, a Process Control Block (PCB) containing a memory copy of all architectural register state [6]; the PCB is extended with memory copies of the LIT-H/LIT-E. The PCB's LIT-H/LIT-E are initialized by the O/S loader from the topmost loops in the program binary's LIS (up to their capacities) (Fig. 7:2). The O/S scheduler loads the CPU's LIT-H/LIT-E from the PCB's LIT-H/LIT-E using a series of load and move instruction pairs (load to general register, move from general register to indexed array of special registers) (Fig. 7:3).

![](_page_6_Figure_13.jpeg)

Fig. 7: From LIS to CPU's LIT-H/LIT-E.

A CPU with SMT support equally divides its LIT-H/E into as many partitions as there are active threads. This reduces the number of loops that may participate in invariant identifiers, for each thread. Threads from the same process or processes launched from the same binary may share a partition. If the O/S suspends a thread and doesn't schedule another one in its place, remaining threads' partitions are upsized and the O/S may load additional loops for them.

#### D. Realizing the identifier

We discuss two ways of implementing the identifier. The first is a literal stack of {header PC, iteration count} pairs (Subsection III-D1). A practical concern with this approach is that, for each access to the SBRB, a signature must be generated by hashing all entries of the stack. This motivates the second approach (Subsection III-D2) in which an ongoing signature is maintained. There is still a stack, but its purpose is to save the signature before entering a loop and restore it after exiting a loop. We call this approach the Signature Stack (SS). In this paper, we use the SS.

1) A literal stack of {header PC, iteration count} pairs: A hit in LIT-H is relayed as hit\_header(PC). A hit in LIT-E is relayed as hit\_exit(pop\_amount) only if the prediction for that exit branch matches with the exit direction. Based on

the hit information from the LIT, the stack is manipulated as follows:

• *hit\_header*(PC):

```
if (PC == top().header_PC)
  top().iteration_count++; // cont. loop\nelse
  push({PC, 1}); // enter loop
```

hit\_exit(pop\_amount):while (pop\_amount--) pop(); // exit loop

We define a **signature** as a fixed-sized (*e.g.*, 32 bits) compressed representation of the hierarchical loop iteration identifier. To compute a signature using the literal stack described here, *all* the bits in the stack have to be suitably hashed together. The literal stack is not a pragmatic design because its hashing may be slow and may need to be performed frequently. Also, while folding and hashing of PCs with XOR preserves adequate information, it is less clear how to effectively hash iteration counts.

- 2) Signature Stack (SS): In this design, an ongoing signature is maintained, that by itself represents (in compressed form) the current hierarchical loop iteration identifier. The signature is stored in a Linear Feedback Shift Register (LFSR). The signature is paired with the header PC of the current loop, which is needed to differentiate between continuing the loop vs. entering a different loop. Thus, we have a pair of registers:
  - hpc: The header PC of the current loop. If a loop header PC is observed, it can be compared against hpc to determine whether the current loop is being continued (observed header PC matches hpc) or a new loop is being entered (observed header PC differs from hpc). To reduce cost<sup>5</sup>, hpc is an 8-bit compressed representation of the header PC: fold8 (header PC) (successively split the bits into two halves, fold the top half over the bottom half, and bitwise-XOR them, until a width of 8 bits is reached). The design space exploration in Section V-B2 shows that fold8 (header PC) is as effective as using the full header PC.
  - sig: A LFSR-based representation of the current hierarchical loop iteration identifier. We use a Galois LFSR [1], [44]. In Section V-B2, we perform a design space exploration of the number of bits for sig and find that the 32-bit Galois LFSR shown in Figure 8a generates 32-bit signatures that rarely or never collide. The LFSR is augmented with the ability to bitwise-XOR sig with an arbitrary data input (of the same bit-width) as it is shifted, as shown in Figure 8b. An incremental update of sig can be written as: sig = LFSR\_step(sig ^ data).

The overall "Signature Stack (SS)" component refers to the two registers, hpc and sig, and a stack for saving/restoring these two registers as explained below. Here is how the SS is managed:

- Entering a loop: A loop is entered when a loop header PC is observed that differs from hpc. First, the values of  $\{hpc, sig\}$  are saved by pushing them to the save/restore stack. This allows updating sig for the newly-entered loop, while still being able to revert back to the sig that existed before this loop, when it is eventually exited. Second, sig is updated as follows:  $sig = LFSR\_step(sig ^ fold32(PC))$ . This reflects adding the loop's header PC (PC, folded to the bit-width of sig) and initial iteration count (1, by virtue of the LFSR rotating by 1 bit) to the hierarchical loop iteration identifier. hpc is set to the entered-loop's header PC (folded according to the bit-width of hpc).
- Continuing a loop: A loop is continued when a loop header PC is observed that matches hpc. In this case, sig is simply updated to reflect another iteration:  $sig = LFSR\_step(sig ^ 0)$ .
- Exiting a loop: The save/restore stack is popped the requisite number of times (pop amount) and {hpc, sig} are restored from the final popped stack entry.

![](_page_7_Figure_13.jpeg)

Fig. 8: (a) The 32-bit LFSR used in this paper. (b) The LFSR with data input, d, where d is either the 32-bit folded header PC for entering a loop or 0 for continuing a loop.

In addition to LIT hits, the SS is also manipulated on call and return instructions (the existing BTB signals calls and returns). This is necessary to ensure unique and invariant signatures for branches across different potential invocations of the same function, from the same loop and iteration: (1) loop header PCs and branch PCs are not unique across function instances, and (2) the signatures at potential invocations are otherwise identical (not changing) if they are in the same loop and iteration. A call causes a save and update as was described above for entering a loop, where the update uses the call instruction's PC (PC of the call site). A return causes a restore operation. Differentiation and invariance is achieved whether there are two different call sites (save sig, augment sig with first caller's PC, restore sig; repeat with second caller's PC; signatures are invariant whether or not either call is actually fetched, as determined by surrounding control-flow) or a single call site with recursion (save sig, augment sig with caller's PC, repeat).

A comprehensive description of SS management is given in Table I. The description uses 8 bits and 32 bits for hpc and sig, respectively.

<sup>&</sup>lt;sup>5</sup>While *hpc* is only one register, it is saved in entries of the save/restore stack. Further, the overall SS component is checkpointed at branches, as explained in Section III-E.

TABLE I: Signature Stack (SS) management.

| hits from LIT and BTB | SS management                         |  |  |  |
|-----------------------|---------------------------------------|--|--|--|
| hit header(PC):       | if (fold8(PC) != hpc) { // enter loop |  |  |  |
|                       | push({hpc, sig});<br>// save          |  |  |  |
|                       | hpc = fold8(PC);                      |  |  |  |
|                       | sig = LFSR_step(sig ˆ fold32(PC));    |  |  |  |
|                       | }                                     |  |  |  |
|                       | else {<br>// continue loop            |  |  |  |
|                       | sig = LFSR_step(sig ˆ 0);             |  |  |  |
|                       | }                                     |  |  |  |
| hit exit(pop amount): | // exit loop                          |  |  |  |
|                       | while (pop_amount)                    |  |  |  |
|                       | pop({hpc, sig});<br>// restore        |  |  |  |
| hit call(PC):         | push({hpc, sig});<br>// save          |  |  |  |
|                       | hpc = 0;                              |  |  |  |
|                       | sig = LFSR_step(sig ˆ fold32(PC));    |  |  |  |
| hit return:           | pop({hpc, sig});<br>// restore        |  |  |  |

## *E. Recovering the Signature Stack*

Our baseline superscalar processor has 64 branch checkpoints to quickly recover various structures or structure pointers when a mispredicted branch resolves (*i.e.*, rename map table, free list head pointer, SQ/LQ tail pointers, *etc.*) [46]. Each branch checkpoint is augmented with storage to also checkpoint the SS. This allows recovering the SS precisely.

To support SS recovery in the case of load violations (memory dependency mispredictions) and exceptions, which initiate recovery when they reach the ROB head, a retirement SS is maintained. Our baseline superscalar processor includes a Branch Queue (BQ) in the fetch unit to train branch prediction tables at retirement. Each BQ entry is augmented with 6 bits (see Table II) to redundantly manage the retirement SS. The retirement SS enables recovery of the (fetch) SS to the precise point just prior to the offending instruction.

## *F. Squashed Branch Reuse Buffer (SBRB)*

A branch accesses the SBRB using a hash of its PC and its signature. The PC is folded once (compress from 64 to 32 bits) and XOR-ed with the signature, to form a 32-bit key into the SBRB: key = fold32(PC) ˆ sig.

The SBRB can be organized as direct-mapped, setassociative, or fully-associative. The 32-bit key, above, is divided into index and tag bits, accordingly (like any other cache structure). Each SBRB entry has a valid bit, tag, replacement counter (if set- or fully-associative), and branch outcome.

Here are the operations on the SBRB:

- *When a branch executes (execute stage)*: The executed branch searches the SBRB using its key.6 If it misses, an entry is allocated and the branch's outcome is deposited. If it hits, the outcome is updated. A hit can occur when the same dynamic branch is squashed multiple times (or when there are key collisions).
- *When a branch is predicted (fetch stage)*: The SBRB is searched using the dynamic branch's key. If it hits and the branch's confidence counter (from the BTB)

6Each BQ entry can be augmented with the corresponding branch's key, that can later be used by a resolved branch (which carries its BQ index in its payload) to access the SBRB. Alternatively, the resolved branch can regenerate its key using its BQ entry's PC and its sig from the SS checkpoint within its branch checkpoint (Sec. III-E). We use the latter approach.

indicates high confidence, the outcome from the SBRB is used as the final prediction, overriding the default branch predictor.

## *G. Training confidence counters*

Each entry in the BTB is augmented with an n-bit confidence counter. When a branch is installed in the BTB, its confidence counter is initialized to 0. The SBRB may only override the default branch predictor if *counter >* 2(n−1) − 1 (*i.e.*, deemed confident when the counter is in the upperhalf of its reachable states). The design space exploration in Section V-C results in defaults of: 3-bit confidence counter, states 0–7, confidence threshold of 3.

A branch's counter learns whether a squashed outcome supplied by the SBRB (when available) tends to be more accurate than the prediction supplied by the default branch predictor. A squashed outcome obtained from the SBRB during the fetch stage is recorded in the branch's BQ entry. So is the prediction from the default branch predictor and, eventually, the actual outcome when the branch executes. When the branch retires (hence, it is at the BQ head), its confidence counter is potentially updated based on four pieces of information: whether a squashed outcome is available, the squashed outcome (if available), the default prediction, and the actual outcome. The counter is only updated if a squashed outcome is available and it differs from the default prediction, as follows:

```
if (squashed_outcome_available &&
    (squashed_outcome != default_prediction))
{
  if (squashed_outcome == actual_outcome) {
    if (counter < max_value) counter++;
  } else {
    if (counter > 0) counter--;
  }
}
```

