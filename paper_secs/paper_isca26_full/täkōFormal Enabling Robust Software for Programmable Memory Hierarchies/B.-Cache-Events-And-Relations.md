# *B. Cache Events And Relations*

Our ISA-level MCM for tak¨ o introduces new events and ¯ relations to enable reasoning about caches and callbacks. Since the semantics of phantom reads and writes require reasoning about callbacks (e.g., Figure 2), we denote phantom reads and writes using Rcb and Wcb events (cb for callback). Regular reads and writes are denoted using R and W. For both address types, we denote an atomic read-modify-write operation with RMWcb and RMW respectively. We add Fl events to represent the flushing of an address by FlushRange.

We add events for the beginning and end of each callback, denoted M<sup>s</sup> and M<sup>e</sup> (OnMiss start and end respectively) and E<sup>s</sup> and E<sup>e</sup> (OnEvict or OnWB start and end respectively). We differentiate between OnEvict and OnWB events using a *dirty bit* for each E<sup>s</sup> or E<sup>e</sup> event. The dirty bit is false for OnEvict events and true for OnWB events.

We add a new relation called cbo (callback order) to enforce orderings on these new events. cbo establishes a total order on all callback events (Rcb, Wcb, RMWcb, Ms, Me, Es, Ee) for a single address, and reflects tak¨ o's serialization of all ¯ callbacks to a given address [55].

Next, we describe how the axioms we develop on these relations forbid the r1=2, r2=0 outcome for Figure 4a's program.

1 In the literature (e.g., [5]), the relation denoting program order is sometimes labeled po and the relation denoting modification order is sometimes labeled co (coherence order).

```
\forall \mathbf{R}.\exists !\mathbf{W}.(\mathbf{W},\mathbf{R}) \in rf
                                                           empty([M_s]; cbo; cbo; [M_e] \cap thd)
                                               RfWf1
                                                                                                                                                   CboM
 rf \subseteq val \cap addr
                                               RfWf2
                                                           empty([E_s]; cbo; cbo; [E_e] \cap thd)
                                                                                                                                                    CboE
\forall A.to(mo, \mathbf{W}^A)
                                              MoWf1
                                                           empty([\mathbf{W_{cb}}]; viscb; [E_s(..., ..., false)])
                                                                                                                                                 EvDirty
 mo \subseteq addr
                                              MoWf2
                                                           empty(viscb; [E_s(..., true)] \setminus [\mathbf{W_{cb}}]; viscb)
                                                                                                                                               WbDirty
 \forall A.to(cbo, CB_{se}^A \cup CB_{me}^A)
                                             CboWf1
                                                           empty([M_e]; cbo; [M_s] \setminus [M_e]; cbo; [E_s]; thd; [E_e]; cbo; [M_s])
                                                                                                                                                   OEInt
 cbo \subseteq addr
                                             CboWf2
                                                           empty([E_e]; cbo; [E_s] \setminus [E_e]; cbo; [M_s]; thd; [M_e]; cbo; [E_s])
                                                                                                                                                  OMInt
 viscb \subseteq val
                                              CboVal
                                                           \forall M_e.\exists!M_s.(M_s,M_e)\in thd
                                                                                                                                                 OMThd
                                                           \forall E_e.\exists !E_s.(E_s,E_e) \in thd
 [M_s]; thd; [M_e] \subseteq cbo
                                                ThdM
                                                                                                                                                 OEThd
                                                           empty([M_s]; cbo; [M_s] \setminus [M_s]; thd; [M_e]; cbo; [M_s])
 [E_s]; thd; [E_e] \subset cbo
                                                                                                                                                   MeInt
                                                 ThdE
 [E_s]; thd; [E_e] \subseteq dirty
                                             DirtyWf
                                                           empty([E_s]; cbo; [E_s] \setminus [E_s]; thd; [E_e]; cbo; [E_s])
                                                                                                                                                    EeInt
                                                           \forall CB_{me}.\exists !M_e.(M_e,CB_{me}) \in vf
 irreflexive(hb)
                                                    Hb
                                                                                                                                                    VfWf
 irreflexive(eco; hb)
                                                           \forall E_s.\exists !M_e.(M_e,E_s) \in ef
                                                    Vis
                                                                                                                                                    EfWf
 irreflexive(rf \cup
                                                           \forall Fl.(\forall M_s.(Fl, M_s) \in addr \Rightarrow (Fl, M_s) \in cbo)) \lor
                                                                                                                                                    EbWf
    (mo; mo; rf^{-1}) \cup (mo; rf)
                                                               (\exists! E_e.(E_e, Fl) \in eb)
 irreflexive(cbo; hb)
                                                VisCb
where
                                             \mathbf{W} = \{W, RMW\}
                                                                                  vf = ([M_e]; cbo; [CB_{me}]) \setminus ([M_e]; cbo; [CB_{se}]; cbo; [CB_{me}])
\mathbf{R} = \{R, RMW\}
                                            \mathbf{W_{cb}} = \{W_{cb}, RMW_{cb}\} \qquad ef = ([M_e]; cbo; [E_s]) \setminus ([M_e]; cbo; [CB_{se}]; cbo; [E_s])
\mathbf{R_{cb}} = \{R_{cb}, RMW_{cb}\}
CB_{me} = \{R_{cb}, W_{cb}, RMW_{cb}\} \quad CB_{se} = \{M_s, M_e, E_s, E_e\} \quad eb = ([E_e]; cbo; [Fl]) \setminus ([E_e]; cbo; [CB_{se}]; cbo; [Fl])
 fr = rf^{-1}; mo
                                            eco = (rf \cup mo \cup fr)^+ sw = ([RMW]; rf; [RMW]) \cup ([RMW_{cb}]; cbo; [RMW_{cb}])
 hb = ((I \times \neg I) \cup sb \cup sw \cup vf \cup eb \cup ([M_e]; cbo; [E_s]) \cup ([E_e]; cbo; [M_s]))^+
 viscb = ([\mathbf{W_{cb}} \cup M_e]; cbo; [\mathbf{R_{cb}} \cup E_s]) \setminus ([\mathbf{W_{cb}} \cup M_e]; cbo; [CB_{se} \cup \mathbf{W_{cb}}]; cbo; [\mathbf{R_{cb}} \cup E_s])
 to(R, S) = irreflexive(R) \land transitive(R) \land (\forall s_1, s_2 \in S.R(s_1, s_2) \lor R(s_2, s_1))
 race = ((((W \cup R \cup W_{cb} \cup R_{cb}) \times (W \cup R \cup W_{cb} \cup R_{cb})) \setminus ((R \times R) \cup (R_{cb} \times R_{cb}))) \cap addr) \setminus (id \cup hb \cup hb^{-1})
```

 $X^A$  is all X events with address A. I denotes initialization events. id is the identity relation, i.e. pairs of identical events. Fig. 6: All axioms for our täkō MCM. Executions must satisfy all axioms to be allowed. A is the elements of type A in relation form [59].  $A \times B$  are pairs of an element of type A and an element of type  $A \setminus B$  is the elements of A that are not in B. Semicolons (;) denote relational composition, e.g., e1; e2 is two relations e1 and e2 where the destination node of

e1 is the source node of e2.  $R^{-1}$  is the inverse of R.  $\exists$ ! specifies existence of a unique element with the specified property.

Each address is of type Synch or Data.  $RMW/RMW_{cb}$  run on Synch addresses;  $R/W/R_{cb}/W_{cb}$  run on Data.

addr, val, dirty denote pairs of events with matching addresses, values, and dirty bits respectively.

Figure 6 contains all our axioms and their names. We refer to axioms using these names throughout the rest of the paper.

### C. Ensuring Phantom Address Sources

Figure 7 depicts execution graphs for the r1=2, r2=0 outcome of Figure 4a's program. This outcome is impossible on täkō due to cache and callback semantics (§III-B). Figure 7a depicts an execution graph for this outcome without täkō-specific axioms. Traditional MCM axioms cannot reason about cache events and callbacks, so täkō-specific axioms are needed to forbid this execution. Graphs (b)-(e) depict execution graphs after the addition of one or more täkō-specific axioms and/or relations that outlaw the previous flawed execution graph but not the forbidden outcome of r1=2, r2=0. Graph (f) depicts the execution graph after enough axioms and relations have been added to forbid the r1=2, r2=0 outcome.

First, consider Figure 7a's execution, in which no callbacks execute. The regular read of [y] can read its initial value of 0 from memory, but [x] is a phantom address that does not live outside the cache. Since the caches are initially empty, for [x] to be read, a value for it must first be created in the cache by running an OnMiss callback for [x]. Thus, we must outlaw

executions in which reads or writes of phantom addresses like [x] are not preceded by the execution of an OnMiss for them.

To this end, we add axioms (**VfWf** in Figure 6) to ensure that any  $R_{cb}$  or  $W_{cb}$  must be preceded in cbo by an  $M_e$ . Analogously, a Fl must be preceded by an  $E_e$ , denoting that an eviction has completed for the address being flushed (**EbWf**). An Fl may also occur before its address is ever brought into the cache (i.e., before there is anything to flush). These axioms are sufficient to outlaw Figure 7a's execution. Adding the required OnMiss gives us Figure 7b, which we discuss next.

