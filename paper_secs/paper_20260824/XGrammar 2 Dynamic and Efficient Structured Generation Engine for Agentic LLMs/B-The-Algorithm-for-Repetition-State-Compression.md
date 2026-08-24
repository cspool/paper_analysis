# B The Algorithm for Repetition State Compression

Algorithm 3 shows the algorithm to perform the repetition state compression algorithm in detail.

#### C More Explanation of the Hash Algorithm

For most FSMs, this algorithm generates a consistent hash value. However, there are two cases where it may produce different hash values for FSMs with the same structure: (1) the FSM is not a deterministic finite automaton (DFA); (2) there are duplicated FSMs in the grammars, and they are referenced by a common FSM. In

#### <span id="page-10-3"></span>Algorithm 3 Repetition State Compression Algorithm

```
Input: A triplet (min, max, context)
Output: A expression expr
Const: kRepetitionThreshold \leftarrow t
if max \le t then
  expr \leftarrow Expand(min, max, context)
  return
end if
if min < t then
  other\_choices \leftarrow Expand(min, t, context)
  choice \leftarrow Concat(Repeat(t, max, context),
  Expand(0, max - t, context))
  expr \leftarrow Union(choice, other\_choices)
  return
end if
for i \in \text{range}(t) do
  expr \leftarrow Concat(expr, context)
end for
expr \leftarrow Concat(expr, Repeat(min - t, max - t, context))
function Repeat(min, max, context)
  return a repetition expression that accepts context at least
  min times and at most max times
end function
function Expand(min, max, context)
  return an explicit expansion equivalent to the repetition ex-
end function
function Union(expr_1, expr_2)
  return an expression that matches either expr_1 or expr_2
end function
function Concat(expr_1, expr_2)
```

these cases, the algorithm may generate inconsistent hash values. Nevertheless, this does not undermine the sufficiency of the algorithm: if two FSMs share the same hash value, they must have the same structure. In addition, in our implementation, we attempt to transform most FSMs into DFAs. Moreover, since we have a deterministic conversion function for JSON Schemas and regular expressions, two FSMs with the same structure are likely to produce the same hash value due to this deterministic transformation. As a result, we can detect and reuse identical structures within and across grammars maximally.

**return** an expression that matches  $expr_1$  followed by  $expr_2$ 

