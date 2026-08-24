# <span id="page-10-6"></span>**Algorithm 2** Iterative Pruning via First-Token Surprisal

```
Require: Coarse-grained Pruned CoT C_{coarse}, Max Tokens
     L_{max}, Model M, Tokenizer T
Ensure: Fine-grained Pruned CoT C'
 1: function FineGrainedPrune(C_{coarse}, L_{max}, M, T)
         if Length(T(C_{coarse})) \leq L_{max} then
 3:
              return C_{coarse}
 4:
         end if
         S \leftarrow \text{SplitStepsByBlankLine}(C_{coarse})
 5:
         SurprisalScores \leftarrow CalculateAll(S, M, T)
         StepsToPrune
     SortByScore(S, SurprisalScores)
 8:
         S_{current} \leftarrow S
 9:
         for each step s_{prune} in StepsToPrune do
10:
              S_{temp} \leftarrow S_{current} \setminus \{s_{prune}\}
11:
              C_{temp} \leftarrow \text{Join}(S_{temp})
              if \operatorname{Length}(T(C_{temp})) \leq L_{max} then
12:
13:
                  S_{current} \leftarrow S_{temp}
14:
                  break
15:
              end if
              S_{current} \leftarrow S_{temp}
16.
17:
         end for
18.
         C' \leftarrow \text{Join}(S_{current})
19:
         return C'
20: end function
```

