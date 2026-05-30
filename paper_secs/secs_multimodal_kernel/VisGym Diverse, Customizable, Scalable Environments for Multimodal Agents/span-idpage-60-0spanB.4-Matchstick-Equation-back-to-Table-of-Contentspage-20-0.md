# <span id="page-60-0"></span>B.4 **Matchstick Equation** [\(back to Table of Contents\)](#page-20-0)

```
Instruction I
You see an image of a broken matchstick equation.
Your goal is to fix the equation by moving ONE match per action.
Symbols (left->right) with indices:
[0:'0', 1:'-', 2:'7', 3:'3', 4:'/', 5:'3', 6:'=', 7:'9', 8:'9']
Available actions:
1. 'move': Remove one match from segment 'src_seg' of symbol at 'src_idx', then add it to segment '
dst_seg' of symbol at 'dst_idx'. Format: '('move', [src_idx, src_seg, dst_idx, dst_seg])' where:
   - src_idx, dst_idx ∈ [0, N-1], with src_idx ̸= dst_idx
   - src_seg, dst_seg ∈ matchstick_puzzles.TOTAL (e.g., {0..12})
   - The move must result in valid symbols at BOTH positions.
2. 'undo': Revert the last move (if any). Format: '('undo', 'undo')'
3. 'stop': Submit your current equation as final. Format: '('stop', 'stop')'
Success: The submitted equation must be mathematically correct (evaluated as lhs == rhs).
Segment legend (indices depend on symbol):
  0..6 : 7-seg digits (a,b,c,d,e,f,g), 6 is also the horizontal stroke used by '+'
  7 : plus vertical stroke (used by '+')
  8 : the multiply sign that goes from top left to bottom right
  9 : the multiply sign that goes from top right to bottom left
  11,12: equals upper/lower bars (used by '=')
A segment is valid for a symbol only if the resulting set of segments maps to a known glyph.
Please respond with exactly one action and its arguments in the specified format. For example:
- To move a match: '('move', [0, 6, 2, 0])'
- To undo: '('undo', 'undo')'
- To stop: '('stop', 'stop')'
This is step 1. You are allowed to take 99 more steps.
Observation o0 Action a1:
                                                       ('move', [7, 5, 0, 6])
```

![](_page_60_Figure_2.jpeg)

![](_page_60_Figure_3.jpeg)

![](_page_61_Figure_0.jpeg)

Environment feedback: Action executed successfully. This is step 4. You are allowed to take 96 more steps.

