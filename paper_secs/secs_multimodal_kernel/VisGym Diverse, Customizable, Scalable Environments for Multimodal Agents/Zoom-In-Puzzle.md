# Zoom-In Puzzle

• The model finalizes the arrangement immediately without performing any swaps, reordering, or verification, accepting the initial sequence as correct regardless of its accuracy. For example, it issues a "stop" command on the first step even if the order is incorrect.

### **Matchstick Rotation**

- The model issues fixed or monotonically decreasing movement and rotation magnitudes without attempting to estimate the unknown scale or using exploratory actions to resolve scale ambiguity, proceeding as if the appropriate step size is already known.
- Action sequences do not adapt based on feedback or observed outcomes; the model follows a predetermined or repetitive strategy without checking if moves are effective or responding to evidence from the environment.

