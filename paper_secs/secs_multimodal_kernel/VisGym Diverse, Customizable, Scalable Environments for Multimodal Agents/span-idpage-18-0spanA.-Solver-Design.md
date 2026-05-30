# <span id="page-18-0"></span>**A. Solver Design**

This section provides detailed descriptions of the multi-step solvers introduced in Sec. [2](#page-2-2) and used for supervised finetuning across all environments.

**Colorization.** The solver computes how far the current hue and saturation are from the target, breaks those differences into small incremental steps, and outputs a sequence of rotate and saturate actions that move steadily toward the correct color. If a target number of steps is requested or if the color is already close enough, it fills the sequence with reversible rotate/saturate pairs that cancel out and don't change the final state.

**Counting.** *mark\_all strategy:* The solver places a dot at the center of each target instance, then submits the correct total count and stops. *guess\_only strategy:* The solver directly submits the correct total count and stops, without placing any dots.

**Jigsaw.** *reorder strategy:* The solver computes a single permutation payload that, when applied via the reorder' action, instantly rearranges the current pieces into their correct target positions. *swap strategy:* The solver generates a minimal sequence of swap' actions by repeatedly finding a misplaced piece and swapping it with the piece at its correct target location. If a target number of steps is requested, it pads this sequence with reversible pairs of swaps (*e.g.*, swapping two pieces and then immediately swapping them back) until the desired length is reached.

**Matchstick Equation.** *bfs strategy:* The solver finds the shortest possible sequence of move' actions to correct the equation using a Breadth-First Search (BFS) and then stops. *dfs strategy:* The solver finds a solution using a Depth-First Search (DFS), producing a sequence of move' actions and undo' actions that represent its full exploratory and backtracking process before stopping. *sos strategy:* The solver first finds the shortest solution path (via BFS), then pads this path by inserting random, reversible detours. Before an optimal step, it takes one or more random move' actions and immediately undo'es them, returning to the optimal path before proceeding.

**Matchstick Rotation.** The solver first performs one or more translation-only move' actions, which are typically unit-length moves in the general direction of the target. It then executes a final move' action that applies the entire required rotation and corrects any remaining translation error, before stopping.

**Maze 2D.** The solver uses a graph search algorithm to find the optimal coordinate path from the agent to the target, which is converted into the shortest sequence of move' actions. If a target number of steps is requested, the solver pads this optimal sequence by inserting random, reversible move' pairs (*e.g.*, move up' followed by move down') at valid locations along the path until the desired length is met, before stopping.

**Maze 3D.** The solver uses a graph search algorithm to find the optimal coordinate path from the agent's location to the target. It then converts this path into the shortest sequence of turn' (left, right, or around) and move' actions required to follow that path, accounting for the agent's current orientation. If a target number of steps is requested, the solver pads this optimal sequence by inserting random, reversible turn' pairs (*e.g.*, turn left' followed by turn right') at locations along the path until the desired length is met, before stopping.

**Mental Rotation 2D.** The solver first calculates the shortest total rotation angle required to align the current image with the target. If the requested number of steps is 1, it outputs a single rotate' action for that total angle. If a larger number of steps is requested, it stochastically divides the total rotation into that many smaller rotate' actions, which are executed sequentially and sum to the correct total angle, before stopping.

**Mental Rotation 3D (Cube).** The solver decomposes the total required rotation into its yaw, pitch, and roll components. It then corrects each component sequentially. Before applying the corrective 'rotate' action for a specific axis (*e.g.*, yaw), it first executes a padding sequence of four 90-degree rotations around that same axis. After this 360-degree padding, it applies the single action to correct the yaw. It repeats this pad-then-correct process for the pitch and roll axes, then stops.

**Mental Rotation 3D (Objaverse).** The same as Mental Rotation 3D (Cube).

**MuJoCo Fetch (Pick-and-Place).** The solver is a state-machine-based oracle. It follows a sequence: (0) move the gripper to a safe height above the object, (1) open the gripper, (2) descend to the object, (3) close the gripper to grasp. (4) Once grasped, it moves the object directly toward the 3D goal position using a greedy, per-axis strategy (correcting the axis with the largest error at each step). (7) Finally, it holds the object at the target location and stops.

**MuJoCo Fetch (Reach).** The solver is a greedy, per-axis oracle. At each step, it identifies the single axis (x, y, or z) with the largest error between the gripper and the goal. It then outputs a 'move' action along that single axis to reduce the error, repeating this process until the goal is reached, at which point it stops.

**Patch Reassembly.** The solver uses a backtracking search to find the optimal sequence of 'place' actions that perfectly tile the grid. If a target number of steps is requested, it pads this sequence by repeatedly inserting "mistake-and-correct" actions: it finds a correct 'place' action in the solution, finds a valid wrong location for that piece, and inserts this "mistake" 'place' action immediately before the "correct" 'place' action. If no valid mistakes can be found, it falls back to inserting a 'remove' and a duplicate 'place' action. This repeats until the desired number of 'place' actions is met.

**Referring Dot-Pointing.** The solver first samples a random pixel from within the target object's segmentation mask and also calculates the mask's center of mass. It then generates a sequence of 'mark' actions by linearly interpolating from the random starting point to the center of mass over the requested number of steps. The final action in this sequence places a mark at the exact center of mass, which is then followed by a 'stop' action.

**Sliding Block.** The solver uses a Breadth-First Search (BFS) to find the shortest sequence of 'move' actions from the current board state to the target configuration. If a target number of steps is requested, it pads this optimal path by first reconstructing all intermediate board states. At each state, it identifies all valid "back-and-forth" moves (*e.g.*, move block 1 right, then move block 1 left). It then randomly samples from these opportunities and inserts the required number of 'move' and 'reverse-move' pairs into the solution path until the desired length is met, before stopping.

**Video Unshuffle.** *reorder strategy:* The solver computes a single permutation payload that, when applied via the 'reorder' action, instantly rearranges the shuffled frames into their correct chronological order, then stops. *swap strategy:* The solver generates a minimal sequence of 'swap' actions to sort the frames. It iterates through the positions, and if a frame is in the wrong place, it finds the correct frame and swaps it into its target position, repeating until all frames are sorted, then stops.

**Zoom-In Puzzle.** The same as Video Unshuffle.

