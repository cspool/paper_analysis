# Examples

## Requires Deep Think

- Input: *"Let A, B, and C be three sets. Prove that* A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)*."* Output: "YES – This problem involves set operations and requires a formal mathematical proof with multi-step logical deductions."
- Input: *"If the speed of light is the cosmic limit, but the universe is expanding, is it possible for two regions to be permanently unobservable from each other?"* Output: "YES – This question involves relativity, cosmology, and hypothesis-based deduction, requiring systematic reasoning."
- Input: *"On an 8x8 chessboard, if two opposite corners are removed, can it be completely covered by 2x1 dominoes?"* Output: "YES – This requires constructing a counterexample, analyzing the board's parity, and recursive reasoning."

## Does Not Require Deep Think

- Input: *"What is* 2 <sup>10</sup>*?"* Output: "NO – This is a straightforward computation that can be answered directly."
- Input: *"Tom is 5 years older than Alice. Alice is 10 years old. How old is Tom?"* Output: "NO – This is a basic arithmetic problem that does not require complex reasoning."
- Input: *"Why is water heavier than oil?"* Output: "NO This is a factual question about density that can be answered using common knowledge."

## <span id="page-19-0"></span>C.5 Experimental Setup and Implementation Details

We employ the high-throughput inference engine vLLM as the execution framework, with generation hyperparameters set to a temperature of 0.8, top-p of 1.0, and a repetition penalty of 1.05. For all evaluated methods, we conduct five independent inference runs per question. The final accuracy and token usage are computed by averaging across these runs.

For our Adaptive Think approach, we segment the model's reasoning trajectory into discrete steps based on paragraph boundaries. Specifically, the occurrence of a double newline ("\n\n") is treated as a trigger flag, prompting an entropy-based decision on whether to continue the reasoning process. To mitigate noise and ensure meaningful intermediate content, we enforce a minimum length of 120 characters per reasoning step.

It is worth noting that for the two mathematical benchmarks—GSM8K and AIME2025—we define the answer candidate space using a top − K = 5. Candidate answer sequences are retrieved via a tree search procedure (detailed in [§C.3\)](#page-17-0), where token continuations are explored recursively. The maximum tree depth is capped at 10, which is sufficient to accommodate the full range of answer lengths found in both benchmarks.

