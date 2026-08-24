# Task Instructions:

### 1. Parse the structured plan:

Parse the plan or summary to understand the parallel execution requirements.

\*\*CRITICAL: All goals MUST be advanced simultaneously in parallel. Each goal's paths MUST be executed sequentially (one path at a time per goal).\*\* ### 2. Execute parallel tool calls:

For each goal in the plan, execute the specified tools in parallel according to the paths defined.

\*\*MANDATORY: Advance ALL goals concurrently. Within each goal, execute paths sequentially (never parallelize paths within a single goal).\*\* ### 3. Handle path diversity:

For each goal, if multiple paths are provided, execute them sequentially as fallback options if the primary path fails.

\*\*ABSOLUTE REQUIREMENT: NEVER prematurely assume a goal is achieved. Continue advancing ALL other goals in parallel while handling fallback paths for any individual goal.\*\*

### 4. Process results:

Synthesize information from all tool outputs to generate comprehensive responses that address all goals.

\*\*ESSENTIAL: Do NOT consider any goal achieved until explicitly verified. Maintain parallel advancement of ALL goals throughout synthesis.\*\* ### 5. Final answer:

Once all goals are addressed, consolidate their results, and ensure that the consolidated outcome can accurately and correctly answer the original task, then call the 'final\_answer' tool with such consolidated results.

\*\*FINAL CONDITION: Only proceed when ALL goals are resolved. NO early termination of individual sub-goals, and the consolidated results must be capable of accurately and correctly answering the original task.\*\*

