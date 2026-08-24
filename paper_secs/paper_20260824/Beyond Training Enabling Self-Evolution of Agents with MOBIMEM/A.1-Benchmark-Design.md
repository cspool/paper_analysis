# A.1 Benchmark Design

### A.1.1 Design Rationale

The benchmark evaluates three core capabilities of agent systems. First, agents must extract structured profile information from task execution traces containing implicit signals about user facts, preferences, habits, and experiences. Second, agents must store this information in memory systems that enable efficient retrieval. Third, agents must apply learned profiles to complete ambiguous tasks by inferring missing task parameters.

### A.1.2 Data Structure

Each user profile contains structured information across five task categories: shopping, hotel booking, travel, food delivery, and entertainment. Within each category, profiles specify 3–7 dimensions covering various aspects such as preferences, habits, constraints, and contextual information. Historical tasks explicitly encode user profile information through natural language instructions. Test tasks are intentionally ambiguous, omitting specific profile details to require agents to retrieve and apply learned profile memory.

## A.2 Task Format and Examples

#### A.2.1 Historical Task Format

Historical tasks are natural language instructions that explicitly contain user profile information. Each task is a complete sentence describing a specific action, with profile information (preferences, habits, constraints) embedded naturally.

Example historical tasks for a budget-conscious user:

"I need to buy a keychain organizer on Taobao—looking for something affordable from a domestic brand, beige color preferred, under 20 yuan, and I'd like next-day delivery if possible."

"Can you help me find high-speed rail tickets from Chengdu to Luzhou on the 12306 app? I'm visiting relatives and prefer a direct train around 11 AM, second-class seat is fine to keep costs down. I usually book three days ahead."

"I'd like to order a light lunch on Eleme—Chinese food, but not too spicy. Something budget-friendly with beef brisket and lots of vegetables would be perfect. Please have it delivered during my lunch break."

These tasks contain explicit profile signals including price preferences ("affordable", "under 20 yuan", "keep costs down", "budget-friendly"), brand preferences ("domestic brand"), style preferences ("beige color", "not too spicy"), timing constraints ("visiting relatives", "lunch break"), and behavioral patterns ("I usually book three days ahead").

### A.2.2 Shared Test Task Format

Shared test tasks are intentionally ambiguous, containing only high-level task descriptions without specific profile details. Agents must infer and apply learned profile information to complete these tasks.

### Example shared test tasks:

"I'm heading to Guangzhou tomorrow to meet some friends. Can you book me a train ticket on 12306, find a hotel on Ctrip near good transit options for one night, and order some household items on Taobao that I can bring with me?"

"Planning to relax at home this weekend. Could you order some Chinese food for me on Meituan, find some interesting short videos on Bilibili, and maybe get me a comfy cushion from Taobao?"

These tasks are ambiguous in several ways. Price constraints are unspecified: "book me a train ticket" does not indicate seat class or budget. Brand and style information is missing: "order some household items" does not specify brand, quality level, or style. Timing and context are unclear: "relax at home this weekend" does not provide exact delivery windows, viewing schedules, or personal taste information.

## A.3 LLM-Based Task Rewriting

To improve the realism of historical tasks, we employ an LLMbased rewriting process that transforms template-generated tasks into natural user utterances. The rewriting process takes structured task templates and produces conversational language while preserving all profile signals embedded in the original tasks.

### A.3.1 Rewriting Process

The rewriting process takes original tasks generated from templates along with user profile summaries as input. The LLM is then prompted to rewrite these tasks into more natural language while preserving the core task intent and all embedded profile information. The process produces rewritten tasks that maintain all profile signals but express them through conversational language that resembles authentic user requests.

#### Example rewriting:

Original: "Search for a pillow on Taobao, choose domestic brand and lowest price".

Rewritten: "I'm looking for an affordable pillow on Taobao—preferably memory foam from a domestic brand. Something really comfortable in beige would be great, and my budget is around 100 yuan. If you can get next-day delivery, that'd be perfect!"

The rewritten version maintains core profile signals (price, brand, style, delivery, timing), uses more natural and conversational language, adds contextual details that make the request sound authentic, and preserves all required profile dimensions while sounding like a real user query.

## A.4 Evaluation Methodology

### A.4.1 Evaluation Workflow

The evaluation follows a three-phase workflow.

Phase 1: Learning. The agent processes historical tasks (500 per user), extracts profile information and stores it in memory, and learning is evaluated by checking if extracted profile information matches the ground-truth profile.

Phase 2: Task Rewriting. The agent receives shared test tasks (30 ambiguous tasks), retrieves relevant learned profile information for each task, uses LLM to rewrite the ambiguous task into a personalized complete task description, and the rewritten task should incorporate learned profile information naturally.

Phase 3: Profile Matching. For each rewritten task, an LLM judge evaluates whether required profile information is present, compares the rewritten task against the ground-truth profile, and computes metrics including matched information count, total information count, and alignment score.

#### A.4.2 Required Profile Information Generation

To determine what profile information should be present in each test task, we employ an LLM-based analysis process. For each user, the process takes three inputs: the user's groundtruth profile, the user's historical task history, and the shared test tasks. The LLM analyzes each test task and identifies

which specific profile elements are necessary to complete that task for the given user.

### Example analysis prompt template:

You are an expert in user profile analysis. For each task in the shared task list, determine what profile information is required to complete that task.

#### User Profile:

[Coarse-grained profile information across different categories]

#### User Task History:

[Sample historical tasks showing fine-grained behavior patterns]

#### Shared Tasks:

[Test tasks to analyze, numbered 1, 2, 3, ...]

#### Output:

For each task, identify the required profile information:

```
Task 1: { element_1: value, ... }
Task 2: { element_1: value, ... }
...
```

This analysis produces a structured mapping from each test task to its required profile information, represented as key-value pairs (e.g., "price\_constraint: budget-friendly", "brand\_preference: domestic brand"). This approach ensures that evaluation is context-aware: different test tasks may require different subsets of profile information, and the required information is determined based on both the task nature and the user's historical behavior patterns.

### A.4.3 Evaluation Metrics

We evaluate systems using three metrics: write latency (time to update profile from each historical task), retrieval latency (time to retrieve profile information for each test task), and profile alignment (percentage of ground-truth profile information successfully retrieved, evaluated by an LLM judge).

For profile alignment evaluation, the system first identifies which profile information should be present in each test task based on task type and ground-truth profile. An LLM judge then evaluates whether the agent's rewritten task contains these required profile elements.

#### Example evaluation prompt template:

You are an evaluator assessing whether a personalized task reflects required user profile information. Determine if the following profile elements are present in the rewritten task.

#### Required Profile Information:

[Profile information based on user profile and task context]

#### Personalized Rewritten Task:

[Rewritten task text]

#### Instructions:

Analyze the rewritten task and determine whether each profile element is clearly reflected.

The judge returns a structured JSON response:

```
{
  "profile_check": [
    {
      "profile_element": "...",
      "expected_value": "...",
      "matched": true/false,
      "evidence": "..."
    },
    ...
}
```

Each element includes the profile element name, expected value (from user profile), a boolean "matched" field indicating whether the profile information is reflected in the task, and an "evidence" field providing justification.

Computed metrics: We compute per-task alignment as the percentage of required profile elements matched in each rewritten task, overall alignment as the average alignment across all 30 test tasks, and per-dimension accuracy as the matching accuracy for each profile dimension. These metrics quantify both the completeness and accuracy of profile learning and retrieval.

