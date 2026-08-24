# B Multi-Task Execution in Real-World Scenarios

To complement the performance evaluation presented in our main paper, this appendix provides detailed descriptions of the real-world testing scenarios used to evaluate the Agent Scheduler's multi-task execution capabilities. These scenarios involve complex workflows requiring data transfer and synchronization across multiple applications, representing typical user interactions in practical mobile usage. We measure endto-end execution latency across three execution modes: Serial, Coarse-grained Parallel, and Fine-grained Parallel.

## B.1 Execution Modes in Practice

We use a representative shopping and social networking scenario to illustrate how the three execution modes handle realworld tasks differently.

Scenario: Query the price of a specific item in two different shopping applications (App A and App B), and then send the gathered information to a contact via a social networking application (App C).

#### B.1.1 Serial Execution

The agent executes tasks strictly sequentially. It first completes the price query in App A, then performs the price query in App B, and finally launches App C to send the message.

*Timeline:* 
$$T_{total} = T_{AppA} + T_{AppB} + T_{AppC}$$

#### B.1.2 Coarse-grained Parallelism

The agent identifies independent sub-tasks at the application level. It executes the price queries in App A and App B simultaneously. The system blocks the execution of App C until both App A and App B have fully completed their tasks and returned the results.

*Timeline: Ttotal* = max(*TAppA*,*TAppB*) +*TAppC*

#### B.1.3 Fine-grained Parallelism

The agent exploits step-level parallelism by analyzing data dependencies. While the price query sub-tasks in shopping apps (App A and App B) are being executed, the agent simultaneously operates in App C (e.g., searching for the contact, entering the chat interface, and activating the input field). The execution in App C is suspended *only* at the specific step where the message content (the prices) is required. Once the query results from App A and App B become available, the execution in App C resumes immediately to send the message.

*Timeline: Ttotal* = max(*TAppA*,*TAppB*,*TAppC*\_*setup*) + *TAppC*\_*send*

