# [EXAMPLE]

In a hypothetical world, there are a number of cities. Each city has a one-way connection to only one other city via a specific transit method. The details of the cities are as follows:

Fort Worth is a lively city. You can travel from Fort Worth to Manchester by ferry.

Leeds is a lively city. You can travel from Leeds to London by bus.

Manchester is a lively city. You can travel from Manchester to Indianapolis by plane.

Houston is a lively city. You can travel from Houston to London by ferry.

Charlotte is a lively city. You can travel from Charlotte to Charlotte by bus.

London is a lively city. You can travel from London to San Antonio by train.

San Antonio is a lively city. You can travel from San Antonio to Kitchener by train.

Seattle is a lively city. You can travel from Seattle to London by train.

Indianapolis is a lively city. You can travel from Indianapolis to Houston by ferry.

Now find the route from Manchester to Kitchener based on the information above.

### <Route>

From Manchester, take a plane to Indianapolis.

From Indianapolis, take a ferry to Houston.

From Houston, take a ferry to London.

From London, take a train to San Antonio.

From San Antonio, take a train to Kitchener.

</Route>

#### [PROBLEM]

#### problem context

Now find the route from src city to dst city based on the information above. Some reminders:

- All connections are one-way. You can solve the problem by iteratively finding the next city to travel to until you reach the destination city.
- Follow the specific format for the route output. Mark the route with <Route>and </Route>tags.

### Prompt H.4: Template for the Countdown Task

### [TASK]

You will be given four numbers and a target number, your task is to find a way to use all four numbers exactly once, along with the basic operations (+, -, \*, /), to reach the target number.

### [RULES]

- You can use each number exactly once.
- You can use the four basic operations (+, -, \*, /).
- The intermediate results must be integers (no decimals allowed).
- The intermediate results must be positive.
- The intermediate results will not exceed 2000.

### [APPROACH]

We will solve the problem by searching. Starting from a given set of four numbers, we will follow this search process:

- At each state, we will consider all possible number pairs (in order) from the current number set. Choose one pair and apply one of the four basic operations to them to obtain a new number.
- \* If there are still numbers left, we will add the new number to the number set and continue the search.
- \* If we have used all numbers, we will check if the new number is equal to the target number. If it is, we have found the solution. Otherwise, we will backtrack.
- Suppose the two numbers we choose are a and b (where a >= b). We will try the four options (a + b), (a - b), (a \* b), (a / b) to obtain the new number. Remember to always use the larger number as the first operand.
- If the new number is a decimal, or exceeds the maximum intermediate result, we will discard this branch and backtrack.
- We will continue this process until we reach the target number with four numbers used or exhaust all possible combinations.

### [EXAMPLES]

few shot examples

