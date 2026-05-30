# B. Case Study

To better illustrate the reasoning and interaction capabilities of Fast-dLLM v2 (7B), we conducted a detailed examination of both single-turn and multi-turn dialogue scenarios. Representative examples are presented in Table [5](#page-15-0) and Table [6.](#page-16-0)

Single-turn Dialogue Scenarios. As shown in Table [5,](#page-15-0) Fast-dLLM v2 is capable of handling complex queries in a single interaction. In the *Math* example, the model correctly analyzes the rational function to determine the number of vertical asymptotes, applying algebraic factoring and solving for the undefined values of . In the *Code* section, the model generates a correct and recursive Python implementation of the Tower of Hanoi problem, along with an appropriate explanation of the input parameters and output.

Multi-turn Dialogue Scenarios. Table [6](#page-16-0) highlights multi-turn dialogues where Fast-dLLM v2 retains context and builds upon previous turns. The *Daily life* example illustrates the model's ability to perform temporal reasoning, such as computing the number of years since a company was founded and determining its future anniversary. The *Math* example showcases step-by-step logical reasoning to solve a real-world arithmetic problem involving truckload capacity and total delivery time. The model effectively breaks the problem into sequential steps, performs intermediate calculations, and presents the final result in the required format (hours and minutes).

These case studies collectively demonstrate Fast-dLLM v2's strength in mathematical reasoning, code generation, temporal understanding, and contextual coherence across both single-turn and multi-turn settings.

