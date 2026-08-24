# <span id="page-16-4"></span>**K. Data synthesis prompts and examples**

#### Table 6 | Model prompts to generate subjects in a domain

```
Generate a list of major subjects in {domain}.
Output using the following format:
```
[subject1, subject2, ...]
```
```

Table 7 | Model prompts to generate schema in a domain

```
```
{demo_schema}
```
Generate another schema of similar formats for {domain}.
```

Table 8 | Model prompts to generate database entry

```
Schema
```
{schema}
```
```

Following the schema, write records in the subject {subject}. Make sure that the values align with the definitions in the schema and are consistent in different places. Use the following format to output:

```
```
{ "...": ..., "...": ..., }
```
```

Wrap the dictionary within ```.

Table 9 | Model prompts to validate database entry

```
Schema
```
{schema}
```
Database entry
```
{db_entry}
```
```

Please check whether the following conditions are satisfied:

Condition 1. The database entry strictly aligns with the fields and type definitions in the schema.

Condition 2. The values in the database entry are consistent across different places, e.g., id, name, etc.

Condition 3. The database content is logical, natural, and reasonable.

Output using the following format:

``` { "condition 1": "satisfied or not satisfied, "condition 2": "satisfied or not satisfied, "condition 3": "satisfied or not satisfied, } ```

#### Table 10 | Model prompts to generate functions

```
Demonstration function
```
{demo_function}
```
```

Following the formats of demonstration function, write frequently-used functions in {domain}. Wrap the functions within ```.

### Table 11 | Model prompts to generate intents

```
Functions
```
{functions}
```
Propose realistic intents in {domain} that could be solved by the functions above. Use
the following format to output:
```.
"purpose 1",
"purpose 2",
...
```.
```

### Table 12 | Model prompts to generate tasks

```
Functions
```
{functions}
```
Database
```
{database}
```
Propose a realistic task with the intent: {intent}. Use the following format to output:
```.
{task_template}
```.
```

#### Table 13 | Model prompts to evolve tasks

```
Functions
```
{functions}
```
Database
```
{database}
```
Old task: {task}
Make a new task by adding more complexity to the old task. You can add constraints,
involve more entities, make the situation more tricky, etc. Use the following format to
output:
```.
{task_template}
```.
```

#### Table 14 | Database schema example

```
{
"movies": {
"MMMMMMM": { movie_id
"movie_id": "MMMMMMM",
"title": "...",
"genres": ["Action", "Adventure", "Comedy", "Drama", "Horror", "Thriller", "Ro-
mance", "Science Fiction", "Fantasy", "Mystery"],
"runtime_minutes": ...,
"mpaa_rating": "...",
"languages_audio": ["..."],
"subtitles": ["..."],
"formats": ["2D", "3D", "IMAX", "Dolby"],
"release_date": "YY-MM-DD",
"end_of_run_est": "YY-MM-DD",
"cast": [
{ "name": "...", "role": "..." }
],
"crew": {
"director": "...",
"writer": "...",
"producer": "...",
"music": "..."
},
"synopsis": "..."
},
...
},
...
}
```

<span id="page-20-3"></span>Table 15 | The average number of calls on each tool when various models serve as the orchestrator to solve an instance (averaged across HLE, Frames and  $\tau^2$ -bench). Qwen-32B refers to Qwen/Qwen3-32B [27], Coder-32B refers to Qwen/Qwen2.5-Coder-32B-Instruct [24], Math-7B refers to https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct [25], Math-72B refers Qwen/Qwen2.5-Math-72B-Instruct [25], and Llama-70B refers to meta-llama/Llama-3.3-70B-Instruct [26]. Compared to other strong foundation models, Orchestrator-8B achieves better results (Table 1) while making few calls to GPT-5.

| Model                    | GPT-5 | GPT-5-mini | Qwen-32B | Coder-32B | Math-72B | Math-7B | Llama-70B | Local search | Web search | Code interpreter |
|--------------------------|-------|------------|----------|-----------|----------|---------|-----------|--------------|------------|------------------|
| Qwen3-8B                 | 6.0   | 0.5        | 1.4      | 0.5       | 0.0      | 0.0     | 0.0       | 0.8          | 1.2        | 1.6              |
| Nemontron-49B            | 5.1   | 1.6        | 0.5      | 0.8       | 0.1      | 0.1     | 0.3       | 0.7          | 0.9        | 1.4              |
| Llama-3.3-70B            | 1.8   | 0.0        | 0.1      | 0.0       | 1.4      | 0.3     | 4.8       | 0.6          | 1.4        | 1.3              |
| ${\it Qwen 3-235B-A22B}$ | 6.2   | 0.3        | 0.6      | 1.2       | 0.6      | 0.1     | 1.1       | 1.4          | 1.0        | 2.2              |
| Claude Opus 4.1          | 6.2   | 0.2        | 0.3      | 0.3       | 0.1      | 0.0     | 0.1       | 1.0          | 1.3        | 1.4              |
| GPT-5                    | 2.7   | 5.6        | 0.0      | 0.2       | 0.0      | 0.0     | 0.0       | 0.5          | 0.7        | 1.0              |
| Orchestrator-8B          | 1.6   | 1.7        | 1.3      | 0.2       | 0.0      | 0.1     | 0.0       | 1.8          | 0.7        | 0.8              |

<span id="page-20-2"></span>Table 16 | The cost and latency of LLMs in  $\tau^2$ -Bench. Orchestrator-8B consistently shows better performance with lower cost and latency.

| Tools            | Model(s)           | $\tau^2$ -Bench (†) | Cost $(\downarrow)$ | Latency $(\downarrow)$ |
|------------------|--------------------|---------------------|---------------------|------------------------|
|                  | Qwen3-8B           | 40.7                | 1.6                 | 2.3                    |
|                  | Llama-Nemotron-49B | 23.2                | 2.7                 | 3.6                    |
| Basic tools      | Llama-3.3-70B      | 17.6                | 3.1                 | 4.5                    |
| Basic tools      | Qwen3-235B-A22B    | 52.9                | 12.6                | 10.6                   |
|                  | Claude Opus 4.1    | 46.0                | 81.2                | 32.8                   |
|                  | GPT-5              | 77.7                | 31.3                | 20.2                   |
|                  | Qwen3-8B           | 72.3                | 27.9                | 18.4                   |
|                  | Llama-Nemotron-49B | 66.7                | 25.8                | 17.5                   |
| Basic tools,     | Llama-3.3-70B      | 55.8                | 20.1                | 14.2                   |
| Specialized LLMs | Qwen3-235B-A22B    | 75.6                | 30.0                | 22.6                   |
| Generalist LLMs  | Claude Opus 4.1    | 76.8                | 52.8                | 24.1                   |
|                  | GPT-5              | 62.3                | 18.2                | 14.5                   |
|                  | Orchestrator-8B    | 80.2                | 10.3                | 8.6                    |

### <span id="page-20-1"></span>L. Calculation of rewards for preference-aware benchmark

During training, we directly follow the Equation 2 to calculate rewards. In the evaluation, we use the following procedure. Following the definition in §3.2, we have a tool set  $\{t_1, t_2, ..., t_n\}$  and a rollout trajectory  $\tau$ , we consider the vector  $M^{\tau} = [m_{t_1}^{\tau}, m_{t_2}^{\tau}, ..., m_{t_n}^{\tau}, r_{\text{outcome}}^{\tau}, r_{\text{compute}}^{\tau}, r_{\text{latency}}^{\tau}]$ , where  $m_{t_{\bullet}}^{\tau}$  is the number of times tool  $t_{\bullet}$  is invoked in  $\tau$ . In the evaluation, we obtain the baseline vector  $M_0$  by running the starting checkpoint, e.g., Qwen3-8B. For example, if we would like to evaluate a checkpoint  $CKPT_s$  that is trained for s steps from a base Qwen3-8B model, then we first run Qwen3-8B on the benchmark and record the vector  $M_0^{\tau(e)}$  as the baseline vector for the Qwen3-8B's trajectory  $\tau(e)$  for each example e of the benchmark. We then obtain  $M_s^{\tau(e)}$  by running  $CKPT_s$  on the same example e.  $M_s^{\tau(e)}$  is normalized as

$$M_{\text{normalized},s}^{\tau(e)}[k] = \begin{cases} M_s^{\tau(e)}[k]/max(1, M_0^{\tau(e)}[k]) & \text{if } 1 \le k \le n+1\\ M_0^{\tau(e)}[k]/max(1, M_s^{\tau(e)}[k]) & \text{otherwise.} \end{cases}$$
(5)

We then proceed to calculate the final preference-aware reward for the example e as:

<span id="page-20-0"></span>
$$R^{e}(\tau) = \begin{cases} M_{\text{normalized},s}^{\tau(e)} \cdot P & \text{if } r_{\text{outcome}(\tau)} \\ 0 & \text{otherwise.} \end{cases}$$
 (6)

The benchmark result is calculated as the sum of  $R^e(\tau)$  over the examples e of the benchmark.