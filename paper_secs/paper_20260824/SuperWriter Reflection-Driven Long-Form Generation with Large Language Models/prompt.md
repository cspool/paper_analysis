# **prompt**

Please refer to the evaluation criteria outlined below:

```
{evaluation_criteria}
```

#### Task:

You are tasked with evaluating the query: '{query}'. From the "Special Criteria" section, select 3 relevant criteria, and from the "General Criteria" section, select all criteria (Relevance, Coherence, Clarity), for a total of 6 criteria.

#### Be sure to:

- 1. Think step-by-step about why each criterion is relevant to the query.
- 2. Think step-by-step through the query and how each criterion applies.
- 3. Provide a brief analysis for each selected criterion on how it applies to the query.
- 4. Integrate the above reasoning into the Definition and Standards sections of each criterion. {format\_query}

```
prompt
### Query: {query}
### Result: <start> {clean_res} <end>
### Evaluation Standard: {json.dumps(evaluate_standard,
ensure_ascii=False)}
Based on the provided info, perform a rigorous evaluation. {format_eval}
```

## <span id="page-22-0"></span>A.5 Evaluation Prompt for Win-Rate Judgment

