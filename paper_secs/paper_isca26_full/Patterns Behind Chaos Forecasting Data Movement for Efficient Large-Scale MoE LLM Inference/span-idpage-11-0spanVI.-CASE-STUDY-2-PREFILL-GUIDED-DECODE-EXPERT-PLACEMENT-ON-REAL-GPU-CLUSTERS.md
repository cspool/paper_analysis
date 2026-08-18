# <span id="page-11-0"></span>VI. CASE STUDY 2: PREFILL-GUIDED DECODE EXPERT PLACEMENT ON REAL GPU CLUSTERS

#### A. Introduction

Workload imbalance is one of the biggest challenges in large-scale MoE serving (200+ GPUs). EPLB [68] addresses this by dynamically adjusting expert placement, but it is triggered every 3000+ steps and relies on periodically collected profiling data [69]. A natural question then arises: how to set expert placement for the initial ~1000 decode tokens when no profiling data are yet available? This is especially pressing for short-output requests, for which EPLB never collects enough data to be effective. Inspired by Insight 1, which reveals temporal correlation between prefill and decode stages, we propose leveraging prefill-stage expert selection information to guide expert placement for initial decode steps.

