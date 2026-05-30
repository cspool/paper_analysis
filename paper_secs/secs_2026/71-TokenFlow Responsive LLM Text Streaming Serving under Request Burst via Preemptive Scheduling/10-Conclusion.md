# 10 Conclusion

We presented TokenFlow, an optimized LLM serving system that significantly enhances LLM text streaming performance through buffer-aware scheduling and hierarchical KV cache management. By dynamically aligning token generation rates with user consumption patterns and proactively managing GPU memory, TokenFlow achieved up to 82.5% higher effective throughput and 80.2% shorter time-to-firsttoken (TTFT). Extensive experiments demonstrated that TokenFlow outperforms state-of-the-art systems across diverse workloads and hardware configurations while sustaining smooth streaming quality. These results establish Token-Flow as a robust and efficient solution for real-time LLM applications.

