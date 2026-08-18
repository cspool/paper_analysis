# *C. Decoding Scheduling*

Since decoding instances operate independently, we can reuse existing scheduling strategies [35], [37], [47]. Currently, we extend the "virtual usage" proposed by Llumnix [37] in decoding scheduler: The KV cache slots of requests with ongoing cache transfer is treated as virtual sage. During scheduling, each new request is routed to the instance with the highest freeness rate, defined as the ratio between available slots (excluding virtual usage) and the active batch size. To improve load estimation accuracy, the scheduler updates slot statistics each time a request returns its decoding output.

```
# HTTP API for scheduler metadata update
@app.post("/update")
async def update(http_request)
# CDSP Scheduler's Metadata
@dataclass
class CDSPScheduleMetadata:
    improvement_rate_mapping: Dict[float, float]
    sp_size_candidates: List[int]
    improvement_rate_update_period: float
# Scheduler interface augmentation
class Scheduler:
    def initialize_schedule(
        self,
        init_improvement_rate,
        latency_model_map,
        cdsp_schedule_metadata
    )
    def update_schedule(
        self, new_cdsp_schedule_metadata
    )
    def cdsp_schedule(
        self, prefill_request
    )
# Model parallelism initialization interface
def initialize_model_parallel(
    prefill_tp, prefill_sp, decoding_tp, decoding_dp
)
```

Code Listing 1. Interface Modification

