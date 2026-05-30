# V. PRO-PROPHET SCHEDULER

<span id="page-5-1"></span>Previous works introduce a search process (corresponding to Plan primitive), model states transferring (corresponding to Trans and Agg primitives) to balance the load. However, their execution is blocked by other operators due to data dependency, constraining further improvement of training efficiency. In this section, we introduce designs of the scheduler which extensively overlap computation and communication based on the locality described in Sec. [II.](#page-1-0)

