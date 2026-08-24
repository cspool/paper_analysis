# **A.4 Cost Analysis**

Table [8](#page-17-1) shows the cost incurred by running each stage of our data synthesis pipeline. With the exception of deduplication, which is done by GPT-4o, each stage of the pipeline is performed by Claude. Table [9](#page-17-2) shows the estimated per claim cost for the na¨ıve versus main approach based on estimated cost for 6 books. For human annotation, NoCha [\(Karpinska](#page-11-1) [et al.,](#page-11-1) [2024\)](#page-11-1) reports that their total cost of annotating 1,001 claim pairs is \$3,327 USD, so each claim costs around \$1.7.

| Stage                                                    | Cost                 |
|----------------------------------------------------------|----------------------|
| Book summary generation                                  | \$0.0021             |
| Chapter outline generation<br>Book-level claim synthesis | \$0.0107<br>\$0.0129 |
| Chapter-level claim synthesis<br>Deduplication           | \$0.0172<br>\$0.0021 |
| Verification                                             | \$0.0064             |
| Total                                                    | \$0.0514             |

<span id="page-17-1"></span>Table 8: Cost to run pipeline per claim (in US dollars, rounded to four decimal places).

|                             | NA¨IVE | CLIPPER |
|-----------------------------|--------|---------|
| Cost per claim (book-level) | \$0.09 | \$0.07  |
| Cost per claim (chap-level) | \$0.04 | \$0.02  |

<span id="page-17-2"></span>Table 9: Estimated cost for our NA¨IVE vs CLIPPER approach (rounded to two decimal places)

