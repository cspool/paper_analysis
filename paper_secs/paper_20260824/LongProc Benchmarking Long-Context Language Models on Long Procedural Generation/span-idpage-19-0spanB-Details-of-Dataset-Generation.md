# <span id="page-19-0"></span>**B Details of Dataset Generation**

### <span id="page-19-1"></span>**B.1 HTML to TSV**

This task involves extracting the information from an HTML page into a structured tabseparated TSV output. An example can be found in Example [F.1.](#page-25-1)

We build upon the *arborist* dataset from [Li et al.](#page-13-2) [\(2024c\)](#page-13-2) where each data example includes a set of web pages of a website scraped from the Internet and a task a program synthesis model can perform based on the web pages. We manually inspect the arborist dataset and select 56 websites that satisfy the following conditions: (1) The web pages contain at least one structured table-like component that can be converted into TSV files; (2) The table-like component has enough rows and properties so that the corresponding TSV file has more than 2K tokens.

For each of the selected websites, we perform the following cleaning steps: (1) If the website consists of multiple HTML files, we merge them into a single HTML file from which the table component can be extracted; (2) We remove any CSS style, metadata, JavaScript script, and comment from the HTML file, and remove unnecessary attributes of HTML tags (such as alt text); (3) We simplify the cleaned HTML tree by removing empty tags; (4) We manually check the cleaned HTML file to make sure that a table can still be extracted from it.

After cleaning the input HTML file, we first extract the ground truth TSV file using heuristics (e.g. extracting the table tag from the HTML file). Because natural websites on the Internet do not necessarily use the formats from common heuristics, we manually annotate the ground truth TSV when the heuristics fail.

We compute the length (in tokens) of the extracted ground truth TSV files, and depending on the length of the ground truth of each website, we split each website into non-overlapping subsamples so that we can obtain more data points for the 8K level and can create the 2K and 0.5K levels. If the ground truth TSV table exceeds 10K tokens, we split the table in half into two non-overlapping tables, each with 2K-8K tokens. After splitting the ground truth, we split the cleaned input HTML file into two HTML files accordingly. In this way, we obtain two data points for the 8K test set by subsampling from a single website. Similarly, we split each website into 1-3 subsamples at the 2K output level and 1-3 subsamples at the 0.5K output level.

To make the task more challenging, we add a filtered version of each website at the 8K and 2K levels where the model is prompted to extract only some rows based on a given specification. Note that the output length after filtering is shorter than the length without filters. We manually set the filtering condition such that the output length of an 8K-level input after filtering is at least 2K, and the length of a 2K level input after filtering is at least 0.5K.

