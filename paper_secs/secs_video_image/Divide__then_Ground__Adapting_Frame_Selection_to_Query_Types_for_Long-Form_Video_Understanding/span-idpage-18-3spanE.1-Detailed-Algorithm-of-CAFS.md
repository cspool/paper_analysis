# <span id="page-18-3"></span>**E.1 Detailed Algorithm of CAFS**

Algorithm [1](#page-19-1) details our CAFS method. The process is structured into three sequential stages, taking a frame-to-frame distance sequence *d* = [*d*1*, . . . , dM*−1] and their corresponding original frame indices *I* = [*I*1*, . . . , IM*] as input, to produce a final set of *r-frame* indices, r\_idx.

**Initial peak detection.** First, we identify all potential content boundaries. It iterates through the distance sequence, identifying any point *d<sup>i</sup>* that is a local maximum, defined as being greater than its two immediate neighbors (*di*−<sup>1</sup> *< d<sup>i</sup> < d<sup>i</sup>*+1). The indices *i* of all such local maxima are collected into an initial peaks set.

**Topographic prominence filtering.** Second, we prune the peaks set, retaining only the most significant content transitions. For each peak *j* ∈ peaks, it calculates its "prominence" by finding the lowest base levels to its left (*l*min) and right (*r*min). The prominence is then defined as the peak's height *d<sup>j</sup>* minus the higher of its two bases (prominence = *d<sup>j</sup>* − max(*l*min*, r*min)). This metric quantifies how much a peak "stands out" from the surrounding distance signal. Only peaks whose prominence exceeds a threshold (e.g., 0.1) are added to the filtered\_peaks set, effectively discarding minor, localized fluctuations.

**R-Frame selection.** Finally, we generate the output by identifying frames that best represent the stable content *between* these significant transitions. The algorithm iterates through consecutive pairs of prominent peaks (*p*1*, p*2)

