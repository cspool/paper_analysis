# 4 Implementation

DMI comprises over 18K lines of Python code and leverages the pywinauto library [\[31\]](#page-13-21) to exercise UIA [1](#page-7-1) .

#### <span id="page-7-0"></span>4.1 UNG Construction

This section describes the methodology for constructing the UNG, which is formally defined in [§ 3.2.](#page-4-0)

Control identifier synthesis: We require a control identifier to label each UI control as a node in the UNG. Since UIA lacks guaranteed globally unique identifiers, we adopt an XPath-like control identifier:

primary\_id|control\_type|ancestor\_path

primary\_id uses the UIA automation\_id (if empty, falling back to the control name, or [Unnamed]); control\_type specifies the UIA-defined type (e.g., TabItem); and ancestor\_path provides a slash-delimited sequence of UI tree ancestors. We avoid index-based addressing since dynamic menus can shift indices unpredictably.

GUI ripping: The UNG models transition relationships between UI controls. The UNG is built via differential capture. Exploration proceeds with depth-first search (DFS). First, obtain the accessibility tree of the target application, from which a list of UI controls is captured. Then activate a candidate control (i.e., click) from the list and capture again. Newly revealed controls define navigation edges. New top-level or modal windows are detected via process\_id and window listeners. The exploration is fully automated.

Access blocklist: The DFS-based ripping requires the application to return to the prior state before exploring new branches. However, certain controls (i) trigger external transitions (e.g., an "Account" button opening a Web browser) or (ii) bring the application into states that cannot be exited using standard commands like Esc or Close. While state consistency can be force-restored by restarting the application after every interaction [\[22\]](#page-13-19), this incurs a prohibitive time cost. To bypass this bottleneck, we adopt a semi-automated approach by implementing a manual blocklist for these controls. The configuration leverages prior application knowledge and can be verified by monitoring the modeling process. This represents a deliberate trade-off: necessary human intervention in exchange for a massive gain in exploration efficiency. Maintaining the blocklist constitutes most of the

manual effort, but it can be automated with additional modeling time.

Context-aware exploration: Some controls are implicitly state-dependent and are only visible under specific conditions (e.g., PowerPoint's "Picture Format" tab appears only when an image is selected). Accordingly, we implement a context manager. We manually instantiate representative objects (e.g., inserting an image or a text box into the slides) along with their associated context types (e.g., image, text box). The explorer traverses each context independently and merges the results into a unified topology. Although this manual setup represents a small fraction of the overall manual effort, automating it remains challenging. Omitting it would compromise the completeness of the UNG.

Root node initialization: A virtual root is introduced, and controls on the initial screen are attached as its children. If multiple TabItem controls exist and one is active by default, we associate otherwise unscoped controls on the initial screen with that active tab control to ensure they are indexable. The process is fully automated.

