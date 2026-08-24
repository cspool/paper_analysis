# <span id="page-25-0"></span>F Additional Data Examples

```
Example F.1: example data point of HTML to TSV
HTML Page
 <a href="https://www.enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.com/enables.c
Search</title>
</head>
<body>
<div>
<div><img/><span>1.</span></div>
<h3><a>AAK Autoservices</a></h3>
<h4><a>Units 62-63 John Wilson Business Park, Kent, CT5 3QT</a></h4>
</div>
</div>
<div>
<div>
<div><a><img/><img/><img/></div>
<div>From 758 surveys <br/>100% recommendation score</div>
</div>
</div>
</div>
</div>
</div>
<div>
<div>
<div>
<div>
<strong>Garage type: </strong>Independent Garage
<strong>Phone: </strong>03301302600
<strong>Code: </strong>Service & mp; Repair
```

```
</div>
</div>
<div>
<div>
<div>
<div>
<div>
<p><img alt="Subscription to Service & Repair"/></p>
<div role="group">
<div role="group"><a>Info.<span><i/></span></a></div>
<div role="group"><a>Rate <span><i/></span></a></div>
</div>
</div>
</div>
</div>
</div>
<div>
<div>
<div>
<div>
... ...
</body>
... ...
</html>
```

### **Target Information**

Based on the HTML webpage above about Garage search, extract the following properties from the items listed on the webpage: (1) Title; (2) Address; (3) Garage type; (4) Phone; Only extract the rows where the garage type is "Independent Garage".

#### **TSV**

AAK Autoservices Units 62-63 John Wilson Business Park, Kent, CT5 3QT Independent Garage 03301302600

Autocare Centre Chepstow Unit 6 Bulwark Business Park Bulwark, Monmouthshire, NP16 5QZ Independent Garage 0330 1303262

Garej Arwyn Cyf Unit Ah Penygroes Ind Est, Gwynedd, LL54 6DB Independent Garage 03301300767

Horsham Motor Services Unit 3 Parsonage Farm Ind Est Parsonage Way, West Sussex, RH12 4AQ Independent Garage 03301301956

Robin Hood Garage Ltd Unit 1, Robin Hood Works Wakefield Road, Yorkshire, HD6 1PE Independent Garage 03301304478

Steve Wood Motors Ltd High Street, Lincolnshire, LN12 2HE Independent Garage 01507 441215 Chester Auto Services Bumpers Lane, Cheshire, CH1 4LT Independent Garage 01244 377747 Hardwick Motor Company Ltd Hardwick Lane, Nottinghamshire, NG17 5EJ Independent Garage 01623555512

Dukes AutoTech Greensplat, Cornwall, PL26 8XY Independent Garage 01726 337171 Nailsea Auto Electrical Unit 42, North Somerset, BS48 4NE Independent Garage 01275261005

### <span id="page-26-0"></span>Example F.2: example data point of Pseudocode to Code

```
Pseudocode Annotations
define the main function
——–str = string array of size 12
——–n, i, j, ck = int
——–read n then str[0]
——–set str[1] to "vaporeon"
——–set str[2] to "jolteon"
——–set str[3] to "flareon"
——–set str[4] to "espeon"
——–set str[5] to "umbreon"
——–set str[6] to "leafeon"
——–set str[7] to "glaceon"
```

```
——–set str[8] to "sylveon"
——–for i = 1 to 8 inclusive
—————-if size of str[i] is n
————————set ck to 1
————————for j = 0 to n
——————————–if str[0][j] >= 'a' and str[0][j] <= 'z'
—————————————-if str[0][j] is not str[i][j]
————————————————set ck to 0
————————————————break
—————————————-end of the block
——————————–end of the block
————————end of the block
————————if ck is 1
——————————–print str[i]
——————————–return to indicate successful execution
————————end of the block
—————-end of the block
——–end of the block
——–print newline
——–return to indicate successful execution
end of the block
```

```
C++ Code
int main() {
——–string str[12];
——–int n, i, j, ck;
——–cin >>n >>str[0];
——–str[1] = "vaporeon";
——–str[2] = "jolteon";
——–str[3] = "flareon";
——–str[4] = "espeon";
——–str[5] = "umbreon";
——–str[6] = "leafeon";
——–str[7] = "glaceon";
——–str[8] = "sylveon";
——–for (i = 1; i <= 8; i++) {
—————-if (str[i].size() == n) {
————————ck = 1;
————————for (j = 0; j <n; j++) {
——————————–if (str[0][j] >= 'a' && str[0][j] <= 'z') {
—————————————-if (str[0][j] != str[i][j]) {
————————————————ck = 0;
————————————————break;
—————————————-}
——————————–}
————————}
————————if (ck == 1) {
——————————–cout <<str[i] <<endl;
——————————–return 0;
————————}
—————-}
——–}
——–cout <<endl;
——–return 0;
}
```

### <span id="page-27-0"></span>Example F.3: example data point of Path Traversal

#### **Problem Description**

In a hypothetical world, there are a number of cities. Each city has a one-way connection to only one other city via a specific transit method. The details of the cities are as follows: Vladivostok is a lively city. You can travel from Vladivostok to Guildford by ferry.

Kendal is a lively city. You can travel from Kendal to Fort McMurray by plane. Fort McMurray is a lively city. You can travel from Fort McMurray to Versailles by train. Astrakhan is a lively city. You can travel from Astrakhan to Akron by bus. Timisoara is a lively city. You can travel from Timisoara to Lafayette by train. Bolzano is a lively city. You can travel from Bolzano to Terrace by bus. Versailles is a lively city. You can travel from Versailles to West Valley City by ferry. Lille is a lively city. You can travel from Lille to Abingdon by ferry. Lafayette is a lively city. You can travel from Lafayette to Tucson by ferry. Reno is a lively city. You can travel from Reno to Lafayette by plane.

... ...

Kamloops is a lively city. You can travel from Kamloops to Fort Collins by ferry. Livorno is a lively city. You can travel from Livorno to Colorado Springs by train. Vladikavkaz is a lively city. You can travel from Vladikavkaz to Bromsgrove by plane. Medicine Hat is a lively city. You can travel from Medicine Hat to Tallinn by ferry. Sandhurst is a lively city. You can travel from Sandhurst to Erfurt by ferry. Tucson is a lively city. You can travel from Tucson to Gateshead by train. Chilliwack is a lively city. You can travel from Chilliwack to Naperville by bus. Sacramento is a lively city. You can travel from Sacramento to Kendal by ferry. Folkestone is a lively city. You can travel from Folkestone to Reno by ferry. Bournemouth is a lively city. You can travel from Bournemouth to Cornwall by plane.

Now find the route from Lille to Bromsgrove based on the information above.

#### **Target Route**

From Lille, take a ferry to Abingdon.

From Abingdon, take a bus to Augsburg.

From Augsburg, take a plane to Lecce.

From Lecce, take a plane to Vladivostok.

From Vladivostok, take a ferry to Guildford.

From Guildford, take a ferry to Gelsenkirchen.

From Gelsenkirchen, take a train to Kamloops.

From Kamloops, take a ferry to Fort Collins.

From Fort Collins, take a train to Basingstoke.

From Basingstoke, take a plane to Medicine Hat.

... ...

From Sacramento, take a ferry to Kendal.

From Kendal, take a plane to Fort McMurray.

From Fort McMurray, take a train to Versailles.

From Versailles, take a ferry to West Valley City.

From West Valley City, take a plane to Crewe.

From Crewe, take a plane to Worthing.

From Worthing, take a ferry to Bournemouth.

From Bournemouth, take a plane to Cornwall.

From Cornwall, take a ferry to Vladikavkaz.

From Vladikavkaz, take a plane to Bromsgrove.

### <span id="page-28-0"></span>Example F.4: example data point of Theory-of-Mind Tracking

#### **Story**

You'll see a story about object placement. Each story involves four components: Agents, Objects, Rooms, and Containers. Given a question about an (agent, object) pair, your task is to track the locations and beliefs in stories about object placement asked in the question.

Step 0: Leon is in the playroom; Carol is in the pantry; the band-aid is on the playroom's pedestal; the tweezers is on the playroom's pedestal.

Step 1: Leon moves to the pantry.

Step 2: Carol moves to the playroom.

Step 3: Leon moves to the playroom.

Step 4: Carol moves to the pantry.

Step 5: Leon moves the band-aid to the playroom's rack.

Step 6: Carol moves to the playroom.

Step 7: Leon moves to the pantry, and moves the tweezers to the pantry's pedestal.

Step 8: Carol moves to the pantry, and moves the band-aid to the pantry's pedestal.

Step 9: Leon moves the tweezers to the pantry's rack.

... ...

Step 30: Carol enters the playroom.

Step 31: Leon moves to the playroom, and moves the band-aid to the playroom's rack.

Step 32: Carol moves to the pantry, and moves the tweezers to the pantry's pedestal.

Step 33: Leon moves to the pantry.

Step 34: Carol moves to the playroom, and moves the tweezers to the playroom's pedestal.

Step 35: Leon moves to the playroom.

Step 36: Carol leaves the room he was in.

Step 37: Leon moves to the pantry, and moves the tweezers to the pantry's rack.

Step 38: Carol enters the playroom.

Step 39: Leon leaves the room he was in.

Step 40: Carol moves to the pantry, and moves the band-aid to the pantry's rack.

Where does Carol believe the tweezers is?

#### **Target Output**

Carol's belief on the tweezers:

- Step 0: (initial state) {Carol location: pantry; tweezers location: playroom's pedestal; Carol sees tweezers: False; Carol's belief on tweezers: None}
- Step 1: (Carol location unchanged; tweezers location unchanged) {Carol location: pantry; tweezers location: playroom's pedestal; Carol sees tweezers: False; Carol's belief on tweezers: None}
- Step 2: (Carol location changed; tweezers location unchanged) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: playroom's pedestal}
- Step 3: (Carol location unchanged; tweezers location unchanged) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: playroom's pedestal}
- Step 4: (Carol location changed; tweezers location unchanged) {Carol location: pantry; tweezers location: playroom's pedestal; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 5: (Carol location unchanged; tweezers location unchanged) {Carol location: pantry; tweezers location: playroom's pedestal; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 6: (Carol location changed; tweezers location unchanged) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: playroom's pedestal}
- Step 7: (Carol location unchanged; tweezers location changed) {Carol location: playroom; tweezers location: pantry's pedestal; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 8: (Carol location changed; tweezers location unchanged) {Carol location: pantry; tweezers location: pantry's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: pantry's pedestal}
- Step 9: (Carol location unchanged; tweezers location changed) {Carol location: pantry; tweezers location: pantry's rack; Carol sees tweezers: True; Carol's belief on tweezers: pantry's rack}

... ...

- Step 30: (Carol location changed; tweezers location unchanged) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: playroom's pedestal}
- Step 31: (Carol location unchanged; tweezers location unchanged) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: playroom's pedestal}
- Step 32: (Carol location changed; tweezers location changed) {Carol location: pantry; tweezers location: pantry's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: pantry's pedestal}
- Step 33: (Carol location unchanged; tweezers location unchanged) {Carol location: pantry; tweezers location: pantry's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: pantry's pedestal}
- Step 34: (Carol location changed; tweezers location changed) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers: playroom's pedestal}
- Step 35: (Carol location unchanged; tweezers location unchanged) {Carol location: playroom; tweezers location: playroom's pedestal; Carol sees tweezers: True; Carol's belief on tweezers:

### playroom's pedestal}

- Step 36: (Carol location changed; tweezers location unchanged) {Carol location: None; tweezers location: playroom's pedestal; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 37: (Carol location unchanged; tweezers location changed) {Carol location: None; tweezers location: pantry's rack; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 38: (Carol location changed; tweezers location unchanged) {Carol location: playroom; tweezers location: pantry's rack; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 39: (Carol location unchanged; tweezers location unchanged) {Carol location: playroom; tweezers location: pantry's rack; Carol sees tweezers: False; Carol's belief on tweezers: playroom's pedestal}
- Step 40: (Carol location changed; tweezers location unchanged) {Carol location: pantry; tweezers location: pantry's rack; Carol sees tweezers: True; Carol's belief on tweezers: pantry's rack} Final Answer: the pantry's rack

