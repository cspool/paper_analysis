# [PSEUDOCODE]:

pseudocode

[CODE]:

"'

### <span id="page-44-0"></span>Prompt H.3: Template for the Path Traversal Task

#### [TASK]

In a completely hypothetical world, there are a number of cities. Each city has a one-way connection to only one other city via a specific transit method (bus, train, plane, or ferry). Your task is to provide a route from a city to another city. You should follow the specific instruction provided later and output the route following the format provided in the instruction.

### [IMPORTANT NOTES]

- All connections are one-way. If city A is connected to city B, you can travel from A to B, but not the other way around.
- Because each city is connected to only one other city, so there's only one possible route. To find the route, you can simply start from the starting city, identify the next city it's connected to, and repeat the process until you reach the destination city.
- Please follow the exact format specified below when outputting the route.

### [OUTPUT FORMAT]

Please mark the route with <Route>and </Route>tags. The route should be in the following format, where one line is one step of the route:

<Route>

From <CITY NAME>, take a <TRANSIT METHOD>to <CITY NAME>.

... From <CITY NAME>, take a <TRANSIT METHOD>to <CITY NAME>. </Route>

