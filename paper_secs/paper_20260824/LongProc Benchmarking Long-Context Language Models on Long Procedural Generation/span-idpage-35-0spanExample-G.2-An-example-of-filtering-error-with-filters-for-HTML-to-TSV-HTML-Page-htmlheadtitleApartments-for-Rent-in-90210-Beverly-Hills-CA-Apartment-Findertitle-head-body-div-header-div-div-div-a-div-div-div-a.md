# <span id="page-35-0"></span>Example G.2: An example of filtering error with filters for HTML to TSV **HTML Page** <html><head><title>Apartments for Rent in 90210, Beverly Hills, CA |Apartment Finder</title> </head> <body> <div> <header> <div/> <div> <div> <a> <div/> <div/> <div/> </a>

```
</div>
<div>
<nav>
<h2>
Header Navigation Links
</h2>
<ul>
<li>
<form>
<div>
<div role="combobox">
<label>Search label</label>
<input placeholder="City, Zip, Neighborhood, Address, Property Name" type="text"/>
<div/>
</div>
</div>
<div role="listbox"/>
<div/>
</form>
</li>
<li>
<a>
<span/>
</a>
</li>
<li>
... ...
<div>
<h2>
<a title="609 N Doheny Dr">
<span>609 N Doheny Dr</span>
</a>
</h2>
<address title="609 N Doheny Dr, Beverly Hills, CA 90210">
609 N Doheny Dr, Beverly Hills, CA 90210
</address>
<div>
<div>
<span>
$4,500
</span>
<span>
<span>|</span>
<span>
2 Beds
</span>
</span>
</div>
</div>
</div>
<div>
<button title="Send Message">
<span>Send Message</span>
<span>Email Property</span>
</button>
<span>|</span>
<a>
<span>Call Now</span>
<span>(657) 289-6985</span>
</a>
</div>
</div>
</article><article>
<div>
```

```
<div>
<button/>
<span>
<span>1</span>
/
<span>48</span>
</span>
<div>
<button role="button" type="button"/>
<button role="button" type="button"/>
... ...
</body>
... ...
</html>
Target Information
Based on the HTML webpage above about Real estate search, extract the following properties from
the items listed on the webpage: (1) Name; (2) Detailed address, including zip code; (3) Rent; (4)
Type of the household (number of beds/studio); (5) Phone;
Only extract the rows where the type of the household is strictly equal to "2 Beds".
Ground Truth TSV
609 N Doheny Dr 609 N Doheny Dr, Beverly Hills, CA 90210 $4,500 2 Beds (657) 289-6985 2 Wks.
Ago
237 S Doheny Dr 237 S Doheny Dr, Beverly Hills, CA 90211 $3,500 2 Beds (805) 833-6612 8 Hrs.
Ago
135 N Doheny Dr 135 N Doheny Dr, West Hollywood, CA 90048 $3,600 - $3,995 2 Beds (858) 704-
2956 10 Hrs. Ago
8871 Burton Way 8871 Burton Way, West Hollywood, CA 90048 $3,500 2 Beds (424) 566-8685 1 Day
Ago
140 S Crescent Dr 140 S Crescent Dr, Beverly Hills, CA 90212 $5,500 2 Beds (424) 380-6264 2 Days
Ago
137 N Wetherly Dr 137 N Wetherly Dr, Los Angeles, CA 90048 $3,100 - $3,200 2 Beds (562) 553-
7450 2 Days Ago
9005 Cynthia St 9005 Cynthia St, West Hollywood, CA 90069 $3,600 2 Beds (805) 833-6586 3 Days
Ago
310 S Almont Dr 310 S Almont Dr, Los Angeles, CA 90048 $2,950 2 Beds (626) 701-8056 3 Days Ago
148 S Maple Dr 148 S Maple Dr, Beverly Hills, CA 90212 $2,995 2 Beds (747) 788-4237 4 Days Ago
9061 Keith Ave 9061 Keith Ave, West Hollywood, CA 90069 $3,100 2 Beds (424) 380-5750 4 Days
Ago
200 S Spalding Dr 200 S Spalding Dr, Beverly Hills, CA 90212 $3,200 2 Beds (844) 653-6049 4 Days
Ago
462 N Almont Dr 462 N Almont Dr, West Hollywood, CA 90048 $3,600 2 Beds (657) 234-6681 6
Days Ago
131 N Wetherly Dr 131 N Wetherly Dr, West Hollywood, CA 90048 $5,150 2 Beds (657) 224-3710 1
Wk. Ago
9011 Rangely Ave 9011 Rangely Ave, West Hollywood, CA 90048 $5,000 2 Beds (661) 388-5353 1
Wk. Ago
Model Output TSV
447 N Oakhurst Dr 447 N Oakhurst Dr, Beverly Hills, CA 90210 $1,650 Studio (424) 313-1711 6
Days Ago
328 N Maple Dr 328 N Maple Dr, Beverly Hills, CA 90210 $4,300 - $7,000 1 - 2 Beds (760) 536-5804 6
Days Ago
609 N Doheny Dr 609 N Doheny Dr, Beverly Hills, CA 90210 $4,500 2 Beds (657) 289-6985 2 Wks.
Ago
237 S Doheny Dr 237 S Doheny Dr, Beverly Hills, CA 90211 $3,500 2 Beds (805) 833-6612 8 Hrs.
Ago
135 N Doheny Dr 135 N Doheny Dr, West Hollywood, CA 90048 $3,600 - $3,995 2 Beds (858) 704-
2956 10 Hrs. Ago
220 S Doheny Dr 220 S Doheny Dr, Beverly Hills, CA 90211 $2,295 - $2,695 1 - 2 Beds (763) 284-
5475 1 Day Ago
115 N Doheny Dr 115 N Doheny Dr, West Hollywood, CA 90048 $1,700 - $3,000 Studio - 2 Beds (562)
349-1380 1 Day Ago
153 S Palm Dr 153 S Palm Dr, Beverly Hills, CA 90212 $5,495 - $7,495 2 - 3 Beds (424) 424-0578 1
```

Day Ago

137 N Wetherly Dr 137 N Wetherly Dr, Los Angeles, CA 90048 \$3,100 - \$3,200 2 Beds (562) 553- 7450 2 Days Ago

9480 Charleville Blvd 9480 Charleville Blvd, Beverly Hills, CA 90212 \$3,500 - \$5,000 1 - 3 Beds (844) 900-0934 2 Days Ago

9005 Cynthia St 9005 Cynthia St, West Hollywood, CA 90069 \$3,600 2 Beds (805) 833-6586 3 Days Ago

310 S Almont Dr 310 S Almont Dr, Los Angeles, CA 90048 \$2,950 2 Beds (626) 701-8056 3 Days Ago 9061 Keith Ave 9061 Keith Ave, West Hollywood, CA 90069 \$3,100 2 Beds (424) 380-5750 4 Days Ago

200 S Spalding Dr 200 S Spalding Dr, Beverly Hills, CA 90212 \$3,200 2 Beds (844) 653-6049 4 Days Ago

9011 Rangely Ave 9011 Rangely Ave, West Hollywood, CA 90048 \$5,000 2 Beds (661) 388-5353 1 Wk. Ago

### **G.2 Theory-of-Mind Tracking**

Example [G.3](#page-38-0) shows an example of GPT-4o incorrectly tracking the location of objects after long range for a 2K example of the theory-of-mind tracking task. The model incorrectly identifies the location of bobby pin at step 28 and carries the error on to later steps.

### <span id="page-38-0"></span>Example G.3: An example of incorrectly updating search states for Countdown

#### **Story**

You'll see a story about object placement. Each story involves four components: Agents, Objects, Rooms, and Containers. Given a question about an (agent, object) pair, your task is to track the locations and beliefs in stories about object placement asked in the question.

Step 0: Kevin is in the living room; Amanda is in the craft room; the USB cable is on the living room's tv stand; the rubber band is on the living room's stool.

Step 1: Kevin moves to the craft room.

Step 2: Amanda moves to the living room.

Step 3: Amanda moves to the craft room, and moves the rubber band to the craft room's tv stand.

Step 4: Amanda moves the rubber band to the craft room's stool.

Step 5: Kevin moves the rubber band to the craft room's tv stand.

Step 6: Amanda moves to the living room, and moves the rubber band to the living room's tv stand.

Step 7: Kevin moves to the living room.

Step 8: Amanda moves the USB cable to the living room's stool.

Step 9: Kevin moves to the craft room.

Step 10: Amanda leaves the room he was in.

Step 11: Kevin moves to the living room.

Step 12: Amanda enters the craft room.

Step 13: Kevin leaves the room he was in.

Step 14: Amanda moves to the living room.

Step 15: Kevin enters the living room.

Step 16: Amanda moves to the craft room, and moves the USB cable to the craft room's stool.

Step 17: Kevin moves to the craft room, and moves the rubber band to the craft room's tv stand.

Step 18: Amanda moves to the living room, and moves the USB cable to the living room's tv stand.

Step 19: Kevin moves to the living room, and moves the rubber band to the living room's tv stand.

Step 20: Amanda moves to the craft room, and moves the USB cable to the craft room's stool.

Step 21: Kevin moves to the craft room.

Step 22: Amanda moves to the living room, and moves the USB cable to the living room's stool.

Step 23: Kevin moves to the living room.

Step 24: Amanda moves the USB cable to the living room's tv stand.

Step 25: Kevin moves the USB cable to the living room's stool.

Step 26: Amanda moves the rubber band to the living room's stool.

Step 27: Kevin moves the USB cable to the living room's tv stand.

Step 28: Amanda leaves the room he was in.

Step 29: Kevin leaves the room he was in.

Step 30: Amanda enters the living room.

Step 31: Kevin enters the craft room.

Step 32: Amanda leaves the room he was in.

- Step 33: Kevin moves to the living room.
- Step 34: Amanda enters the living room.
- Step 35: Kevin moves the rubber band to the living room's tv stand.
- Step 36: Amanda moves to the craft room.
- Step 37: Kevin moves to the craft room, and moves the USB cable to the craft room's tv stand.
- Step 38: Amanda moves to the living room.
- Step 39: Kevin moves to the living room, and moves the USB cable to the living room's stool.
- Step 40: Amanda leaves the room he was in.
- Where does Amanda believe the USB cable is?

#### **Ground Truth**

Gwendolyn's belief on the bobby pin:

- Step 0: (initial state) {Gwendolyn location: closet; bobby pin location: pantry's ledge; Gwendolyn sees bobby pin: False; Gwendolyn's belief on bobby pin: None}

... ...

- Step 27: (Gwendolyn location unchanged; bobby pin location unchanged) {Gwendolyn location: closet; bobby pin location: closet's ledge; Gwendolyn sees bobby pin: True; Gwendolyn's belief on bobby pin: closet's ledge}
- Step 28: (Gwendolyn location unchanged; bobby pin location unchanged) {Gwendolyn location: closet; bobby pin location: closet's ledge; Gwendolyn sees bobby pin: True; Gwendolyn's belief on bobby pin: closet's ledge}
- Step 29: (Gwendolyn location changed; bobby pin location unchanged) {Gwendolyn location: None; bobby pin location: closet's ledge; Gwendolyn sees bobby pin: False; Gwendolyn's belief on bobby pin: closet's ledge}

... ...

### **Model Output**

Gwendolyn's belief on the bobby pin:

- Step 0: (initial state) {Gwendolyn location: closet; bobby pin location: pantry's ledge; Gwendolyn sees bobby pin: False; Gwendolyn's belief on bobby pin: None}

... ...

- Step 27: (Gwendolyn location unchanged; bobby pin location unchanged) {Gwendolyn location: closet; bobby pin location: closet's ledge; Gwendolyn sees bobby pin: True; Gwendolyn's belief on bobby pin: closet's ledge}
- Step 28: (Gwendolyn location unchanged; bobby pin location changed) {Gwendolyn location: closet; bobby pin location: pantry's ledge; Gwendolyn sees bobby pin: False; Gwendolyn's belief on bobby pin: closet's ledge}
- Step 29: (Gwendolyn location changed; bobby pin location unchanged) {Gwendolyn location: None; bobby pin location: pantry's ledge; Gwendolyn sees bobby pin: False; Gwendolyn's belief on bobby pin: closet's ledge}

... ...

