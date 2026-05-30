# B Implementation Details

#### <span id="page-11-0"></span>B.1 FEVoRI

```
1 def execute_command ( video , annotation , possible_answers , query ):
2 # Trope : Big Bad
3 # Definition : The character who is the direct cause of all of the bad happenings in a story .
4 # Thought Process :
5 # 1. Frame Selection : Analyze each frame to identify key characters and their actions .
6 # 2. Character Analysis : Identify the main antagonist and their actions throughout the video .
7 # 3. Answer Selection : Determine if there is a single character causing most of the negative events .
8
9 video_segment = VideoSegment ( video , annotation )
10 info = {
11 " character_actions ": {} ,
12 " negative_impacts ": {}
13 }
14 for i , frame in enumerate ( video_segment . frame_iterator ()):
15 # Identify all characters in the frame
16 for character in frame . find (" person "):
17 character_id = video_segment . face_identify ( character )
18 if character_id is None :
19 continue
20 # Query the action of the character in the frame
21 action_query = frame . simple_query (" What is this person doing ?")
22 # Check if the action has a negative impact
23 negative_query = f" Does the action '{ action_query }' have a negative impact ?"
24 has_negative_impact = frame . llm_query ( negative_query , to_yesno = True )
25 # Store character actions and their impacts
26 if character_id not in info [" character_actions "]:
27 info [" character_actions "][ character_id ] = []
28 info [" character_actions "][ character_id ]. append ( action_query )
29 if " yes " in has_negative_impact . lower ():
30 if character_id not in info [" negative_impacts "]:
31 info [" negative_impacts "][ character_id ] = 0
32 info [" negative_impacts "][ character_id ] += 1
34 # After collecting information , use it to determine the presence of the trope
35 answer , reason = video_segment . select_answer ( info , query , possible_answers )
36 return answer , reason , info
```

Listing 1: FEVoRI ICL Example

We have integrated *face\_identify*, which utilizes DeepFace [\[39\]](#page-10-10) to assign a unique ID to each character. As shown in Line 17, FEVoRI enhances fine-grained tokenization, extending beyond the generic object "human" to more effectively address Abstract Perception.

#### <span id="page-11-1"></span>B.2 ConQueR

```
1 def execute_command ( video , annotation , possible_answers , query ) - >[str , str , dict ]:
2 # Trope : Big Bad
3 # Definition : The character who is the direct cause of all of the bad happenings in a story .
4 # Thought Process :
5 # 1. Character Identification : Identify characters and track their actions across frames .
6 # 2. Event Linking : Determine which negative events are directly caused by the actions of a character .
7 # 3. Consistency Check : Check for consistency in the character 's negative influence over the story arc .
8 video_segment = VideoSegment ( video , annotation )
9 # Initialize a dictionary to store information collected during analysis
10 info = {
11 " happened bad events ": {} ,
12 " character infos ": {}
13 }
14 for i , frame in enumerate ( video_segment . frame_iterator ()):
15 for person in frame . find (" person "):
16 # identify the person in the frame
17 person_id = video_segment . face_identify ( person )
18 if person_id is None :
19 continue
20 # query the character "s description and add into character_description
21 if person_id not in info [" character infos "]:
22 descriptino_query = " Please describe his / her appearance in 10 words "
23 character_description = person . simple_query ( descriptino_query )
24 info [" character infos "][ person_id ] = {
25 " description ": character_description ,
26 " actions ": {}
27 }
28 # query the character "s action in the frame
29 action = person . simple_query (" Please describe his / her action in the scene ")
30 info [" character infos "][ person_id ][" actions "][ f"{i} frame "] = action
31 # check if there is any negative event happening in the scene
32 check_negative_query = "Is there any negative event happening in the scene ?"
33 any_negative_event = frame . simple_query ( check_negative_query , to_yesno = True )
34 if " yes " in any_negative_event . lower ():
```

```
35 # query the negative events happening in the scene
36 event = frame . simple_query (" What 's happening in the scene ")
37 info [" happened bad events "][ f"{i} frame "] = {
38 " event ": event ,
39 " potential cause ": []
40 }
41 for pid , character_infos in info [" character infos "]. items ():
42 # check if the character is a potential cause of the negative event
43 character_description = character_infos [" description "]
44 for prev_i in range (i , max(i -5 , 0) , -1):
45 prev_action = character_infos [" actions "]. get (f"{ prev_i } frame ", None )
46 if prev_action is not None :
47 person_query = f"Is person with '{ character_description } ' a potential cause of '{ event } '?"
48 is_person_potential = frame . simple_query ( person_query , to_yesno = True )
49 action_query = f"Is action '{ prev_action } ' a potential cause of '{ event } '?"
50 is_action_potential = frame . simple_query ( action_query , to_yesno = True )
51 if " yes " in is_person_potential . lower () or "yes " in is_action_potential :
52 info [" happened bad events "][ f"{i} frame "][" potential cause "]. append ( pid )
53 break
54 # After collecting information , use it to determine the presence of the trope
55 answer , reason = video_segment . select_answer ( info , query , possible_answers )
56 return answer , reason , info
```

Listing 2: ConQueR ICL Example

ConQueR enhances the model's ability to tackle Long-range Compositional Reasoning by decomposing the movie narrative (context) and the trope (query). In this instance, ConQueR systematically breaks down the identified characters and actions to align with the "Big Bad" trope query, as demonstrated in Lines 33, 36, 48, and 50.

### <span id="page-12-0"></span>B.3 ABCD

We utilized all generated code from TiM and sampled 512 codes from NExT-QA [\[10\]](#page-8-4), OKVQA [\[18\]](#page-9-6), and GQA [\[17\]](#page-9-5). We constructed AST trees using the Python AST module and excluded codes that could not be parsed by AST (less than 3%) from our analysis. For VLM token analysis, we used NLTK's word\_tokenize to split the VLM queries into tokens. The implementation details can be found in the repository.