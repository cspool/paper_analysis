# <span id="page-30-0"></span>Example F.5: example data point of Countdown

#### **Problem**

You will be given four numbers and a target number, your task is to find a way to use all four numbers exactly once, along with the basic operations (+, -, \*, /), to reach the target number. Numbers: [44, 48, 35, 6]

Target: 25

#### **Solving Procedure**

Initial number set: [44, 48, 35, 6], target: 25. Options for choosing two numbers: [(44, 48), (44, 35), (44, 6), (48, 35), (48, 6), (35, 6)].

- –|- Pick two numbers (44, 48) (numbers left: [35, 6]). Try possible operations.
- —-|- Try 48 + 44 = 92. Add 92 to the number set. Current number set: [92, 35, 6], target: 25. Options for choosing two numbers: [(92, 35), (92, 6), (35, 6)].
- ——|- Pick two numbers (92, 35) (numbers left: [6]). Try possible operations.
- ——–|- Try 92 + 35 = 127. Add 127 to the number set. Current number set: [127, 6], target: 25, just two numbers left.
- ———-|- Try 127 + 6 = 133. Evaluate 133 != 25, drop this branch.
- ———-|- Try 127 6 = 121. Evaluate 121 != 25, drop this branch.
- ———-|- Try 127 \* 6 = 762. Evaluate 762 != 25, drop this branch.
- ———-|- Try 127 / 6 = 21.2. 21.2 is a decimal, drop this branch.
- ——–|- Try 92 35 = 57. Add 57 to the number set. Current number set: [57, 6], target: 25, just two numbers left.
- ———-|- Try 57 + 6 = 63. Evaluate 63 != 25, drop this branch.
- ———-|- Try 57 6 = 51. Evaluate 51 != 25, drop this branch.
- ———-|- Try 57 \* 6 = 342. Evaluate 342 != 25, drop this branch.
- ———-|- Try 57 / 6 = 9.5. 9.5 is a decimal, drop this branch.
- ——–|- Try 92 \* 35 = 3220. 3220 exceeds the maximum intermediate result, drop this branch.
- ——–|- Try 92 / 35 = 2.6. 2.6 is a decimal, drop this branch.
- ——|- Pick two numbers (92, 6) (numbers left: [35]). Try possible operations.
- ——–|- Try 92 + 6 = 98. Add 98 to the number set. Current number set: [98, 35], target: 25, just two numbers left.
- ———-|- Try 98 + 35 = 133. Evaluate 133 != 25, drop this branch.
- ———-|- Try 98 35 = 63. Evaluate 63 != 25, drop this branch.
- ———-|- Try 98 \* 35 = 3430. 3430 exceeds the maximum intermediate result, drop this branch.
- ———-|- Try 98 / 35 = 2.8. 2.8 is a decimal, drop this branch.
- ——–|- Try 92 6 = 86. Add 86 to the number set. Current number set: [86, 35], target: 25, just two numbers left.
- ———-|- Try 86 + 35 = 121. Evaluate 121 != 25, drop this branch.
- ———-|- Try 86 35 = 51. Evaluate 51 != 25, drop this branch.
- ———-|- Try 86 \* 35 = 3010. 3010 exceeds the maximum intermediate result, drop this branch.
- ———-|- Try 86 / 35 = 2.5. 2.5 is a decimal, drop this branch.
- ——–|- Try 92 \* 6 = 552. Add 552 to the number set. Current number set: [552, 35], target: 25, just two numbers left.
- ———-|- Try 552 + 35 = 587. Evaluate 587 != 25, drop this branch.

```
———-|- Try 552 - 35 = 517. Evaluate 517 != 25, drop this branch.
———-|- Try 552 * 35 = 19320. 19320 exceeds the maximum intermediate result, drop this branch.
———-|- Try 552 / 35 = 15.8. 15.8 is a decimal, drop this branch.
——–|- Try 92 / 6 = 15.3. 15.3 is a decimal, drop this branch.
——|- Pick two numbers (35, 6) (numbers left: [92]). Try possible operations.
——–|- Try 35 + 6 = 41. Add 41 to the number set. Current number set: [41, 92], target: 25, just two
numbers left.
———-|- Try 92 + 41 = 133. Evaluate 133 != 25, drop this branch.
———-|- Try 92 - 41 = 51. Evaluate 51 != 25, drop this branch.
———-|- Try 92 * 41 = 3772. 3772 exceeds the maximum intermediate result, drop this branch.
———-|- Try 92 / 41 = 2.2. 2.2 is a decimal, drop this branch.
——–|- Try 35 - 6 = 29. Add 29 to the number set. Current number set: [29, 92], target: 25, just two
numbers left.
———-|- Try 92 + 29 = 121. Evaluate 121 != 25, drop this branch.
———-|- Try 92 - 29 = 63. Evaluate 63 != 25, drop this branch.
———-|- Try 92 * 29 = 2668. 2668 exceeds the maximum intermediate result, drop this branch.
———-|- Try 92 / 29 = 3.2. 3.2 is a decimal, drop this branch.
——–|- Try 35 * 6 = 210. Add 210 to the number set. Current number set: [210, 92], target: 25, just
two numbers left.
———-|- Try 210 + 92 = 302. Evaluate 302 != 25, drop this branch.
———-|- Try 210 - 92 = 118. Evaluate 118 != 25, drop this branch.
———-|- Try 210 * 92 = 19320. 19320 exceeds the maximum intermediate result, drop this branch.
———-|- Try 210 / 92 = 2.3. 2.3 is a decimal, drop this branch.
——–|- Try 35 / 6 = 5.8. 5.8 is a decimal, drop this branch.
—-|- Try 48 - 44 = 4. Add 4 to the number set. Current number set: [4, 35, 6], target: 25. Options for
choosing two numbers: [(4, 35), (4, 6), (35, 6)].
——|- Pick two numbers (4, 35) (numbers left: [6]). Try possible operations.
——–|- Try 35 + 4 = 39. Add 39 to the number set. Current number set: [39, 6], target: 25, just two
numbers left.
———-|- Try 39 + 6 = 45. Evaluate 45 != 25, drop this branch.
———-|- Try 39 - 6 = 33. Evaluate 33 != 25, drop this branch.
———-|- Try 39 * 6 = 234. Evaluate 234 != 25, drop this branch.
———-|- Try 39 / 6 = 6.5. 6.5 is a decimal, drop this branch.
——–|- Try 35 - 4 = 31. Add 31 to the number set. Current number set: [31, 6], target: 25, just two
numbers left.
———-|- Try 31 + 6 = 37. Evaluate 37 != 25, drop this branch.
———-|- Try 31 - 6 = 25. Evaluate 25 == 25, target found!
Now we have found the target, let's trace back the solution:
Final step: 31 - 6 = 25
The step before: 35 - 4 = 31
The first step: 48 - 44 = 4
Output the solution in the required format:
<Solution>
48 - 44 = 4
35 - 4 = 31
31 - 6 = 25
</Solution>
```

### Example F.6: example data point of Travel Planning

### **Problem**

You plan to visit 5 European cities for 20 days in total. You only take direct flights to commute between cities. You want to spend 7 days in Hamburg. You would like to visit Munich for 6 days. You want to spend 2 days in Manchester. You plan to visit relatives in Manchester between day 19 and day 20. You plan to stay in Lyon for 2 days. From day 13 to day 14, there is a annual show you want to attend in Lyon. You would like to visit Split for 7 days.

Here are the cities that have direct flights:

from Split to Munich, from Munich to Manchester, from Hamburg to Manchester, from Hamburg to Munich, from Split to Lyon, from Lyon to Munich, from Hamburg to Split, from Manchester to Split.

Find a trip plan of visiting the cities for 20 days by taking direct flights to commute between them.

### **Solving Procedure**

Read the requirements and identify the cities that have fixed schedules and the cities that need to be arranged.

- \* City: Hamburg, Duration: 7 days.
- \* City: Munich, Duration: 6 days.
- \* City: Manchester, Duration: 2 days, Fixed Schedule: Day 19 20.
- \* City: Lyon, Duration: 2 days, Fixed Schedule: Day 13 14.
- \* City: Split, Duration: 7 days.

Cities that have fixed schedules (sorted by their arrival days):

- \* City: Lyon, Fixed Schedule: Day 13 14.
- \* City: Manchester, Fixed Schedule: Day 19 20.

### Cities needing arrangement:

- \* City: Hamburg, Duration: 7 days.
- \* City: Munich, Duration: 6 days.
- \* City: Split, Duration: 7 days.

Current day: 1. Current plan: [].

Check whether the city with an arrival day of Day 1 - is fixed.

No. Consider possible options from cities needing arrangement: [Hamburg, Munich, Split] and explore these options in order.

- –|- Try arranging to visit Hamburg from Day 1. Duration: 7 days. Schedule: Day 1 7.
- –|- Check for direct flight from the starting point to Hamburg.
- –|- Yes.
- –|- Check whether this arrangement is compatible with the next fixed schedule after Day 1: Lyon (Day 13 - 14).
- –|- The departure day of Hamburg is Day 7. The arrival day of Lyon is Day 13. Day 7 is not later than (<=) Day 13. This arrangement is compatible.
- –|- This arrangement is feasible for now. Continue to arrange the rest of the plan.
- —-|- Current day: 7. Current plan: [Hamburg].
- —-|- Check whether the city with an arrival day of Day 7 is fixed.
- —-|- No. Consider possible options from cities needing arrangement: [Munich, Split] and explore these options in order.
- ——|- Try arranging to visit Munich from Day 7. Duration: 6 days. Schedule: Day 7 12.
- ——|- Check for direct flight from Hamburg to Munich.
- ——|- Yes.
- ——|- Check whether this arrangement is compatible with the next fixed schedule after Day 7: Lyon (Day 13 - 14).
- ——|- The departure day of Munich is Day 12. The arrival day of Lyon is Day 13. Day 12 is not later than (<=) Day 13. This arrangement is compatible.
- ——|- This arrangement is feasible for now. Continue to arrange the rest of the plan.
- ——–|- Current day: 12. Current plan: [Hamburg, Munich].
- ——–|- Check whether the city with an arrival day of Day 12 is fixed.
- ——–|- No. Consider possible options from cities needing arrangement: [Split] and explore these options in order.
- ———-|- Try arranging to visit Split from Day 12. Duration: 7 days. Schedule: Day 12 18.
- ———-|- Check for direct flight from Munich to Split.
- ———-|- No. Drop this branch.
- ———-|- Fail to arrange any option on day 12 in the current arrangement. Drop this branch.
- ——|- Try arranging to visit Split from Day 7. Duration: 7 days. Schedule: Day 7 13.
- ——|- Check for direct flight from Hamburg to Split.
- ——|- Yes.
- ——|- Check whether this arrangement is compatible with the next fixed schedule after Day 7: Lyon (Day 13 - 14).
- ——|- The departure day of Split is Day 13. The arrival day of Lyon is Day 13. Day 13 is not later than (<=) Day 13. This arrangement is compatible.
- ——|- This arrangement is feasible for now. Continue to arrange the rest of the plan.

```
——–|- Current day: 13. Current plan: [Hamburg, Split].
——–|- Check whether the city with an arrival day of Day 13 - is fixed.
——–|- Yes. The city with an arrival day of Day 13 - is fixed: Lyon.
———-|- Try arranging to visit Lyon from Day 13. Duration: 2 days. Schedule: Day 13 - 14.
———-|- Check for direct flight from Split to Lyon.
———-|- Yes.
———-|- Check whether this arrangement is compatible with the next fixed schedule after Day 13:
Manchester (Day 19 - 20).
———-|- The departure day of Lyon is Day 14. The arrival day of Manchester is Day 19. Day 14 is
not later than (<=) Day 19. This arrangement is compatible.
———-|- This arrangement is feasible for now. Continue to arrange the rest of the plan.
————|- Current day: 14. Current plan: [Hamburg, Split, Lyon].
————|- Check whether the city with an arrival day of Day 14 - is fixed.
————|- No. Consider possible options from cities needing arrangement: [Munich] and explore
these options in order.
————–|- Try arranging to visit Munich from Day 14. Duration: 6 days. Schedule: Day 14 - 19.
————–|- Check for direct flight from Lyon to Munich.
————–|- Yes.
————–|- Check whether this arrangement is compatible with the next fixed schedule after Day
14: Manchester (Day 19 - 20).
————–|- The departure day of Munich is Day 19. The arrival day of Manchester is Day 19. Day
19 is not later than (<=) Day 19. This arrangement is compatible.
————–|- This arrangement is feasible for now. Continue to arrange the rest of the plan.
—————-|- Current day: 19. Current plan: [Hamburg, Split, Lyon, Munich].
—————-|- Check whether the city with an arrival day of Day 19 - is fixed.
—————-|- Yes. The city with an arrival day of Day 19 - is fixed: Manchester.
——————|- Try arranging to visit Manchester from Day 19. Duration: 2 days. Schedule: Day 19 -
20.
——————|- Check for direct flight from Munich to Manchester.
——————|- Yes.
——————|- Check whether this arrangement is compatible with the next fixed schedule after
Day 19: None.
——————|- No following fixed schedules. This arrangement is compatible.
——————|- This arrangement is feasible for now. Continue to arrange the rest of the plan.
——————–|- Current day: 20. Current plan: [Hamburg, Split, Lyon, Munich, Manchester].
——————–|- All 5 cities are arranged. Complete plan is found!
Output the plan in the required format:
<Plan>
**Day 1-7:** Arriving in Hamburg and visit Hamburg for 7 days.
**Day 7:** Fly from Hamburg to Split.
**Day 7-13:** Visit Split for 7 days.
**Day 13:** Fly from Split to Lyon.
**Day 13-14:** Visit Lyon for 2 days.
**Day 14:** Fly from Lyon to Munich.
**Day 14-19:** Visit Munich for 6 days.
**Day 19:** Fly from Munich to Manchester.
**Day 19-20:** Visit Manchester for 2 days.
</Plan>
```

