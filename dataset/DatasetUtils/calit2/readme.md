# CalIt2 Building People Counts

https://archive.ics.uci.edu/dataset/156/calit2+building+people+counts
https://doi.org/10.24432/C5NG78

This data comes from the main door of the CalIt2 building at UCI.

Dataset Characteristics: Multivariate, Time-Series

Subject Area: Other

Associated Tasks: -

Feature Type: Categorical, Integer

Instances: 10080

Features: 
- value-0: out flow
- value-1: in flow

---

Observations come from 2 data streams (people flow in and out of the building),  over 15 weeks, 48 time slices per day (half hour count aggregates). 

The purpose is to predict the presence of an event such as a conference in the building that is reflected by unusually high people counts for that day/time period.

---

1.  Flow ID: 7 is out flow, 9 is in flow
2.  Date: MM/DD/YY
3.  Time: HH:MM:SS
4.  Count: Number of counts reported for the previous half hour
  
Rows: Each half hour time slice is represented by 2 rows: one row for the out flow during that time period (ID=7) and one row for the in flow during that time period (ID=9)

Attributes in .events file ("ground truth")
1.  Date: MM/DD/YY
2.  Begin event time: HH:MM:SS (military) 
3.  End event time: HH:MM:SS (military)
4.  Event name (anonymized)
