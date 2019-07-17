# pnet2019 - Sepsis Challenge 2019 (PLUX Wireless Biosignals S.A. Lisboa)

[1. Challenge info](#site)  
[2. Data (only private access, after team's disclaimer)](#data)  
[3. Instructions](#instr)  
[4. Leaderboard](#lead)  
[5. TODO](#todo)    


##  1. Challenge info <a name="site"></a>
-  https://physionet.org/challenge/2019/  
-  http://cinc2019.org/

##  2. Data <a name="data"></a>
https://physionet.org/users/shared/challenge-2019/  
(only private access, after team's disclaimer)

##  3. Instructions <a name="instr"></a>
-  [Google Doc](https://docs.google.com/document/d/1-YCLmie2_1gM4FrpBaSfkhYt8xpYghs8l2vbPemODkw)  
-  https://github.com/physionetchallenges/python-example-2019
-  Use https://cloud.google.com

#  4. Leaderboard <a name="lead"></a>
https://physionet.org/challenge/2019/leaderboard/

##  5. TODO <a name="todo"></a>
-  <mark>Implement Custom StratifiedKFold (test needed)</mark>  
-  <mark>Assess sigmoid bias effect</mark>
-  <mark>Perform grid-search param optimization (N, scale, mem, exp)</mark>
-  <mark>Prepare [spreadsheet](https://docs.google.com/spreadsheets/d/1qoer2i_GP-9oS2-ZxLZC_7PBIqKRqS_byEtowJbXamg/edit#gid=0) (current approach + stats)</mark>  
-  ESN to RF feature[6:] train + classify  
-  Prepare optimisation heatmap  
-  Re-run cross database examples (with optimal values)   
-  Clean repo (stash, backup, rm folders)  
-  Run sample test in a Docker via GCloud  
-  Run utility function in Hosp A (B) results  
-  Create submission root repository +buddy  
-  Compare get_sepsis_patient function (speed, same results) ~done   
-  Alternative normalization

## Check this out  
ESN Time series https://towardsdatascience.com/predicting-stock-prices-with-echo-state-networks-f910809d23d4  

##### Approaches (Deprecated)
- Combined RandomForest + ESN classification
- hospA + hospB Cross Validation results  
- MinMax Normalised ESN
- N neurons study (40, 100, 200, ... 1000)
- Separate ESN generation and import as module
- Create bash sed modification files
- Establish a fix report structure
