# pnet2019 - Sepsis Challenge 2019 (PLUX Wireless Biosignals S.A. Lisboa)
##  Overview (August 2019)<a name="overview"></a>
Ring Echo State Network (ESN) and machine learning tools. 
Goal: CinC/Physionet 2019 challenge on ICU Sepsis prediction. 
Check our open source paper presented in Singapore: 
[Ring-Topology Echo State Networks for ICU Sepsis Classification](https://doi.org/10.22489/CinC.2019.327)  
[Poster](https://drive.google.com/file/d/1EgfHBGUYj3hBWp2vRqQITn7lswkTzmvm/view?usp=sharing)    

Main outcome: 
[ESN tools Python module](/ESNtools)  
Reuse freely under this license terms.  

This is an ad-hoc Python ESN implementation with a simple ring topology. For more general approaches, please look for alternatives such as: 

[cknd/pyESN](https://github.com/cknd/pyESN)  
[anvien/pyESN](https://github.com/anvien/PyESN)   
[kalekiu/easyesn](https://github.com/kalekiu/easyesn)  

This work was supported by Marie Skłodowska Curie Actions ITN [AffecTech](http://affectech.org/) (ERC H2020 Project ID: 722022) and Fundacão para a Ciência e a Tecnologia (FCT, Portugal), Phd grant PD/BDE/150304/2019.

This is a collaboration between PLUX Wireless Biosignals S.A. (Lisbon), Universitat Jaume I (Castelló) and LIBPhys-UNL (Caparica).



[1. Challenge info](#site)  
[2. Data (only private access, after team's disclaimer)](#data)  
[3. Instructions](#instr)  
[4. Results](#results)  
[5. Processing and submission steps](#steps)    


##  1. Challenge info <a name="site"></a>
-  https://physionet.org/challenge/2019/  
-  http://cinc2019.org/
-  [Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019](https://doi.org/10.1097/ccm.0000000000004145)
-  [Preprint (colour + supplementary material)](https://physionet.org/content/challenge-2019/1.0.0/physionet_challenge_2019_ccm_manuscript.pdf)
-  [2019 Challenge results](#results)

##  2. Data <a name="data"></a>
https://archive.physionet.org/users/shared/challenge-2019/  
(public access under the Physionet terms)  

##  3. Submission instructions <a name="instr"></a>
-  [Google Doc](https://docs.google.com/document/d/1-YCLmie2_1gM4FrpBaSfkhYt8xpYghs8l2vbPemODkw)  
-  https://github.com/physionetchallenges/python-example-2019
-  Use https://cloud.google.com

##  4. Results <a name="results"></a>
[Teams / Results](https://docs.google.com/spreadsheets/d/1qt2SllYISP7LUFtxrhqpfJtV8ewcd7krsRGtrB-r_BA/edit#gid=0)  

##  5. Processing and submission steps <a name="steps"></a>
-  <mark>Checks (nanfill flag, new get_sepsis, comment get_sepsis before import platform))</mark>
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
  ### Other: 
  - Completed/deprecated:
    - Combined RandomForest + ESN classification
    - hospA + hospB Cross Validation results  
    - MinMax Normalised ESN
    - N neurons study (40, 100, 200, ... 1000)
    - Separate ESN generation and import as module
    - Create bash sed modification files
    - Establish a fix report structure
    
## Extra references:  
ESN Time series https://towardsdatascience.com/predicting-stock-prices-with-echo-state-networks-f910809d23d4  
[Automated real-time method for ventricular heartbeat classification](https://doi.org/10.1016/j.cmpb.2018.11.005)  
[ESN A Fast Machine Learning Model for ECG-Based Heartbeat Classification and Arrhythmia Detection](https://doi.org/10.3389/fphy.2019.00103)  


