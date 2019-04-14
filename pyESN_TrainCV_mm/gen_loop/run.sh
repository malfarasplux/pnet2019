## Apply config and run 
sed -i 's:^dataset =.*:dataset = "training":' ./genloopCV.py
sed -i 's:^scale_def.*:scale_def = 0.005   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

##################################################################
sed -i 's:^scale_def.*:scale_def = 0.010   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py

##################################################################
sed -i 's:^scale_def.*:scale_def = 0.015   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py


##################################################################
sed -i 's:^scale_def.*:scale_def = 0.020   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py


##################################################################
sed -i 's:^scale_def.*:scale_def = 0.025   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py


##################################################################
sed -i 's:^scale_def.*:scale_def = 0.050   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py


##################################################################
sed -i 's:^scale_def.*:scale_def = 0.075   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py


##################################################################
sed -i 's:^scale_def.*:scale_def = 0.100   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py


##################################################################
sed -i 's:^scale_def.*:scale_def = 0.500   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py

##################################################################
sed -i 's:^scale_def.*:scale_def = 1.000   # scaling:' ./genloopCV.py

## Loop over N
sed -i 's:^mem_def.*:mem_def = 0.00      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.01      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.02      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.03      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.04      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.05      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.06      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.07      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.08      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.09      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.10      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.11      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.12      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.14      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.15      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.16      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.17      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.18      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.19      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.20      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 0.50      # memory:' ./genloopCV.py
python3 ./genloopCV.py

sed -i 's:^mem_def.*:mem_def = 1.0      # memory:' ./genloopCV.py
python3 ./genloopCV.py

