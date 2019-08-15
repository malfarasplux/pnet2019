## Apply config and run 
## Loop over scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.1     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.2     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.3     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.4     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.5     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.6     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.7     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.8     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 0.9     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

########################################################################
## Change scale
cp ./source/*.py ./
sed -i 's:^scale_def.*:scale_def = 1.0     # scaling:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py


## Loop over mem
sed -i 's:^mem_def.*:mem_def = 0.1         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.2         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.3         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.4         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.5         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.6         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.7         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.8         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 0.9         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

sed -i 's:^mem_def.*:mem_def = 1.0         # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

