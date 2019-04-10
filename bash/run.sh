## Apply config and run 
cp ./source/* ./
sed -i 's:^path =.*:path = "../training_1/":' ./ESNtrainCV.py
sed -i 's:^N_def.*:N_def = 40          # Neurons:' ./ESNtrainCV.py
sed -i 's:^scale_def.*:scale_def = 0.001   # scaling:' ./ESNtrainCV.py
sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

