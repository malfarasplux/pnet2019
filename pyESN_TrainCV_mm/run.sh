## Apply config and run 
cp ./source/* ./
sed -i 's:^dataset =.*:dataset = "training_setA":' ./ESNtrainCV.py
sed -i 's:^scale_def.*:scale_def = 0.001   # scaling:' ./ESNtrainCV.py
sed -i 's:^mem_def.*:mem_def = 0.13      # memory:' ./ESNtrainCV.py

## Loop over N
cp ./source/* ./
sed -i 's:^N_def.*:N_def = 40          # Neurons:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

cp ./source/* ./
sed -i 's:^N_def.*:N_def = 100         # Neurons:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

cp ./source/* ./
sed -i 's:^N_def.*:N_def = 200         # Neurons:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

cp ./source/* ./
sed -i 's:^N_def.*:N_def = 300         # Neurons:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

cp ./source/* ./
sed -i 's:^N_def.*:N_def = 400         # Neurons:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py

cp ./source/* ./
sed -i 's:^N_def.*:N_def = 500         # Neurons:' ./ESNtrainCV.py
python3 ./ESNtrainCV.py



