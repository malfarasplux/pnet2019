import numpy as np
import matplotlib.pyplot as plt

N = 100
scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mem = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

val_key = {}
with open("./AUC_report.txt") as f: 
    for i, line in enumerate(f):      
        lineval = line.split()[0]        
        print ("line {0} = {1}".format(i, lineval)) 
        val_key[lineval.split(".txt:")[0][18:]] = float(lineval.split(".txt:")[1])


AUC_matrix = np.zeros((len(scale),len(mem)),dtype=np.float)
N_i = str(N)
for i in range(len(scale)):
    scale_i = '{:1.3f}'.format(scale[i])
    for j in range(len(mem)):
        mem_i =  '{:1.3f}'.format(mem[j])
        key_i = N_i + "_" + scale_i + "_" + mem_i
        AUC_matrix[i,j] = val_key[key_i]


fig, ax = plt.subplots()
im = ax.imshow(AUC_matrix)
ax.set_title("Grid search AUC opt")
ax.set_xticks(np.arange(len(mem))) 
ax.set_yticks(np.arange(len(scale)))
ax.set_xticklabels(mem) 
ax.set_yticklabels(scale) 
ax.set_xlabel('mem')
ax.set_ylabel('scale')
cbar = ax.figure.colorbar(im, ax=ax)


# Loop over data dimensions and create text annotations.
for i in range(len(scale)):
    for j in range(len(mem)):
        text = ax.text(j, i, AUC_matrix[i, j],
                       ha="center", va="center", color="w")
