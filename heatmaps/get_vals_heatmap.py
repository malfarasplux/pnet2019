import numpy as np
# import matplotlib.pyplot as plt

N = 200
scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mem = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

val_key = {}
with open("./F1_report.txt") as f: 
    for i, line in enumerate(f):      
        lineval = line.split()[0]        
        print ("line {0} = {1}".format(i, lineval)) 
        val_key[lineval.split(".txt:")[0][18:]] = float(lineval.split(".txt:")[1])


F1_matrix = np.zeros((len(scale),len(mem)),dtype=np.float)
N_i = str(N)
for i in range(len(scale)):
    scale_i = '{:1.3f}'.format(scale[i])
    for j in range(len(mem)):
        mem_i =  '{:1.3f}'.format(mem[j])
        key_i = N_i + "_" + scale_i + "_" + mem_i
        F1_matrix[i,j] = val_key[key_i]


fig, ax = plt.subplots()
im = ax.imshow(F1_matrix)
ax.set_title("Grid search F1 opt")
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
        text = ax.text(j, i, F1_matrix[i, j],
                       ha="center", va="center", color="w")
