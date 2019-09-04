# grep AUC report_ESNtrainCV_* > AUC_report.txt
import numpy as np
import matplotlib.pyplot as plt
show_val = False
save_png = True

## ESN parameters
N = 100                                     # Neurons
scale_def = np.linspace(0.0001, 0.025, 100)
mem_def = np.linspace(1.0, 10, 10)
scale = []
mem = []
for i in range(len(scale_def)):
    scale.append(('{:1.4f}'.format(scale_def[i])))
for i in range(len(mem_def)):
    mem.append(('{:1.4f}'.format(mem_def[i])))
    
exponent = 1.0                                # sigmoid exponent



val_key = {}
with open("./AUC_report.txt") as f: 
    for i, line in enumerate(f):      
        lineval = line.split()[0]        
        print ("line {0} = {1}".format(i, lineval)) 
        val_key[lineval.split(".txt:")[0][18:]] = float(lineval.split(".txt:")[1])


AUC_matrix = np.ones((len(scale),len(mem)),dtype=np.float)*np.nan
N_i = str(N)
for i in range(len(scale)):
    scale_i = scale[i]
    for j in range(len(mem)):
        mem_i =  mem[j]
        key_i = N_i + "_" + scale_i + "_" + mem_i
        try :
            AUC_matrix[i,j] = val_key[key_i]
        except:
            print(key_i, " missing")


apply_trans = True
fig, ax = plt.subplots()
ax.set_title("Grid search AUC opt")
if not apply_trans:
    im = ax.imshow(AUC_matrix)
    ax.set_xticks(np.arange(len(mem))) 
    ax.set_yticks(np.arange(len(scale)))
    ax.set_xticklabels(mem,  rotation='vertical') 
    ax.set_yticklabels(scale) 
    ax.set_xlabel('mem')
    ax.set_ylabel('scale')
    cbar = ax.figure.colorbar(im, ax=ax)

else:
    im = ax.imshow(AUC_matrix.T)
    ax.set_xticks(np.arange(len(scale)))
    ax.set_yticks(np.arange(len(mem))) 
    ax.set_xticklabels(scale) 
    ax.set_yticklabels(mem,  rotation='vertical') 
    ax.set_xlabel('scale')
    ax.set_ylabel('mem')
    cbar = ax.figure.colorbar(im, ax=ax)



    

# Loop over data dimensions and create text annotations.
if show_val:
    for i in range(len(scale)):
        for j in range(len(mem)):
            text = ax.text(j, i, AUC_matrix[i, j],
                           ha="center", va="center", color="w")

if save_png:
    fig.savefig('AUC.png')
