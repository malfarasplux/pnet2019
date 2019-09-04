# grep AUC report_ESNtrainCV_* > AUC_report.txt
import numpy as np 
import matplotlib.pyplot as plt
show_val = False
save_png = True

## ESN parameters
N = 100                                     # Neurons
scale_def = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.025, 0.050, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0] #scaling
mem_def = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.025, 0.050, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]  #memory
exponent = 1.0                                # sigmoid exponent

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
        val_key[lineval.split(".txt:")[0][1:]] = float(lineval.split(".txt:")[1])


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

AUC_max = np.nanmax(AUC_matrix)
print("AUC max", AUC_max)

# Find max position

for key, val in list(val_key.items()):    # for name, age in dictionary.iteritems():  (for Python 2.x)
    if val == AUC_max:
        max_pos = key
print("max_pos", max_pos)


apply_trans = False
fig, ax = plt.subplots()
ax.set_title("Grid search AUC opt")
if not apply_trans:

    plt.set_cmap('hot')
    im = ax.imshow(AUC_matrix, interpolation = 'none', vmin= 0.7, vmax=0.8 )
    ax.set_xticks(np.arange(len(mem))) 
    ax.set_yticks(np.arange(len(scale)))
    ax.set_xticklabels(mem,  rotation='180') 
    ax.set_yticklabels(scale) 
    ax.set_xlabel('mem')
    ax.set_ylabel('scale')
    ax.tick_params(axis='x', rotation=60)
    cbar = ax.figure.colorbar(im, ax=ax)


else:
    plt.set_cmap('hot')
    im = ax.imshow(AUC_matrix.T, interpolation = 'none', vmin= 0.7, vmax=0.8 )
    ax.set_xticks(np.arange(len(scale)))
    ax.set_yticks(np.arange(len(mem))) 
    ax.set_xticklabels(scale) 
    ax.set_yticklabels(mem) 
    ax.set_xlabel('scale')
    ax.set_ylabel('mem')
    ax.tick_params(axis='x', rotation=60)
    cbar = ax.figure.colorbar(im, ax=ax)




    

# Loop over data dimensions and create text annotations.
if show_val:
    for i in range(len(scale)):
        for j in range(len(mem)):
            text = ax.text(j, i, AUC_matrix[i, j],
                           ha="center", va="center", color="w")

if save_png:
    fig.savefig('AUC.png')
