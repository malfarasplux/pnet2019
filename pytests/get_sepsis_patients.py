## Get sepsis patients MODIFIED
def get_sepsis_patients(sepsis_label, patient):
    patient_sep = np.zeros(len(sepsis_label),dtype=np.int)
    for i in range(len(np.unique(patient))):
        i_pat = np.where(patient==i)[0]
        patient_sep[i_pat] = int(np.sum(sepsis_label[i_pat])>0)*np.ones(len(i_pat), dtype=np.int)
        
    patient_sep_idx = np.where(patient_sep!=0)[0]
    patient_healthy_idx = np.where(patient_sep==0)[0]
    return patient_sep, patient_sep_idx, patient_healthy
