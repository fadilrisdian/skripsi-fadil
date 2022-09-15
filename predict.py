from operator import index
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

def prediksi(model, df, kelasdata):
    
    #Predict data herbal
    yhat_herbal_proba =  model.predict(df)
    # # round probabilities to class labels
    yhat_herbal = yhat_herbal_proba.round()
    yhat_herbal
    
    hasil_prediksi_proba = pd.DataFrame(yhat_herbal_proba, columns = kelasdata.columns)
    hasil_prediksi = pd.DataFrame(yhat_herbal, columns = kelasdata.columns)
    
    return [hasil_prediksi, hasil_prediksi_proba]


def testing_herbal(hasil_prediksi, hasil_prediksi_proba, data_herbal):
    # Isi kembali nama senyawa 
    hasil_prediksi['Senyawa']=data_herbal['Senyawa']
    hasil = pd.DataFrame(columns = ['Senyawa', 'List Protein', 'Probability'])
    hasil
    
    # Cari nama protein yang sesuai dengan hasil prediksi
    for i in range(len(hasil_prediksi)):
        #Array protein
        protein_list = []
        #Array Probabilitas
        proba_list = []
        for j in range(0,7):
            #apabila hasil prediksi senyawa bernilai 1 
            if hasil_prediksi.iloc[i,j]==1:
                #Cari nama senyawa yang sesuai
                hasil.loc[i,'Senyawa'] = hasil_prediksi['Senyawa'][i]
                #Cari nama protein yang sesuai
                protein_name = hasil_prediksi.columns[j]
                proba = hasil_prediksi_proba.iloc[i,j]
                #Simpan nama protein dan probabilitasnya
                protein_list.append(protein_name)
                proba_list.append(proba)
        #apabila hasil prediksi bernilai 0, maka Protein None
        if len(protein_list)==0:
            hasil.loc[i,'List Protein'] = None
        else:
            hasil.loc[i,'List Protein'] = protein_list
        hasil.loc[i,'Probability'] = proba_list
    #Hapus missing values
    hasil = hasil.dropna()
    #Hitung total protein yang berinteraksi
    hasil['Total'] = hasil_prediksi.sum(axis=1)[hasil_prediksi.sum(axis=1)>=1]
    #Urutkan berdasarkan banyak protein
    hasil.sort_values(by=['Total'], inplace=True)
    # Simpan ke file
    
    return hasil

#main 

df_pubchem = pd.read_csv('dataset/interim/com_fp_pubchem.csv')
kelasdata = pd.read_csv('dataset/interim/kelas_data.csv')
data_herbal = pd.read_csv('dataset/interim/data_herbal_siap.csv')

model_pubchem = load_model('models/sae_dnn_pubchem_tuned.h5')

df = df_pubchem
model = model_pubchem

prediksi = prediksi(model, df, kelasdata)
print(prediksi)

hasil_prediksi = prediksi[0]
hasil_prediksi_proba = prediksi[1]

testing_herbal = testing_herbal(hasil_prediksi, hasil_prediksi_proba, data_herbal)
print(testing_herbal)

#save csv file
testing_herbal.to_csv('dataset/processed/hasil_pubchem.csv', index=False)