import pickle
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import time
import math

def predict_(model, df):
    predictions_data = model.predict(df)
    return predictions_data[0]

def find_similar_cluster(df,cls):
    return df[df.cluster_kmeans == 0].head().drop(columns=['cluster_kmeans','trans_publik'])
    
def find_similar_price(df,harga):
    temp = df.drop(columns=['cluster_kmeans'])
    arr = abs(df.head().harga - harga)
    temp.insert(0, "cek_harga", arr, True)
    return temp[temp.cek_harga <= 1000000000].drop(columns=['cek_harga','trans_publik'])
    #return arr
    #return int(harga)
    
st.title("Perkiraan Wajar Harga Objek PBB")
st.caption("Aplikasi ini bertujuan memberikaan perkiraan harga wajar dari objek bumi dan bangunan dengan menggunakan data iklan yang ada di marketplace")

loaded_model = pickle.load(open('model.pkl', 'rb'))
loaded_model_cl = pickle.load(open('cluster.pkl', 'rb'))

df =  pd.read_csv('data_train.csv', sep=',')
    
with st.sidebar:
    st.write('Silahkan masukkan informasi berikut')
    with st.form('Form1'):
        c1,c2 = st.columns(2)
        lok_lat = c1.text_input('Latitude', -6.243)                
        lok_lon = c2.text_input('Longitude', 106.8)                
        lt = st.text_input('Luas Tanah (m2)', 100)
        lb = st.text_input('Luas Bangunan (m2)', 100)
        j_lt = st.slider(label = 'Jumlah Lantai', min_value = 1.0,
                        max_value = 4.0 ,
                        value = 1.0,
                        step = 1.0)
        j_km = st.slider(label = 'Jumlah Kamar Mandi', min_value = 1.0,
                max_value = 6.0 ,
                value = 1.0,
                step = 1.0)                
        j_kt = st.slider(label = 'Jumlah Kamar Tidur', min_value = 1.0,
                max_value = 6.0 ,
                value = 1.0,
                step = 1.0)     
        tipe = st.selectbox("Tipe Properti: ",['Rumah', 'Apartemen'])
        sertf = st.selectbox("Jenis Sertifikat: ",
                     ['SHM', 'HGB','Lainnya (PPJB, Girik, Adat, dll)'])             
        trans_publik = st.checkbox("Dekat Transportasi Publik (Stasiun, Krl, Mrt, Halte)")
        univ = st.checkbox("Dekat Universitas")
        sekolah = st.checkbox("Dekat Sekolah")
        tol = st.checkbox("Dekat Tol")
        bandara = st.checkbox("Dekat Bandara")
        rs = st.checkbox("Dekat Rumah Sakit")
        mall = st.checkbox("Dekat Mall")
        cluster = st.checkbox("Di dalam cluster")
        banjir = st.checkbox("Bebas banjir")
        garasi_carport = st.checkbox("Ada garasi/carport")
        kamar_pembantu = st.checkbox("Ada kamar pembantu")
        
        submit = st.form_submit_button('Predict')

placeholder = st.empty()

if submit:
    placeholder.empty()
    st.spinner()    
    with st.spinner(text='In progress'):
        time.sleep(1)
        tipe = 1 if(tipe == 'Rumah') else 0
        sertf = 1 if(sertf == 'SHM') else 2 if (sertf == 'HGB')  else 3
        features = {
        'trans_publik':trans_publik, 'luas_tanah':lt, 'luas_bangunan':lb, 'lok_lat':lok_lat,'lok_lon':lok_lon,
        'universitas':univ, 'sekolah':sekolah,'tol':tol, 'bandara':bandara, 'bebas_banjir':banjir, 'mall':mall,  
          'garasi_carport':garasi_carport,
           'kamar_pembantu':kamar_pembantu, 'cluster':cluster, 'rmh_sakit':rs, 
           'label_properti':tipe,
           'label_sertifikat':sertf, 'kmr_mandi':j_km, 'kmr_tidur':j_kt
        }
        harga = predict_(loaded_model,pd.DataFrame([features]))
        #placeholder.success(f'Perkiraan Harga Objek: Rp {"{:,.0f}".format(harga)}')
        #placeholder.success(f'Perkiraan Harga Objek: Rp :'.format(harga))
        features['harga'] = harga
        cls = predict_(loaded_model_cl,pd.DataFrame([features]))
        st.text("perbandingan objek PBB lain dengan harga yang mirip")
        st.write(find_similar_price(df,harga))        
        st.text("perbandingan objek PBB lain (metode clustering)")
        st.write(find_similar_cluster(df,pd.DataFrame([features])))


       # print('{} is {} years old'.format(name, age))

        
    


