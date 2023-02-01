from pandas.core.indexing import _iLocIndexer
from datetime import datetime
import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split #data 나누기
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

################################################################
def user_setting_plt():
    st.title('Results page')
    st.text('\n')
    st.text('Please use it after learning the data.')
    st.text('\n')
    st.text('\n')

    st.text('Please select a file to test')
    user_file = st.file_uploader('Upload CSV File',type=['csv'])
    user_setting = pd.read_csv(user_file)  
    results = pd.read_csv('User_DBYHS_SPCHCKN.csv')

    user_setting['DBYHS_SPCHCKN'] = results['DBYHS_SPCHCKN']
    st.dataframe(user_setting)
    user_setting['ID'] = range(len(user_setting))

    results = user_setting['DBYHS_SPCHCKN']
    

    No_1=list(user_setting[user_setting['DBYHS_SPCHCKN'] == 1].index) #세균갈색무늬병
    No_2=list(user_setting[user_setting['DBYHS_SPCHCKN'] == 2].index) #푸른곰팡이병
    No_3=list(user_setting[user_setting['DBYHS_SPCHCKN'] == 3].index) #솜털곰팡이병
    No_4=list(user_setting[user_setting['DBYHS_SPCHCKN'] == 4].index) #세균성검은썩음병
    No_5=list(user_setting[user_setting['DBYHS_SPCHCKN'] == 5].index) #흰곰팡이병
    total_len=len(user_setting)


    st.text('The results of your environmental element learning are as follows.')
    st.text('Probability of getting Marssonina blotch')
    st.text(len(No_1)/total_len)

    st.text('Probability of getting Penicillium italicum')
    st.text(len(No_2)/total_len)

    st.text('Probability of getting Peronospora farinosa')
    st.text(len(No_3)/total_len)

    st.text('Probability of getting Xanthomonas campestris')
    st.text(len(No_4)/total_len)

    st.text('Probability of getting Powdery mildew')
    st.text(len(No_5)/total_len)
    st.text('\n')



    col_list = ['WIND_SPEED', 'AIR_VELOCITY', 'TEMPERATURE', 'HUMIDITY','ILLUMINATION_INTENSITY', 'CARBON_DIOXIDE']
    col_list_rnt = st.radio('Please select a list of environmental elements to check', col_list)

    if st.button('select'):
        if col_list_rnt == 'WIND_SPEED':
            fig = plt.figure(figsize=(20, 10))
            plt.xlabel('ID')
            plt.title('Graph of WIND_SPEED')    
            plt.ylabel('WIND_SPEED')
            x=user_setting['ID']
            y = user_setting['WIND_SPEED']
            plt.plot(x,y)
            st.pyplot(fig)

        elif col_list_rnt == 'AIR_VELOCITY':
            fig = plt.figure(figsize=(20, 10))
            plt.xlabel('ID')
            plt.title('Graph of AIR_VELOCITY')    
            plt.ylabel('AIR_VELOCITY')
            x=user_setting['ID']
            y = user_setting['AIR_VELOCITY']
            plt.plot(x,y)
            st.pyplot(fig)

        elif col_list_rnt == 'TEMPERATURE':
            fig = plt.figure(figsize=(20, 10))
            plt.xlabel('ID')
            plt.title('Graph of TEMPERATURE')    
            plt.ylabel('TEMPERATURE')
            x=user_setting['ID']
            y = user_setting['TEMPERATURE']
            plt.plot(x,y)
            st.pyplot(fig)

        elif col_list_rnt == 'HUMIDITY':
            fig = plt.figure(figsize=(20, 10))
            plt.xlabel('ID')
            plt.title('Graph of HUMIDITY')    
            plt.ylabel('HUMIDITY')
            x=user_setting['ID']
            y = user_setting['HUMIDITY']
            plt.plot(x,y)
            st.pyplot(fig)

        elif col_list_rnt == 'ILLUMINATION_INTENSITY':
            fig = plt.figure(figsize=(20, 10))
            plt.xlabel('ID')
            plt.title('Graph of ILLUMINATION_INTENSITY')    
            plt.ylabel('ILLUMINATION_INTENSITY')
            x=user_setting['ID']
            y = user_setting['ILLUMINATION_INTENSITY']
            plt.plot(x,y)
            st.pyplot(fig)

        elif col_list_rnt == 'CARBON_DIOXIDE':
            fig = plt.figure(figsize=(20, 10))
            plt.xlabel('ID')
            plt.title('Graph of CARBON_DIOXIDE')    
            plt.ylabel('CARBON_DIOXIDE')
            x=user_setting['ID']
            y = user_setting['CARBON_DIOXIDE']
            plt.plot(x,y)
            st.pyplot(fig)   
    
    problem_factors=[]
    for i in range(total_len):
        
        if user_setting['DBYHS_SPCHCKN'][i] != 0:
            problem_factors.append(i+1)

    st.text('The ID that may be problematic are as follows')
    st.text(problem_factors)
    st.text('Please check the ID')
 
if __name__ == '__main__':
    user_setting_plt()
