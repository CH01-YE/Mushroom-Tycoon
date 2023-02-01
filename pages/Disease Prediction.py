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
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
sns.set(style = 'white', context = 'notebook', palette = 'deep')
from sklearn.metrics import accuracy_score

def train_data_load():

    st.text('Please select a file to test')
    user_file = st.file_uploader('Upload CSV File',type=['csv'])
    user_data = pd.read_csv(user_file)
    st.dataframe(user_data)

    rnt_list=['oyster', 'pleurotus', 'portobello', 'shiitake', 'winter']
    train_data_rnt = st.radio('Please select a type of mushroom.', rnt_list)

    if st.button('select'):
        if train_data_rnt == 'oyster':
            st.info('Selected. loading corresponding mushroom data..')
            train_data = pd.read_csv('train_oyster_data.csv')

        elif train_data_rnt == 'pleurotus':
            st.info('Selected. loading corresponding mushroom data..')
            train_data = pd.read_csv('train_pleurotus_data.csv')

        elif train_data_rnt == 'portobello':
            st.info('Selected. loading corresponding mushroom data..')
            train_data = pd.read_csv('train_portobello_data.csv')

        elif train_data_rnt == 'shiitake':
            st.info('Selected. loading corresponding mushroom data..')
            train_data = pd.read_csv('train_shiitake_data.csv')

        elif train_data_rnt == 'winter':
            st.info('Selected. loading corresponding mushroom data..')
            train_data = pd.read_csv('train_winter_data.csv')
###############################################################################

    st.info('Data import complete! Training...')
    x=train_data.drop('DBYHS_SPCHCKN', axis=1)
    y=train_data['DBYHS_SPCHCKN']
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, random_state=1)

    kfold = StratifiedKFold(n_splits=10)
    random_state = 2
    classifiers = []
    classifiers.append(SVC(random_state=random_state))
    classifiers.append(DecisionTreeClassifier(random_state=random_state))
    classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state, learning_rate=0.1))
    classifiers.append(RandomForestClassifier(random_state=random_state))
    classifiers.append(ExtraTreesClassifier(random_state=random_state))
    classifiers.append(GradientBoostingClassifier(random_state=random_state))
    classifiers.append(MLPClassifier(random_state=random_state))
    classifiers.append(KNeighborsClassifier())
    classifiers.append(LogisticRegression(random_state=random_state))
    classifiers.append(LinearDiscriminantAnalysis())

    # DTC
    DTC = DecisionTreeClassifier()
    adaDTC = AdaBoostClassifier(base_estimator = DTC, random_state=7)
    ada_param_grid = {"base_estimator__criterion": ["gini", "entropy"], 
    "base_estimator__splitter": ["best", "random"], 
    "algorithm": ["SAMME", "SAMME.R"], 
    "n_estimators": [1, 2],
    "learning_rate":  [0.1, 0.2]}
    gsadaDTC = GridSearchCV(adaDTC, param_grid=ada_param_grid,cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
    gsadaDTC.fit(X_train, y_train)
    ada_best = gsadaDTC.best_estimator_

    st.text(gsadaDTC.best_score_)

    #Ext
    ExtC = ExtraTreesClassifier()
    ex_param_grid = {"max_depth": [None],
    "max_features": [1, 3],
    "min_samples_split": [2, 3],
    "min_samples_leaf": [1, 3],
    "bootstrap": [False],
    "n_estimators" :[100],
    "criterion": ["gini"]}
    gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)
    gsExtC.fit(X_train,y_train)
    ExtC_best = gsExtC.best_estimator_

    st.text(gsExtC.best_score_)

    # RFC
    RFC = RandomForestClassifier()
    rf_param_grid = {'max_depth' : [None],
                    'max_features' : [1,3],
                    'min_samples_split' : [2,3],
                    'min_samples_leaf' : [1,3],
                    'bootstrap' : [False],
                    'n_estimators' : [100],
                    'criterion' :['gini']}
    gsRFC = GridSearchCV(RFC, param_grid = rf_param_grid, cv = kfold, scoring='accuracy', n_jobs = -1, verbose = 1)
    gsRFC.fit(X_train, y_train)
    RFC_best = gsRFC.best_estimator_
    st.text(gsRFC.best_score_)
    
    # Gradient boosting
    GBC = GradientBoostingClassifier()
    gb_param_grid = {'loss' : ['deviance'],
                    'n_estimators' : [100],
                    'learning_rate' : [0.1, 0.05],
                    'max_depth' : [4, 8],
                    'min_samples_leaf' : [100],
                    'max_features' : [0.3, 0.1]}
    gsGBC = GridSearchCV(GBC, param_grid= gb_param_grid, cv = kfold, scoring= 'accuracy', n_jobs = -1, verbose = 1)
    gsGBC.fit(X_train, y_train)
    GBC_best = gsGBC.best_estimator_
    st.text(gsGBC.best_score_)

    # SVC
    SVMC = SVC(probability=True)
    svc_param_grid = {'kernel' : ['rbf'],
                    'gamma': [0.1, 1],
                    'C':[1, 10]}
    gsSVMC = GridSearchCV(SVMC, param_grid = svc_param_grid, cv = kfold, scoring='accuracy', n_jobs = -1, verbose = 1)
    gsSVMC.fit(X_train, y_train)
    SVMC_best = gsSVMC.best_estimator_
    st.text(gsSVMC.best_score_)

    st.info('Learning Completed!')
######################################################################

    votingC = VotingClassifier(estimators=[('rfc', RFC_best), ('extc', ExtC_best),
    ('svc', SVMC_best), ('adac',ada_best),('gbc',GBC_best)], voting='soft', n_jobs=4)
    
    votingC = votingC.fit(X_train, y_train)

    result = pd.Series(votingC.predict(user_data), name="DBYHS_SPCHCKN")
    result.to_csv('User_DBYHS_SPCHCKN.csv', index=False)

    st.text('Prediction result saved as user_results.csv')


if __name__ == '__main__':
    train_data_load()
