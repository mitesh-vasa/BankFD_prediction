import streamlit as st
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

import pickle
file1=open('scale.pkl','rb')
file2=open('decisiontreegini.pkl','rb')
file3=open('decisiontreeentropy.pkl','rb')
file4=open('decisiontreemaxdept.pkl','rb')
file5=open('decisiontreeminsamples.pkl','rb')
file6=open('Randomforest.pkl','rb')
file7=open('adaptorboost.pkl','rb')
file8=open('logisticregression.pkl','rb')
file9=open('gradientboost.pkl','rb')
file10=open('extremegradientboost.pkl','rb')
file11=open('supportvector','rb')
scale=pickle.load(file1) # scale user defined function
dtg=pickle.load(file2)
dte=pickle.load(file3)
dtmd=pickle.load(file4)
dtms=pickle.load(file5)
rfc=pickle.load(file6)
ada=pickle.load(file7)
lr=pickle.load(file8)
gbc=pickle.load(file9)
xgc=pickle.load(file10)
svc=pickle.load(file11)

st.write("## Fixed Deposit prediction")
#st.image()
age=st.number_input('Age',value=0, step=1,format='%d')
job=st.number_input('job',value=0, step=1,format='%d')
marital=st.number_input("marital ststus 0-married,1-not married",value=0, step=1,format='%d')
education = st.number_input("education enter number from 0 to 3",value=0, step=1,format='%d')
default= st.number_input("default 0-no,1-yes",value=0, step=1,format='%d')
balance=st.number_input("bank balance",value=0, step=1,format='%d')
housing=st.number_input("housing 0-no,1-yes",value=0, step=1,format='%d')
loan=st.number_input("loan 0-no,1-yes",value=0, step=1,format='%d')
contact=st.number_input('contact enter number from 0-2',value=0, step=1,format='%d')
day=st.number_input('day enter number from 1 to 31',value=0, step=1,format='%d')
month=st.number_input('month enter month number',value=0, step=1,format='%d')
duration=st.number_input('duration enter month number',value=0, step=1,format='%d')
campaign=st.number_input('campaign enter number from 1 to 63',value=1, step=1,format='%d')
pdays=st.number_input('pdays enter number from -1 to 854',value=-1, step=1,format='%d')
previous=st.number_input('previous enter number from -1 to 53 ',value=-1, step=1,format='%d')
poutcome=st.number_input('poutcome enter number from 0 to 41 ',value=0, step=1,format='%d')
model_list=['decision tree entropy','decision tree max dept','decison tree min samples','random forest','adaptorboost','logisticregression','gradientboost','extremegradientboost','supportvector']
ml=[dte,dtmd,dtms,rfc,ada,lr,gbc,xgc,svc]
model=st.selectbox('select prediction model',model_list,index=model_list.index('decision tree entropy'))
a=model_list.index(model)
mode=ml[a]



if st.button('Predict'):
    import numpy as np
    x=[age,job,marital,education,default,balance,housing,loan,contact,day,month,duration,campaign,pdays,previous,poutcome]
    features=np.array([x])
    features=scale.transform(features)
    Y_pred=mode.predict(features)[0]
    print(Y_pred)
    if Y_pred==1:
        st.write('Customer will open a fixed deposit account')
    else:
        st.write('Customer will not open a fixed deposit account')