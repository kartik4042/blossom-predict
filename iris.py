import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Simple Iris Flower Prediction App
This app predicts **Iris Flower** type !
""")

st.sidebar.header("User input parameters :")

def user_parameters():
	sepal_length=st.sidebar.slider('Sepal length',4.3,7.9,5.4)
	sepal_width=st.sidebar.slider('Sepal width',2.0,4.4,3.4)
	petal_length=st.sidebar.slider('Petal length',1.0,6.9,1.3)
	petal_width=st.sidebar.slider('Petal width',0.1,2.4,0.2)
	data={'sepal length':sepal_length,
	      'sepal width':sepal_width,
	      'petal length':petal_length,
	      'petal width':petal_width}
	features=pd.DataFrame(data,index=[0])
	return features
	
df=user_parameters()
st.subheader('User input parameters :')
st.write(df)

iris=pd.read_csv(r"iris.csv")

attributes = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
feat= iris[attributes]
target=iris.Species

clf=RandomForestClassifier()

clf.fit(feat,target)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader('Class labels corresponding to their index')
st.write(iris.Species.unique())

st.subheader('Prediction :')
st.write(prediction)

st.subheader("Prediction Probability :")
st.write(prediction_proba)
