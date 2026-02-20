import streamlit as st
from sklearn.datasets import load_iris
data = load_iris()
from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier()
x = data.data
y = data.target
random_forest_model.fit(x,y)
st.header("Iris Flower classification")
sl = st.number_input("Enter sepal Length")
sw = st.number_input("Enter sepal width")
pl = st.number_input("Enter petal Length")
pw = st.number_input("Enter petal width")
y_pred = random_forest_model.predict([[sl,sw,pl,pw]])
op = data.target_names[y_pred[0]]
st.write(op)
