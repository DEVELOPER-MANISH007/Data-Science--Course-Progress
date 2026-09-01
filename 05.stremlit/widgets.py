import streamlit as st
import pandas    as pd
data = {
    "Name": ["Amit", "Sara", "John", "Priya", "Rahul"],
    "Age": [25, 30, 22, 28, 35],
    "City": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune"],
    "Salary": [50000, 60000, 45000, 52000, 70000]
}
df = pd.DataFrame(data)


st.title("Streamlit Text Input")

name = st.text_input("Enter Your Name:")
age = st.slider("select your age:")
st.write(f"your  age is :{age}")
options = ["Python","Java","C++","javascripts"]
choice = st.selectbox("choose your favorite lang:",options)
st.write(f"YOUR FAVORITE LANG IS :{choice}")
if name:
    st.write(f"Hello,{name}")
df.to_csv("ouput.csv")
st.write(df)

## uplaod btn
uploaded = st.file_uploader("choose your csv files  ",type= "csv")
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(df)
