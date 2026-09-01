import streamlit as st
import pandas as pd
import numpy as np

#  title of the application
st.title("Hello Streamlit")

# display a simple text
st.write("this is a simple text")

## create a simple dataframe
import pandas as pd

data = {
    "Name": ["Amit", "Sara", "John", "Priya", "Rahul"],
    "Age": [25, 30, 22, 28, 35],
    "City": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Pune"],
    "Salary": [50000, 60000, 45000, 52000, 70000]
}

df = pd.DataFrame(data)

## Display the dataframe
st.write("Here is the dataframe")
st.write(df)


## create a line chart
chart_data = pd.DataFrame(
    np.random.randn(20,3), columns=["a",'b','c']
)
st.line_chart(chart_data)