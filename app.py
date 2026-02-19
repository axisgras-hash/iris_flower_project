import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

st.title('🪻Iris Flower Classification Project using ML🪻')
st.image('https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png')

data = load_iris()

y = data['target']
X = pd.DataFrame(data['data'],columns = data['feature_names'])
target_class = data['target_names']
# target_class

st.sidebar.title('Select Iris Features')
st.sidebar.image('https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png')


user_input = []
for i in X:
  min_value = X[i].min()
  max_value = X[i].max()

  ans = st.sidebar.slider(f'Select value of {i}',min_value,max_value)
  user_input.append(ans)


final_input = [user_input]

with open('iris_model.pkl', 'rb') as f:
  chatgpt = pickle.load(f)

final_ans = chatgpt.predict(final_input)[0]
flower_name = target_class[final_ans]
prob = chatgpt.predict_proba(final_input).ravel()

for i,j in enumerate(prob):
  flower = target_class[i]
  st.write(f'Probability of {flower} is : {round(j*100,2)}')

with st.spinner('Wait for it...'):
    time.sleep(2)
st.success(f'The Final predicted Flower is : {flower_name}')

st.image(f'{flower_name.lower()}' + '.png')

st.markdown("### 👨‍💻 Connect with Me")

st.markdown(
    """
    🔗 **GitHub:** [https://github.com/axisgras-hash](https://github.com/axisgras-hash)  
    💼 **LinkedIn:** [https://linkedin.com/in/axisgras-hash](https://linkedin.com/in/axisgras-hash)
    """
)





