#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip install streamlit')

import io
import streamlit as st

# pandas for data structures (pre-processing) and operations for manipulating numerical tables and time series
import pandas as pd
from pandas.plotting import scatter_matrix

# matplotlib.pyplot for data plots
import matplotlib.pyplot as plt

# sklearn for machine learning methods
from sklearn import tree
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# for numeric calculations
import numpy as np

import graphviz
import seaborn as sns

st.write("Hello")

# Convert a Matplotlib figure to a PNG image
def fig_to_image(fig):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Title and description
st.title("Diabetes Prediction Web App")
st.write("Upload a CSV file to predict diabetes using a Decision Tree Classifier.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded data
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head(10))

    # Data Cleaning
    df = df.dropna()

    # Summary Statistics
    st.write("Descriptive Statistics:")
    st.write(df.describe())

    # Visualization
    st.write("Box-Whisker Plots:")
    # Create a Matplotlib figure from the box plot
    fig, ax = plt.subplots()
    df.plot(kind='box', subplots=True, layout=(3, 5), sharex=False, sharey=False, ax=ax)
    st.pyplot(fig)

    st.write("Histograms:")
    # Create a Matplotlib figure from the histograms
    fig, ax = plt.subplots()
    df.hist(ax=ax)
    st.pyplot(fig)

    # Model Training
    array = df.values
    X, y = array[:, :-1], array[:, -1]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=7)

    params = {'max_depth': 5}
    classifier = DecisionTreeClassifier(**params)
    classifier.fit(X_train, y_train)

    st.write("Decision Tree Model Trained")

    # Predictions
    y_testp = classifier.predict(X_test)

    st.write("Predicted Labels:")
    st.write(y_testp)

    st.write("Accuracy:", accuracy_score(y_test, y_testp))

    # Confusion Matrix
    confusion_mat = confusion_matrix(y_test, y_testp)
    st.write("Confusion Matrix:")
    st.write(confusion_mat)

    # Visualization of Confusion Matrix
    st.write("Visualized Confusion Matrix:")

    # Interactive controls to adjust heatmap
    annot = st.checkbox("Show Annotations", value=True)
    cmap = st.selectbox("Colormap", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])
    
    fig, ax = plt.subplots()
    sns.heatmap(confusion_mat, annot=annot, cmap=cmap, ax=ax)
    st.pyplot(fig)

    # Save the heatmap as an image
    heatmap_img = fig_to_image(fig)

    # Display the image
    st.image(heatmap_img)

    st.pyplot(sns.heatmap(confusion_mat, annot=True))

    # Classifier performance
    class_names = ['Class0', 'Class1']
    st.write("Classifier Performance on Test Dataset:")
    st.write(classification_report(y_test, y_testp, target_names=class_names))

    # Graphviz visualization of the decision tree
    dot_data = tree.export_graphviz(classifier, out_file=None,
                                    feature_names=list(df.columns)[:-1], class_names=[str(i) for i in set(y)],
                                    filled=True, rounded=True, proportion=False, special_characters=True)
    graph = graphviz.Source(dot_data)
    st.write("Decision Tree Visualization:")
    st.graphviz_chart(graph)

    
    
    
    
    
    