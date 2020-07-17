import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    # Streamlit decorator to cache the output and use it whenever rerendered
    @st.cache(persist=True)
    def load_data():
        data = pd.read_csv(
            r"C:\Users\Utkarsh.DESKTOP-6FKGAET\Utkarsh\Projects\Classification-Algorithms-Visualiser\Mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    df = load_data()

    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Dataset (Classification)")
        st.write(df)


if __name__ == '__main__':
    main()
