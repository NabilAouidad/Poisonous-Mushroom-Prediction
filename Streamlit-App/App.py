import streamlit as st
import torch
import pandas as pd
import plotly.express as px
from distributions import plotBars, plotHistograms
from correlations import plotPairPlots, plotHeatMap
import joblib
from torch.utils.data import TensorDataset, DataLoader

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout = "wide", page_title = "Poisonous Mushroom Prediction")

style = """
    <style>
    button[data-baseweb="tab"] {
        margin: 0 auto;
    }

    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:1.3rem;
    }
    </style>
    """

st.markdown(style, unsafe_allow_html = True)

container = st.container(height = 125, border = False)
container.write('<h1 style="text-align: center;">Poisonous Mushroom Prediction Web App</h1>', 
         unsafe_allow_html=True)

st.empty()

#-----------------------------------------------------------

mushroom_df = pd.read_csv("Streamlit-App/train_clean_sample.csv")
mushroom_df.drop("id", axis = 1, inplace = True)

def eda():
    tab1, tab2, tab3 = st.tabs(["About the dataset", "Data Distributions", "Data Relationships"])

    with tab1:
        shapeColumn, featuresColumn, descriptiveStatsColumn = st.columns(3)

        shapeColumn.subheader("Shape")
        shapeColumn.write(mushroom_df.shape)

        featuresColumn.subheader("Features")
        featuresColumn.write(pd.DataFrame({"Column Name" : mushroom_df.columns}, index = range(1, len(mushroom_df.columns)+1)))

        descriptiveStatsColumn.subheader("Descriptive Statistics")
        descriptiveStatsColumn.dataframe(mushroom_df.describe())

        st.subheader("Data Sample")
        st.dataframe(mushroom_df.sample(20))

    with tab2:
        st.subheader("Bar Plots")
        st.plotly_chart(plotBars(mushroom_df))

        st.subheader("Histograms")
        st.plotly_chart(plotHistograms(mushroom_df))

    with tab3:
        st.subheader("Pair Plot")
        st.plotly_chart(plotPairPlots(mushroom_df))

        st.subheader("Heat Map")
        st.plotly_chart(plotHeatMap(mushroom_df))

st.sidebar.title("Options")

box_values = st.sidebar.selectbox(" ", options = ["EDA", "Make Predictions"])

if box_values == "EDA":
    eda()