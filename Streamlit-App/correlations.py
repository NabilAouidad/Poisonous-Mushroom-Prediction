import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plotPairPlots(df):
    num_cols = list(df.describe(include = np.number).columns)
    fig = go.Figure(data = px.scatter_matrix(df[num_cols],
                                             color = df["class"],
                                         dimensions = num_cols,
                                           width = 1000, height = 1000)
    )
    return fig

def plotHeatMap(df):
    num_cols = list(df.describe(include = np.number).columns)
    correlation_matrix = df[num_cols].corr()
    fig = go.Figure(data = px.imshow(correlation_matrix,
                                     color_continuous_scale = "Inferno",
                                     text_auto = ".3f",
                                     width = 1000, height = 1000)
    )
    return fig