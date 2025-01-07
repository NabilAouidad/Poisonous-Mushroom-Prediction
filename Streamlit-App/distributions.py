import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def plotBars(df):
    num_cols = list(df.describe(include = np.number).columns)
    obj_cols = list(df.describe(include = np.object_).columns)
    n_rows, n_cols = 1, 1
    c = 0
    fig = make_subplots(rows = n_rows, cols = n_cols,
                        column_widths = [500], 
                        row_heights = [500],
                        horizontal_spacing = 0.13, vertical_spacing = 0.13)
    fig.update_layout({"height":500, "width":750})

    for i in range(1, n_rows+1):
        for j in range(1, n_cols+1):
            if i*j > len(obj_cols):
                break
            else :
                count = df[obj_cols[c]].value_counts().sort_values().head(10)
                fig.add_trace(go.Bar(x = count.values, y = count.index,
                                      orientation = 'h', 
                                      name = obj_cols[c]), row = i, col = j)
                c += 1
    fig.update_layout({"title":"Sex distribution of the abalone dataset"})
    return fig

def plotHistograms(df):
    num_cols = list(df.describe(include = np.number).columns)
    obj_cols = list(df.describe(include = np.object_).columns)
    n_rows, n_cols = 3, 3
    colors = ["red", "magenta", "green", "yellow", "orange", "blue", "red", "magenta", "green"]
    c = 0
    fig = make_subplots(rows = n_rows, cols = n_cols,
                        column_widths = [1000, 1000, 1000], 
                        row_heights = [1000, 1000, 1000],
                        horizontal_spacing = 0.13, vertical_spacing = 0.17)
    fig.update_layout({"height":900, "width":900})
    
    for i in range(1, n_rows+1):
        for j in range(1, n_cols+1):
            if i*j > len(num_cols):
                break
            else :
                fig.add_trace(go.Histogram(x = df[num_cols[c]],
                                       histfunc = "count",
                                       name = num_cols[c], marker = {"color":colors[c]},
                                       nbinsx = 20),
                                       row = i, col = j)
                c += 1
    fig.update_layout({"title":"Numerical Features distributions"})
    return fig