import plotly
import plotly.plotly as py
import plotly.graph_objs as go
#The line below will let you run this offline.
from plotly.offline import download_plotlyjs,plot, iplot
import numpy as np


def make__3d_plot(W, labels, xcol=0, ycol=1, zcol=2):
    '''Take in a W matrix from NMF clustering and create a 3-d plot'''
    x = W[:,xcol].flatten()
    y = W[:,ycol].flatten()
    z = W[:,zcol].flatten()

    trace1 = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=5,color=labels,colorscale='Portland',opacity=0.6))
    data = [trace1]
    layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
    fig = go.Figure(data=data, layout=layout)

    # This will create the html for the plot
    plot(fig, filename='templates/3d-scatter-plot.html')

    # This will make a <div> that you can paste into existing code:
    # print(plot(fig, include_plotlyjs=False, output_type='div'))
    # To use this, you will need to include <script src="https://cdn.plot.ly/plotly-latest.min.js"></script> in your html

if __name__ == '__main__':
    make__3d_plot(matrix, labels, xcol=0, ycol=1, zcol=2)
