import plotly_express as px
from itertools import combinations


def plot3D(df,X,Y,Z,color):
    fig = px.scatter_3d(df,x=X,y=Y,z=Z,color=color,opacity=0.5)
    fig.show(renderer="browser")


def plot3d_combinations(df,features):
    """
    Realiza graficas 3D con diferentes combinaciones de ejes
    Parameters
    ----------
    df: datos a graficar
    features: lista con los ejes (características) que se desean visualizar

    Returns
    -------
    Plot 3D por cada combinación de ejes
    """
    temp = combinations(features, 3)
    for i in list(temp):
        plot3D(df, i[0], i[1], i[2], 'cluster')
