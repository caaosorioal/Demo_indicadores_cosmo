import streamlit as st
import pandas as pd
import numpy as np

# Crear una aplicación de streamlit
st.title('Matriz indicadores - Demo')

# Cargar datos
df = pd.read_csv('data.csv', sep = ';', encoding = 'latin-1')

# Seleccionar las columnas del eje horizontal
cols_y = [
    '%pagos_sin_becas',
    'ocupacion',
    'Matriculados'
]

cols_x = [
    'p_welbin',
    'ambiente_escolar_familias',
    'promedio_linea_base'
]

# Nombres para etiquetas
nombres = {
    '%pagos_sin_becas': '% Pagos sin becas',
    'ocupacion': '% Ocupación',
    'Matriculados': 'Número Matriculados',
    'p_welbin': 'Indicador Welbin',
    'ambiente_escolar_empleados': 'Ambiente escolar empleados',
    'ambiente_escolar_familias': 'Ambiente escolar familias',
    'promedio_linea_base': 'Promedio línea base'
}

# Normalizar todas las columnas entre 0 y 1 usando la función sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x_):
    x = np.array(x_)
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

# Normalizar las columnas de x entre 0 y 1 usando MinMaxScaler
df[cols_x] = df[cols_x].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Normalizar las columnas de y entre 0 y 1 usando MinMaxScaler
df[cols_y] = df[cols_y].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

# Crear dos columnas, una para los sliders de x y otra para los sliders de y
col1, col2 = st.columns(2)

# Mostrar los sliders de x
with col1:
    st.header('Indicadores de aprendizaje y ambiente')
    # Para cada una de las columnas de x, crear un slider
    pesos_x = {}
    for col in cols_x:
        st.text(nombres[col])
        pesos_x[col] = st.slider(col, min_value = 0.1, max_value = 1.0, value = 0.5, step = 0.05, label_visibility='collapsed')

# Mostrar los sliders de y
with col2:
    st.header('Indicadores de gestión')
    # Para cada una de las columnas de y, crear un slider
    pesos_y = {}
    for col in cols_y:
        st.text(nombres[col])
        pesos_y[col] = st.slider(col, min_value = 0.1, max_value = 1.0, value = 0.5, step = 0.05, label_visibility='collapsed')

# Crear un diagrama de dispersión con la suma ponderada en x, y utilizando plotly
import plotly.graph_objects as go
import plotly.express as px

# Calcular la suma ponderada de las columnas de x
df['x'] = df[cols_x].mul(softmax(list(pesos_x.values())), axis = 1).sum(axis = 1)

# Calcular la suma ponderada de las columnas de y
df['y'] = df[cols_y].mul(softmax(list(pesos_y.values())), axis = 1).sum(axis = 1)

# Crear columna dummy para tamaño
df['size'] = 20

# Crear el diagrama de dispersión con plotly pero no usando plotly.express
# fig = go.Figure(
#     data = go.Scatter(
#     x = df['x'], 
#     y = df['y'], 
#     mode = 'markers', 
#     marker = dict(size = 16, color = df['Matriculados'], colorscale = 'Viridis', showscale = False, opacity = 0.8),
#     showlegend = True),
#     layout_yaxis_range=[0,1],
#     layout_xaxis_range=[0,1]
# )

fig = px.scatter(
    df,
    x = 'x',
    y = 'y',
    color = 'colegio',
    size = 'size',

)

# Adicionar la leyenda para ver los colegios
fig.update_layout(
    title = 'Matriz de indicadores',
    xaxis_title = 'Indicadores de aprendizaje',
    yaxis_title = 'Indicadores de gestión'
)

# Colocar el eje y siempre entre 0 y 1
fig.update_yaxes(range=[0,1])
fig.update_xaxes(range=[0,1])
st.plotly_chart(fig)

