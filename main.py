import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sns.set_palette('pastel')  # Cambia los colores a tonos pastel
sns.set_style('whitegrid')  # Cambia el estilo a 'whitegrid'
from cycler import cycler
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.'])))
plt.style.use('ggplot')  # Cambia el estilo a 'ggplot'

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache_data
def get_data():
    # Carga los datos desde tu repositorio
    data_url = "base.csv"
    return pd.read_csv(data_url)

@st.cache_data
def get_resumen(tipo_accidente):
    return df.query("""tipo_accidente==@tipo_accidente""").describe().T

df = get_data()
csv = convert_df(df)

def inicio():
    st.title("Equipo 1")
    st.header("Accidentes de Tráfico en Jalisco")
    image = Image.open('traffic_light.png')
    st.image(image, caption='Traffico de Guadalajara')
    st.text("Brenda Cristina Yepiz\nHéctor Calderón Reyes\nAxel Jarquín\nJohn Paul Cueva Osete")

def datos():
    st.header("Datos")
    df = get_data()
    imagen = Image.open('city.png')
    st.image(imagen, caption='Satelital')
    # Paso 1 - Ordenar
    st.subheader("Ordenar en tablas")
    st.text("Los cinco accidentes más recientes")
    st.write(df.sort_values("anio", ascending=False).head())
    # Paso 3 - Filtrado de columnas
    st.subheader("Visualización personalizada")
    st.markdown("Selecciona columnas para mostrar")
    default_cols = ["anio", "mes", "dia", "mun", "tipo_accidente"]
    cols = st.multiselect("Columnas", df.columns.tolist(), default=default_cols)
    st.dataframe(df[cols].head(10))
    st.markdown("\nDescargar dataset: ")
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='base.csv',
        mime='text/csv',
)

def mapa():
    # Paso 2 - Visualización en mapa
    st.header("Mapa")
    st.subheader("Accidentes en el mapa")
    # Renombrar las columnas 'y' y 'x' a 'lat' y 'lon' respectivamente
    df.rename(columns={'y': 'lat', 'x': 'lon'}, inplace=True)
    # Obtener todas las columnas excepto 'Id'
    columns = [col for col in df.columns if col != 'Id']
    # Añadir un selector para la columna de filtrado en la barra lateral
    filter_column = st.sidebar.selectbox("Selecciona una columna para filtrar el mapa:", columns)
    # Añadir un selector para el valor de filtrado en la barra lateral
    filter_value = st.sidebar.selectbox("Valor del filtro del mapa:", df[filter_column].unique())
    # Filtrar los datos según la selección del usuario
    filtered_data = df[df[filter_column] == filter_value]
    # Mostrar el mapa
    st.map(filtered_data.dropna(subset=["lat", "lon"])[["lat", "lon"]])

def analisis():
    st.header("Análisis")
    # Paso 4 - Agrupación estática
    st.subheader("Cantidad de accidentes por tipo")
    st.table(df.groupby("tipo_accidente").size().reset_index(name="Cantidad").sort_values("Cantidad", ascending=False))
    # Paso 6 - Botones de radio
    st.subheader("Tipo de Accidentes")
    tipo_accidente = st.radio("", df.tipo_accidente.unique())
    st.table(get_resumen(tipo_accidente))
    # Paso 7 number summary (BoxPlot)
    st.subheader("Resumen de 5 números")
    st.write(df.describe())
    # Paso 8 - Skewness
    st.subheader("Skewness")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    skewness = df[numerical_columns].skew()
    st.write(skewness)
    # Paso 9 - Kurtosis
    st.subheader("Kurtosis")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    kurtosis = df[numerical_columns].kurtosis()
    st.write(kurtosis)

def visualizaciones():
    st.header("Visualizaciones")
    # Paso 10 - Análisis de Correlaciones
    st.subheader("Análisis de Correlaciones")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_columns].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    # Paso 11 - Análisis de Outliers
    st.header("Análisis de Outliers")
    # Seleccionar columnas numéricas para el análisis de outliers
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Seleccionar columna para el análisis de outliers
    selected_col = st.sidebar.selectbox("Selecciona una columna numérica para el análisis de outliers", numeric_cols)
    # Verificar si se ha seleccionado una columna
    if selected_col:
        st.subheader(f"Análisis de outliers para {selected_col}")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[selected_col], ax=ax)
        ax.set_xlabel(selected_col)
        st.pyplot(fig)
    else:
        st.write("No se ha seleccionado una columna para el análisis de outliers.")
    # Paso 13 - Distribución
    st.header("Análisis de Distribución")
    # Diccionario de mapeo de nombres de mes a valores numéricos
    meses_dict = {
        "Enero": 1,
        "Febrero": 2,
        "Marzo": 3,
        "Abril": 4,
        "Mayo": 5,
        "Junio": 6,
        "Julio": 7,
        "Agosto": 8,
        "Septiembre": 9,
        "Octubre": 10,
        "Noviembre": 11,
        "Diciembre": 12
    }
    # Función de conversión de valores numéricos a nombres de mes
    def number_to_month(num):
        meses = {
            1: "Enero",
            2: "Febrero",
            3: "Marzo",
            4: "Abril",
            5: "Mayo",
            6: "Junio",
            7: "Julio",
            8: "Agosto",
            9: "Septiembre",
            10: "Octubre",
            11: "Noviembre",
            12: "Diciembre"
        }
        return meses.get(num, "")
    # Aplicar el mapeo a la columna de mes
    df["mes_num"] = df["mes"].map(meses_dict)
    # Obtener los valores mínimos y máximos de los meses numéricos
    min_month = int(df["mes_num"].min())
    max_month = int(df["mes_num"].max())
    # Agregar la barra deslizante en el sidebar para seleccionar el rango de meses
    selected_range = st.sidebar.slider("Selecciona el rango de meses", min_month, max_month, (min_month, max_month))
    # Filtrar los datos según el rango de meses seleccionado
    filtered_df = df[df["mes_num"].between(selected_range[0], selected_range[1])]
    st.subheader(f"Análisis de outliers para el rango de meses {number_to_month(selected_range[0])} a {number_to_month(selected_range[1])}")
    # Mostrar la cantidad de accidentes por mes en una gráfica
    accidents_by_month = filtered_df["mes_num"].value_counts().sort_index()
    st.bar_chart(accidents_by_month)

PAGES = {
    "Inicio": inicio,
    "Datos": datos,
    "Mapa": mapa,
    "Análisis": analisis,
    "Visualizaciones": visualizaciones
}

def main():
    st.sidebar.title("Navegación")
    selection = st.sidebar.radio("Ir a:", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
