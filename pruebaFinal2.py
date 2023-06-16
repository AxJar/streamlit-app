import pandas as pd
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import requests
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

@st.cache_data
def get_data():
    # Carga los datos desde tu repositorio
    data_url = "base.csv"
    df = pd.read_csv(data_url)
    
    # FillNA
    df["clave_mun"] = df["clave_mun"].fillna(97)
    df["herido"] = df["herido"].fillna("Ileso")
    df["tipo_accidente"] = df["tipo_accidente"].fillna(df["tipo_accidente"].mode()[0])
    df["x"] = df["x"].fillna(df["x"].mean())
    df["y"] = df["y"].fillna(df["y"].mean())
    df["cruce_osm"] = df["cruce_osm"].fillna("Ninguno")
    
    return df

df = get_data()

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

@st.cache_data
def get_resumen(tipo_accidente):
    return df.query("""tipo_accidente==@tipo_accidente""").describe().T

#Convertir todas las columnas strings a ints/floats
def encode_column(df, column_name):
    if df[column_name].dtype == 'object':
        le = LabelEncoder()
        df[column_name] = le.fit_transform(df[column_name].astype(str))
    return df

def inicio():
    df = get_data()
    csv = convert_df(df)
    st.title("Equipo 1")
    st.header("Accidentes de Tráfico en Jalisco")
    image = Image.open('traffic_light.png')
    st.image(image, caption='Traffico de Guadalajara')

    st.header("Nuestro Equipo")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.header("Brenda Cristina Yepiz")
        st.image("Brenda.jpg")  # Asegúrate de tener la imagen en la ruta especificada

    with col2:
        st.header("Héctor Calderón Reyes")
        st.image("Hector.jpg")  # Asegúrate de tener la imagen en la ruta especificada

    with col3:
        st.header("Axel Jarquín Morga")
        st.image("Axel.jpg")  # Asegúrate de tener la imagen en la ruta especificada

    with col4:
        st.header("John Paul Cueva Osete")
        st.image("JP.jpg")  # Asegúrate de tener la imagen en la ruta especificada
        
def datos():
    df = get_data()
    csv = convert_df(df)
    st.header("Datos")
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
    df = get_data()
    csv = convert_df(df)
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
    # Crear un diccionario para mapear los nombres de los meses a números
    meses_dict = {
        'Enero': '01',
        'Febrero': '02',
        'Marzo': '03',
        'Abril': '04',
        'Mayo': '05',
        'Junio': '06',
        'Julio': '07',
        'Agosto': '08',
        'Septiembre': '09',
        'Octubre': '10',
        'Noviembre': '11',
        'Diciembre': '12'
    }
    # Crear un diccionario inverso para mapear los números del mes de vuelta a los nombres del mes
    meses_dict_inv = {v: k for k, v in meses_dict.items()}
    # Reemplazar los nombres de los meses por números en la columna 'mes'
    df['mes_num'] = df['mes'].replace(meses_dict)
    # Crear una nueva columna que una las columnas 'mes_num' y 'anio'
    df['mes_anio'] = df['anio'].astype(str) + df['mes_num']
    # Crear un dataframe pivot para obtener el conteo de Hombres y Mujeres por mes por año
    df_pivot = df.pivot_table(index='mes_anio', columns='sexo', aggfunc='size', fill_value=0)
    # Obtener la lista de años únicos en el dataframe
    years = sorted(df['anio'].unique())
    # Agregar un selector de años en el sidebar de Streamlit
    selected_years = st.sidebar.multiselect('Selecciona los años de las gráficas', years, default=years)
    # Agregar un selector de opciones en la barra lateral para elegir cómo ordenar los datos
    order_option = st.sidebar.selectbox('Ordenar datos por', ['Mes', 'Año'], index=1)
    # Filtrar el dataframe pivot por los años seleccionados
    df_pivot_filtered = df_pivot[df_pivot.index.str[:4].isin(map(str, selected_years))]
    # Ordenar el dataframe filtrado por el índice o por el año dependiendo de la opción seleccionada
    if order_option == 'Año':
        df_pivot_filtered = df_pivot_filtered.sort_index()
    else:
        df_pivot_filtered = df_pivot_filtered.sort_index(key=lambda x: x.str[-2:] + x.str[:4])
    # Convertir los números del mes de vuelta a los nombres del mes en el índice del dataframe filtrado
    df_pivot_filtered.index = df_pivot_filtered.index.map(lambda x: meses_dict_inv[x[-2:]] + x[:4])
    # Crear una gráfica de barras apiladas
    fig, ax = plt.subplots(figsize=(10, 7))
    df_pivot_filtered.plot(kind='bar', stacked=True, ax=ax)
    ax.set_title('Conteo de Accidentes por Sexo por Mes por Año')
    ax.set_xlabel('Mes y Año')
    ax.set_ylabel('Conteo de Accidentes')
    ax.legend(title='Sexo')
    # Mostrar la gráfica en Streamlit
    st.subheader("Proporción de Accidentes por Sexo")
    st.pyplot(fig)
    # Agregar un slider para seleccionar el grado de la regresión en el sidebar de Streamlit
    degree = st.sidebar.slider('Selecciona el grado de la regresión', min_value=1, max_value=10)
    # Filtrar el dataframe por los años seleccionados
    df_filtered = df[df['anio'].isin(selected_years)]
    # Filtrar el dataframe por el valor específico en la columna 'herido'
    df_herido = df_filtered[df_filtered['herido'] == 'Lesionado']
    # Agrupar el dataframe filtrado por 'mes_anio' y obtener el conteo de accidentes
    df_herido_grouped = df_herido.groupby('mes_anio').size()
    # Crear una matriz de características con los números de mes y año
    X = np.array([int(i) for i in df_herido_grouped.index]).reshape(-1, 1)
    # Crear un vector de destino con el conteo de accidentes
    y = df_herido_grouped.values
    # Crear una instancia de PolynomialFeatures para transformar la matriz de características
    poly = PolynomialFeatures(degree=degree)
    # Transformar la matriz de características
    X_poly = poly.fit_transform(X)
    # Crear una instancia de LinearRegression y ajustar el modelo a los datos
    model = LinearRegression().fit(X_poly, y)
    # Predecir el conteo de accidentes con el modelo
    y_pred = model.predict(X_poly)
    # Crear una gráfica de dispersión con los datos originales y la línea de regresión polinomial
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X, y, color='blue', label='Datos originales')
    ax.plot(X, y_pred, color='red', label='Regresión polinomial')
    ax.set_title('Regresión Polinomial del Conteo de Accidentes por Mes por Año: Lesionado')
    ax.set_xlabel('Mes y Año')
    ax.set_ylabel('Conteo de Accidentes')
    ax.legend()
    # Mostrar la gráfica en Streamlit
    st.subheader("Regresión Polinomial del valor Lesionado")
    st.pyplot(fig)
    # Filtrar el dataframe por el valor específico en la columna 'herido'
    df_herido = df_filtered[df_filtered['herido'] == 'Fallecido']
    # Agrupar el dataframe filtrado por 'mes_anio' y obtener el conteo de accidentes
    df_herido_grouped = df_herido.groupby('mes_anio').size()
    # Crear una matriz de características con los números de mes y año
    X = np.array([int(i) for i in df_herido_grouped.index]).reshape(-1, 1)
    # Crear un vector de destino con el conteo de accidentes
    y = df_herido_grouped.values
    # Crear una instancia de PolynomialFeatures para transformar la matriz de características
    poly = PolynomialFeatures(degree=degree)
    # Transformar la matriz de características
    X_poly = poly.fit_transform(X)
    # Crear una instancia de LinearRegression y ajustar el modelo a los datos
    model = LinearRegression().fit(X_poly, y)
    # Predecir el conteo de accidentes con el modelo
    y_pred = model.predict(X_poly)
    # Crear una gráfica de dispersión con los datos originales y la línea de regresión polinomial
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X, y, color='blue', label='Datos originales')
    ax.plot(X, y_pred, color='red', label='Regresión polinomial')
    ax.set_title('Regresión Polinomial del Conteo de Accidentes por Mes por Año: Fallecido')
    ax.set_xlabel('Mes y Año')
    ax.set_ylabel('Conteo de Accidentes')
    ax.legend()
    # Mostrar la gráfica en Streamlit
    st.subheader("Regresión Polinomial del valor Fallecido")
    st.pyplot(fig)
    # Filtrar el dataframe por el valor específico en la columna 'herido'
    df_herido = df_filtered[df_filtered['herido'] == 'Ileso']
    # Agrupar el dataframe filtrado por 'mes_anio' y obtener el conteo de accidentes
    df_herido_grouped = df_herido.groupby('mes_anio').size()
    # Crear una matriz de características con los números de mes y año
    X = np.array([int(i) for i in df_herido_grouped.index]).reshape(-1, 1)
    # Crear un vector de destino con el conteo de accidentes
    y = df_herido_grouped.values
    # Crear una instancia de PolynomialFeatures para transformar la matriz de características
    poly = PolynomialFeatures(degree=degree)
    # Transformar la matriz de características
    X_poly = poly.fit_transform(X)
    # Crear una instancia de LinearRegression y ajustar el modelo a los datos
    model = LinearRegression().fit(X_poly, y)
    # Predecir el conteo de accidentes con el modelo
    y_pred = model.predict(X_poly)
    # Crear una gráfica de dispersión con los datos originales y la línea de regresión polinomial
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(X, y, color='blue', label='Datos originales')
    ax.plot(X, y_pred, color='red', label='Regresión polinomial')
    ax.set_title('Regresión Polinomial del Conteo de Accidentes por Mes por Año: Ileso')
    ax.set_xlabel('Mes y Año')
    ax.set_ylabel('Conteo de Accidentes')
    ax.legend()
    # Mostrar la gráfica en Streamlit
    st.subheader("Regresión Polinomial del valor Ileso")
    st.pyplot(fig)

def analisis():
    df = get_data()
    csv = convert_df(df)
    st.header("Análisis")
    # Paso 4 - Agrupación estática
    st.subheader("Cantidad de accidentes por tipo")
    st.table(df.groupby("tipo_accidente").size().reset_index(name="Cantidad").sort_values("Cantidad", ascending=False))
    # Paso 6 - Botones de radio
    st.subheader("Tipo de Accidentes")
    tipo_accidente = st.radio("", df.tipo_accidente.unique())
    st.table(get_resumen(tipo_accidente))
    # Agrupación estática 2
    st.subheader("Cantidad de accidentes por cruce de calles")
    st.table(df.groupby("cruce_setrans").size().reset_index(name="Cantidad").sort_values("Cantidad", ascending=False).head())
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
    df = get_data()
    csv = convert_df(df)
    st.header("Visualizaciones")
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
    # Paso 10 - Análisis de Correlaciones
    st.subheader("Análisis de Correlaciones")
    encode_column(df, "Id")
    encode_column(df, "mes")
    encode_column(df, "dia_sem")
    encode_column(df, "mun")
    encode_column(df, "herido")
    encode_column(df, "ibaen_atro")
    encode_column(df, "rango_hora")
    encode_column(df, "tipo_accidente")
    encode_column(df, "cruce_setrans")
    encode_column(df, "cruce_osm")
    encode_column(df, "rango_edad")
    encode_column(df, "tipo_usuario")
    encode_column(df, "sexo")
    # Eliminar la columna "mes_num" antes de calcular la matriz de correlación
    df_without_mes_num = df.drop(columns=["mes_num"])
    # Correlación de Pearson
    st.subheader("Correlación de Pearson")
    corr_matrix_pearson = df_without_mes_num.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix_pearson, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    # Correlación de Kendall
    st.subheader("Correlación de Kendall")
    corr_matrix_kendall = df_without_mes_num.corr(method='kendall')
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix_kendall, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    # Correlación de Spearman
    st.subheader("Correlación de Spearman")
    corr_matrix_spearman = df_without_mes_num.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(corr_matrix_spearman, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def regresionLog():
    # Seleccionar las variables relevantes
    variables_relevantes = ['tipo_usuario', 'dia_sem', 'rango_hora']

    # Convertir las variables categóricas a números
    label_encoders = {}
    for col in variables_relevantes:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X = df[variables_relevantes]
    y = df['dileso']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo de regresión logística
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Evaluar el modelo
    st.markdown(f'*Puntuación del modelo:* {modelo.score(X_test, y_test)}')

    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Crear un histograma de las predicciones y los valores reales
    fig, ax = plt.subplots(figsize=(10, 6))  # Crea una nueva figura con un subplot
    ax.hist([y_pred, y_test], label=['Predicciones', 'Valores Reales'], bins=np.arange(3)-0.5, edgecolor='black')
    ax.legend(loc='upper right')
    ax.set_xticks([0,1])
    ax.set_title('Comparación de Predicciones y Valores Reales')

    st.pyplot(fig)  # Muestra la figura en Streamlit

    plt.subplots_adjust(hspace = 0.5)  # Agregar espacio entre las dos gráficas

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    # Crear un dataframe para visualizar mejor la matriz de confusión
    cm_df = pd.DataFrame(cm, index=[0,1], columns=[0,1])

    # Crear una figura y un conjunto de ejes
    fig, ax = plt.subplots()

    # Crear un mapa de calor de la matriz de confusión utilizando seaborn
    sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g', ax=ax)
    ax.set_title('Matriz de confusión')
    ax.set_xlabel('Predicciones')
    ax.set_ylabel('Valores reales')

    # Mostrar la figura en Streamlit
    st.pyplot(fig)

    # Añadir menús desplegables para las variables de entrada en la barra lateral
    tipo_usuario = st.sidebar.selectbox('Tipo de Usuario', label_encoders['tipo_usuario'].classes_)
    dia_sem = st.sidebar.selectbox('Día de la Semana', label_encoders['dia_sem'].classes_)
    rango_hora = st.sidebar.selectbox('Rango de Hora', label_encoders['rango_hora'].classes_)

    # Codificar la entrada del usuario
    user_input = pd.DataFrame([[tipo_usuario, dia_sem, rango_hora]])
    for col in range(user_input.shape[1]):
        user_input[col] = label_encoders[variables_relevantes[col]].transform(user_input[col])

    # Realizar una predicción para la entrada del usuario
    user_pred = modelo.predict(user_input)
    
    # Decodificar la predicción antes de desplegarla
    user_pred = ['Ileso' if pred == 1 else 'Herido o Fallecido' for pred in user_pred]
    st.sidebar.write(f'La prediccion para los valores seleccionados es: {user_pred[0]}')
    
    # Crear una gráfica de barras para la predicción del usuario
    fig, ax = plt.subplots()
    ax.bar([0], [1] if user_pred[0] == 'Ileso' else [0], label='Ileso', color='g')
    ax.bar([1], [1] if user_pred[0] == 'Herido o Fallecido' else [0], label='Herido o Fallecido', color='r')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Ileso', 'Herido o Fallecido'])
    ax.set_ylim([0,1])
    ax.set_ylabel('Predicción')
    ax.set_title('Predicción del Modelo para la Entrada del Usuario')
    ax.legend()

    # Mostrar la figura en Streamlit
    st.pyplot(fig)
    
def prediccion():
    st.markdown("# Predicción")
    st.markdown("## Predice si una persona saldría ilesa o no de un accidente")
    
    # Cargar los modelos entrenados
    with open('logistic_regression_model2.pkl', 'rb') as f:
        logistic_regression_model = pickle.load(f)
    with open('decision_tree_model2.pkl', 'rb') as f:
        classification_tree_model = pickle.load(f)

    # Opciones para las variables de entrada
    tipo_usuario_options = sorted(df['tipo_usuario'].unique().tolist())
    dia_sem_options = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    rango_hora_options = [f'{i:02d}:00 a {i:02d}:59' for i in range(24)] + ['No especificado']
    sexo_options = sorted(df['sexo'].unique().tolist())
    rango_edad_options = sorted(df['rango_edad'].unique().tolist())
    ibaen_atro_options = sorted(df['ibaen_atro'].unique().tolist())
    tipo_accidente_options = sorted(df['tipo_accidente'].unique().tolist())

    # Selección de valores para las variables de entrada
    tipo_usuario = st.sidebar.selectbox("Selecciona el tipo de usuario", tipo_usuario_options)
    dia_sem = st.sidebar.selectbox("Selecciona el día de la semana", dia_sem_options)
    rango_hora = st.sidebar.selectbox("Selecciona el rango de hora", rango_hora_options)
    sexo = st.sidebar.selectbox("Selecciona el sexo", sexo_options)
    rango_edad = st.sidebar.selectbox("Selecciona el rango de edad", rango_edad_options)
    ibaen_atro = st.sidebar.selectbox("Selecciona si iba en atropello", ibaen_atro_options)
    tipo_accidente = st.sidebar.selectbox("Selecciona el tipo de accidente", tipo_accidente_options)

    # Codificación de las variables de entrada
    le = LabelEncoder()
    le.fit(tipo_usuario_options)
    tipo_usuario_encoded = le.transform([tipo_usuario])
    le.fit(dia_sem_options)
    dia_sem_encoded = le.transform([dia_sem])
    le.fit(rango_hora_options)
    rango_hora_encoded = le.transform([rango_hora])
    le.fit(sexo_options)
    sexo_encoded = le.transform([sexo])
    le.fit(rango_edad_options)
    rango_edad_encoded = le.transform([rango_edad])
    le.fit(ibaen_atro_options)
    ibaen_atro_encoded = le.transform([ibaen_atro])
    le.fit(tipo_accidente_options)
    tipo_accidente_encoded = le.transform([tipo_accidente])

    le = LabelEncoder()
    le.fit(df['herido'])
    print(le.classes_)

    # Creación de un dataframe con las variables de entrada codificadas
    input_data = pd.DataFrame([tipo_usuario_encoded, dia_sem_encoded, rango_hora_encoded, sexo_encoded, rango_edad_encoded, ibaen_atro_encoded, tipo_accidente_encoded]).T


    # Predicción con los modelos
    logistic_regression_prediction = logistic_regression_model.predict(input_data)
    classification_tree_prediction = classification_tree_model.predict(input_data)

    # Obtener las probabilidades de las predicciones
    logistic_regression_proba = logistic_regression_model.predict_proba(input_data)
    classification_tree_proba = classification_tree_model.predict_proba(input_data)

    # Convertir las predicciones codificadas de vuelta a las etiquetas originales
    original_labels_lr = le.inverse_transform(logistic_regression_prediction)
    original_labels_ct = le.inverse_transform(classification_tree_prediction)

    # Mostrar las predicciones y sus probabilidades
    st.markdown("### Predicciones y sus probabilidades")
    st.markdown(f"*Predicción con regresión logística:* {original_labels_lr[0]}")
    st.markdown(f"*Probabilidad de la predicción con regresión logística:* {max(logistic_regression_proba[0])}")
    st.markdown(f"*Predicción con árbol de clasificación:* {original_labels_ct[0]}")
    st.markdown(f"*Probabilidad de la predicción con árbol de clasificación:* {max(classification_tree_proba[0])}")

    # Imprimir los valores seleccionados y los datos de entrada codificados
    st.markdown("---")
    st.markdown("### Valores seleccionados y datos de entrada codificados")
    st.markdown(f"*Valores seleccionados:* Tipo de usuario = {tipo_usuario}, Día de la semana = {dia_sem}, Rango de hora = {rango_hora}")
    st.markdown(f"*Datos de entrada codificados:* {input_data}")

   # Verificar las predicciones y mostrar los mensajes correspondientes
    st.markdown("---")
    st.markdown("### Resultados de las predicciones")
    if logistic_regression_prediction[0] == 1:
        st.markdown("<span style='color:green'>😁 Modelo de regresión logista predice que esta persona saldría ilesa del accidente</span>", unsafe_allow_html=True)
    elif logistic_regression_prediction == 2:
        st.markdown("<span style='color:red'>😵‍💫 Modelo de regresión logista predice que esta persona saldría herida del accidente</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red'>☠ Modelo de regresión logista predice que esta persona saldría muerta del accidente</span>", unsafe_allow_html=True)

    if classification_tree_prediction[0] == 1:
        st.markdown("<span style='color:green'>😁 Modelo árbol de decisión predice que esta persona saldría ilesa del accidente</span>", unsafe_allow_html=True)
    elif classification_tree_prediction == 2:
        st.markdown("<span style='color:red'>😵‍💫 Modelo árbol de decisión predice que esta persona saldría herida del accidente</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:red'>☠ Modelo árbol de decisión predice que esta persona saldría muerta del accidente</span>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Modelo predictivo de Regresión Logística sobre dileso")
    regresionLog()
    
PAGES = {
    "Inicio": inicio,
    "Datos": datos,
    "Mapa": mapa,
    "Análisis": analisis,
    "Visualizaciones": visualizaciones,
    "Predicción": prediccion
}
def main():
    st.sidebar.title("Navegación")
    selection = st.sidebar.radio("Ir a:", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
