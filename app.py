# ==============================================================================
# LIBRERÍAS REQUERIDAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import chardet  as ch # NUEVA LIBRERÍA PARA DETECTAR CODIFICACIÓN

# Configuración de la página de Streamlit (Layout Ancho para mejor visualización de tableros)
st.set_page_config(
    page_title="App Analizadora de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 0. FUNCIONES DE CARGA Y OPTIMIZACIÓN DE MEMORIA (CACHING)
# ==============================================================================
def detectar_codificacion(ruta_archivo):
    """
    Detecta automáticamente la codificación de un archivo.
    CORRECCIÓN: Soluciona el error 'utf-8' codec can't decode byte
    """
    try:
        with open(ruta_archivo, 'rb') as file:
            raw_data = file.read(10000)  # Leer primeros 10KB para detectar codificación
            resultado = chardet.detect(raw_data)
            return resultado['encoding']
    except Exception as e:
        st.warning(f"No se pudo detectar la codificación automáticamente: {e}")
        return 'latin1'  # Fallback a latin1 que es más tolerante

@st.cache_data
def cargar_datos_locales(ruta_archivo):
    """
    Carga un archivo CSV desde el repositorio local de forma eficiente.
    CORRECCIÓN: Manejo automático de diferentes codificaciones (UTF-8, Latin1, etc.)
    """
    try:
        # Intento 1: Detectar codificación automáticamente
        encoding_detectado = detectar_codificacion(ruta_archivo)
        df = pd.read_csv(ruta_archivo, encoding=encoding_detectado)
        return df
    except UnicodeDecodeError:
        try:
            # Intento 2: Probar con latin1 (común en archivos en español)
            df = pd.read_csv(ruta_archivo, encoding='latin1')
            return df
        except UnicodeDecodeError:
            try:
                # Intento 3: Probar con ISO-8859-1
                df = pd.read_csv(ruta_archivo, encoding='ISO-8859-1')
                return df
            except UnicodeDecodeError:
                try:
                    # Intento 4: Probar con cp1252 (Windows)
                    df = pd.read_csv(ruta_archivo, encoding='cp1252')
                    return df
                except Exception as e:
                    st.error(f"Error al cargar el archivo local con múltiples codificaciones: {e}")
                    return None
    except Exception as e:
        st.error(f"Error general al cargar el archivo: {e}")
        return None

@st.cache_data
def cargar_datos_subidos(archivo):
    """
    Carga un archivo CSV subido por el usuario con manejo de codificación.
    CORRECCIÓN: Manejo específico para archivos subidos a Streamlit
    """
    try:
        # Para archivos subidos, leer como bytes y detectar codificación
        contenido = archivo.getvalue()
        encoding_detectado = chardet.detect(contenido[:10000])['encoding']
        df = pd.read_csv(archivo, encoding=encoding_detectado)
        return df
    except UnicodeDecodeError:
        try:
            # Fallback a latin1
            archivo.seek(0)  # Reiniciar el puntero del archivo
            df = pd.read_csv(archivo, encoding='latin1')
            return df
        except Exception as e:
            st.error(f"Error al procesar el archivo subido: {e}")
            return None

def detectar_y_clasificar_variables(df):
    """Identifica automáticamente variables numéricas, categóricas, de fecha y binarias."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Intento de identificación de variables temporales basada en nombres o formatos
    cat_cols_previa = df.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist()
    
    date_cols = []
    cat_cols = []
    bin_cols = []
    
    for col in cat_cols_previa:
        # Si la columna contiene la palabra 'date' o 'fecha' (case insensitive), intentamos clasificarla temporal
        if 'date' in col.lower() or 'fecha' in col.lower() or 'time' in col.lower():
            date_cols.append(col)
        # Si tiene exactamente 2 valores únicos, la consideramos binaria
        elif df[col].nunique() == 2:
            bin_cols.append(col)
            cat_cols.append(col)
        else:
            cat_cols.append(col)
            
    # Verificar si columnas numéricas actúan como binarias (0 y 1 únicamente)
    for col in num_cols:
        if df[col].nunique() == 2 and set(df[col].dropna().unique()).issubset({0, 1}):
            if col not in bin_cols:
                bin_cols.append(col)

    return {
        "num": num_cols,
        "cat": cat_cols,
        "date": date_cols,
        "bin": bin_cols
    }

# ==============================================================================
# 1. GESTIÓN DEL ESTADO DE LA SESIÓN (st.session_state)
# ==============================================================================
# Inicializamos las variables globales de sesión para asegurar la persistencia entre pestañas/módulos.
if 'df_original' not in st.session_state:
    st.session_state['df_original'] = None
if 'df_procesado' not in st.session_state:
    st.session_state['df_procesado'] = None
if 'nombre_dataset' not in st.session_state:
    st.session_state['nombre_dataset'] = ""
if 'columnas_seleccionadas' not in st.session_state:
    st.session_state['columnas_seleccionadas'] = []

# ==============================================================================
# ESPACIO DE MENÚ - SIDEBAR (NAVEGACIÓN GLOBAL)
# ==============================================================================
st.sidebar.title("📌 Panel de Control")
st.sidebar.markdown("---")

# Menú principal de navegación por módulos
opcion_modulo = st.sidebar.selectbox(
    "Selecciona un Módulo:",
    ["1. Home", "2. Carga y Perfil del Dataset", "3. Procesamiento de Datos", "4. Análisis Visual"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Asegúrate de cargar o seleccionar un dataset en el Módulo 2 antes de proceder al análisis.")

# ==============================================================================
# MÓDULO 1: HOME (PRESENTACIÓN Y CONTEXTUALIZACIÓN)
# ==============================================================================
if opcion_modulo == "1. Home":
    st.title("📊 App Analizadora de Datasets con Streamlit")
    st.markdown("### Proyecto Final Integrador - Especialización Python for Analytics")
    st.caption("**Autor:** [Jhoana Gonzales Caballero] | **Año:** 2026")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 🎯 Objetivo del Proyecto")
        st.write(
            "Esta aplicación interactiva ha sido diseñada para servir como una herramienta analítica funcional, "
            "flexible y robusta. Permite realizar un Análisis Exploratorio de Datos (EDA) completo y dinámico "
            "sobre diferentes estructuras y contextos de datos, garantizando una experiencia similar a un producto "
            "de analítica real."
        )
        
        st.markdown("#### 🛠️ Tecnologías Empleadas")
        st.markdown(
            "- **Lenguaje Principal:** Python\n"
            "- **Framework Web Interactivo:** Streamlit\n"
            "- **Procesamiento de Datos:** Pandas y NumPy\n"
            "- **Visualización Interactiva:** Plotly\n"
            "- **Visualización Estática/Avanzada:** Matplotlib y Seaborn\n"
            "- **Control de Versiones y Despliegue:** GitHub & Streamlit Cloud"
        )
        
        st.warning(
            "⚠️ **Nota de Uso Responsable:** Los resultados obtenidos a través de esta aplicación son de carácter "
            "estrictamente exploratorio. No reemplazan ni constituyen una validación clínica, técnica o profesional autorizada."
        )

    with col2:
        st.markdown("#### 📂 Datasets Preconfigurados")
        st.info("**1. AI Impact on Jobs 2030**\n\nImpacto de la inteligencia artificial en empleos, salarios, habilidades y demanda laboral futura.")
        st.info("**2. Sample - Superstore**\n\nRegistro histórico de ventas, pedidos, rentabilidad y comportamiento comercial de una tienda.")
        st.info("**3. E-commerce Order Risk**\n\nEvaluación de pedidos en comercio electrónico enfocados en la detección de fraude y riesgos operativos.")
        st.info("**4. Teen Mental Health**\n\nHábitos digitales y variables de bienestar socioemocional en adolescentes (Enfoque exploratorio).")

# ==============================================================================
# MÓDULO 2: CARGA Y PERFIL DEL DATASET
# ==============================================================================
elif opcion_modulo == "2. Carga y Perfil del Dataset":
    st.title("📂 Carga y Perfil Inicial del Dataset")
    st.markdown("Configure la procedencia de sus datos y visualice la estructura inicial del archivo.")
    st.markdown("---")
    
    # Opciones de carga dual (Predefinidos o Personalizados)
    metodo_carga = st.radio(
        "Seleccione el origen de los datos:",
        ["Usar un Dataset Preconfigurado del Proyecto", "Subir un archivo propio (.CSV)"],
        horizontal=True
    )
    
    df_cargado = None
    nombre_actual = ""
    
    if metodo_carga == "Usar un Dataset Preconfigurado del Proyecto":
        # CORRECCIÓN: Nombre correcto del archivo (AI, no Al)
        seleccion_csv = st.selectbox(
            "Seleccione un archivo preconfigurado:",
            [
                "Seleccione una opción...",
                "AI_Impact_on_Jobs_2030.csv",  # CORREGIDO: "AI" en lugar de "Al"
                "sample_-_superstore.csv",
                "synthetic_ecommerce_order_risk_dataset.csv",
                "Teen_Mental_Health_Dataset.csv"
            ]
        )
        if seleccion_csv != "Seleccione una opción...":
            # Construcción estandarizada de la ruta dentro de la carpeta /data
            ruta_completa = f"data/{seleccion_csv}"
            
            # Mostrar mensaje de carga
            with st.spinner(f"Cargando {seleccion_csv}..."):
                df_cargado = cargar_datos_locales(ruta_completa)
                nombre_actual = seleccion_csv
            
            # Mensaje de éxito o error con información de codificación
            if df_cargado is not None:
                st.success(f"✅ Archivo cargado exitosamente: **{seleccion_csv}**")
            else:
                st.error(f"❌ No se pudo cargar el archivo: {seleccion_csv}")
                st.info("💡 **Sugerencia:** Verifica que el archivo existe en la carpeta 'data/' y no esté corrupto.")
            
    else:
        archivo_subido = st.file_uploader("Suba su archivo en formato CSV", type=["csv"])
        if archivo_subido is not None:
            with st.spinner("Procesando archivo subido..."):
                df_cargado = cargar_datos_subidos(archivo_subido)
                nombre_actual = archivo_subido.name
            
            if df_cargado is not None:
                st.success(f"✅ Archivo subido exitosamente: **{archivo_subido.name}**")
            else:
                st.error("❌ No se pudo procesar el archivo subido. Verifica el formato.")
                
    # Si se cargaron datos exitosamente, los guardamos en el session_state
    if df_cargado is not None:
        st.session_state['df_original'] = df_cargado.copy()
        # Inicialmente el df_procesado es una copia idéntica del original
        if st.session_state['df_procesado'] is None or st.session_state['nombre_dataset'] != nombre_actual:
            st.session_state['df_procesado'] = df_cargado.copy()
        st.session_state['nombre_dataset'] = nombre_actual
        
    # Verificación e impresión del Perfil de Datos si existe un dataframe activo
    if st.session_state['df_original'] is not None:
        df_activo = st.session_state['df_original']
        st.success(f"✔️ Dataset activo actual: **{st.session_state['nombre_dataset']}**")
        
        # Extracción automática de metadatos básicos
        dic_vars = detectar_y_clasificar_variables(df_activo)
        total_nulos = df_activo.isnull().sum().sum()
        total_duplicados = df_activo.duplicated().sum()
        
        # Sección de Métricas Rápidas en Columnas
        st.markdown("### 📊 Métricas de Dimensión y Calidad")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Filas (Registros)", f"{df_activo.shape[0]:,}")
        m2.metric("Columnas (Variables)", df_activo.shape[1])
        m3.metric("Variables Numéricas", len(dic_vars["num"]))
        m4.metric("Variables Categóricas", len(dic_vars["cat"]))
        m5.metric("Valores Nulos", f"{total_nulos:,}")
        m6.metric("Filas Duplicadas", f"{total_duplicados:,}")
        
        st.markdown("---")
        
        # Selección dinámica de columnas para visualización o sub-análisis
        st.markdown("### 👀 Vista Previa y Filtro de Estructura")
        todas_columnas = df_activo.columns.tolist()
        
        columnas_filtro = st.multiselect(
            "Seleccione columnas específicas para previsualizar (Deje vacío para ver todas):",
            options=todas_columnas,
            default=[]
        )
        
        df_mostrar = df_activo[columnas_filtro] if columnas_filtro else df_activo
        
        # Muestra una tabla con los primeros registros
        st.dataframe(df_mostrar.head(10), use_container_width=True)
        
        # Información técnica detallada de tipos de datos
        with st.expander("🔍 Ver Detalle de Columnas y Tipos de Datos Técnicos"):
            info_df = pd.DataFrame({
                "Tipo de Dato": df_activo.dtypes.astype(str),
                "Valores No Nulos": df_activo.notnull().sum(),
                "Valores Nulos": df_activo.isnull().sum(),
                "% Nulos": (df_activo.isnull().sum() / len(df_activo) * 100).round(2),
                "Valores Únicos": df_activo.nunique()
            })
            st.dataframe(info_df, use_container_width=True)
            
            # Alertas en caso de datasets limitados o problemáticos
            if len(dic_vars["num"]) == 0:
                st.warning("⚠️ Este dataset no contiene variables numéricas detectadas automáticamente.")
            if len(dic_vars["cat"]) == 0:
                st.warning("⚠️ Este dataset no contiene variables categóricas detectadas automáticamente.")
    else:
        st.info("ℹ️ Por favor, cargue un archivo o seleccione un dataset predefinido para comenzar.")
        st.markdown("""
        ### 💡 Solución al error de codificación:
        Si encuentras errores de codificación (`utf-8` codec can't decode byte), el sistema ahora:
        1. **Detecta automáticamente** la codificación del archivo
        2. **Prueba múltiples codificaciones** (latin1, ISO-8859-1, cp1252)
        3. **Muestra mensajes específicos** para ayudar a diagnosticar el problema
        
        **Recomendación:** Asegúrate que los archivos CSV estén guardados en UTF-8 o Latin1.
        """)

# ==============================================================================
# MÓDULO 3: PROCESAMIENTO DE DATOS
# ==============================================================================
elif opcion_modulo == "3. Procesamiento de Datos":
    st.title("🛠️ Procesamiento y Limpieza Flexible de Datos")
    st.markdown("Detecte inconsistencias, limpie estructuras, estandarice formatos y aplique filtros dinámicos.")
    st.markdown("---")
    
    if st.session_state['df_original'] is not None:
        # Trabajamos sobre la copia procesada para permitir transformaciones iterativas
        df_proc = st.session_state['df_procesado']
        
        st.info(f"Trabajando sobre el dataset: **{st.session_state['nombre_dataset']}**")
        
        # ACOPLAMIENTO 1: Estandarización de nombres de columnas
        st.markdown("### 1. Estandarización de Columnas")
        col_est1, col_est2 = st.columns(2)
        with col_est1:
            if st.button("🧼 Estandarizar Nombres (Quitar espacios y pasar a Minúsculas)"):
                df_proc.columns = df_proc.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.lower()
                st.session_state['df_procesado'] = df_proc
                st.success("Columnas estandarizadas con éxito.")
                st.rerun()
        with col_est2:
            if st.button("🔄 Resetear a Dataset Original"):
                st.session_state['df_procesado'] = st.session_state['df_original'].copy()
                st.success("Se han revertido todos los cambios al estado inicial.")
                st.rerun()
                
        st.markdown("---")
        
        # ACOPLAMIENTO 2: Conversión Automática de Fechas
        st.markdown("### 2. Conversión y Detección de Fechas")
        dic_vars = detectar_y_clasificar_variables(df_proc)
        
        if dic_vars["date"]:
            st.write("Columnas sospechosas de ser fechas basadas en su nombre:", dic_vars["date"])
            col_conversion = st.selectbox("Seleccione columna para forzar conversión a Tipo Fecha (Datetime):", ["Ninguna"] + df_proc.columns.tolist())
            if col_conversion != "Ninguna":
                if st.button(f"Convertir '{col_conversion}' a Datetime"):
                    df_proc[col_conversion] = pd.to_datetime(df_proc[col_conversion], errors='coerce')
                    st.session_state['df_procesado'] = df_proc
                    st.success(f"Columna '{col_conversion}' convertida correctamente aplicando errores='coerce'.")
                    st.rerun()
        else:
            st.info("No se detectaron columnas con patrones explícitos de fecha. Puede forzar una conversión abajo si lo desea:")
            col_conversion = st.selectbox("Forzar columna a Tipo Fecha:", ["Ninguna"] + df_proc.columns.tolist())
            if col_conversion != "Ninguna":
                if st.button(f"Convertir '{col_conversion}'"):
                    df_proc[col_conversion] = pd.to_datetime(df_proc[col_conversion], errors='coerce')
                    st.session_state['df_procesado'] = df_proc
                    st.success(f"Columna '{col_conversion}' transformada.")
                    st.rerun()

        st.markdown("---")

        # ACOPLAMIENTO 3: Tratamiento de Valores Faltantes y Duplicados
        st.markdown("### 3. Integridad de Datos (Nulos y Duplicados)")
        c_limp1, c_limp2 = st.columns(2)
        
        with c_limp1:
            num_duplicados = df_proc.duplicated().sum()
            st.metric("Registros duplicados totales", num_duplicados)
            if num_duplicados > 0:
                if st.button("❌ Eliminar Filas Duplicadas"):
                    df_proc = df_proc.drop_duplicates()
                    st.session_state['df_procesado'] = df_proc
                    st.success("Duplicados removidos.")
                    st.rerun()
                    
        with c_limp2:
            st.write("**Resumen de Valores Faltantes Críticos:**")
            nulos_por_col = df_proc.isnull().sum()
            nulos_filtrados = nulos_por_col[nulos_por_col > 0]
            if not nulos_filtrados.empty:
                st.dataframe(pd.DataFrame({"Nulos": nulos_filtrados, "% del Total": (nulos_filtrados/len(df_proc)*100).round(2)}))
                if st.button("🩹 Imputar Nulos Numéricos con la Mediana"):
                    for c in df_proc.select_dtypes(include=[np.number]).columns:
                        df_proc[c] = df_proc[c].fillna(df_proc[c].median())
                    st.session_state['df_procesado'] = df_proc
                    st.success("Valores numéricos faltantes reemplazados por su mediana.")
                    st.rerun()
            else:
                st.success("🎉 ¡Perfecto! El dataset no presenta valores nulos en ninguna columna.")

        st.markdown("---")

        # ACOPLAMIENTO 4: Identificación Avanzada de Outliers (IQR)
        st.markdown("### 4. Detección de Outliers (Valores Atípicos)")
        columnas_numericas = df_proc.select_dtypes(include=[np.number]).columns.tolist()
        
        if columnas_numericas:
            col_outlier = st.selectbox("Seleccione variable numérica para calcular Outliers bajo regla IQR:", columnas_numericas)
            
            # Computación del Rango Intercuartílico (IQR)
            q1 = df_proc[col_outlier].quantile(0.25)
            q3 = df_proc[col_outlier].quantile(0.75)
            iqr = q3 - q1
            limite_inferior = q1 - 1.5 * iqr
            limite_superior = q3 + 1.5 * iqr
            
            outliers = df_proc[(df_proc[col_outlier] < limite_inferior) | (df_proc[col_outlier] > limite_superior)]
            
            o_col1, o_col2 = st.columns([1, 2])
            with o_col1:
                st.metric("Cantidad de Outliers Detectados", len(outliers))
                st.write(f"**Límite Inferior:** {limite_inferior:.2f}")
                st.write(f"**Límite Superior:** {limite_superior:.2f}")
            with o_col2:
                # Boxplot interactivo rápido para validar la distribución visual de outliers
                fig_box = px.box(df_proc, y=col_outlier, title=f"Boxplot de {col_outlier} (Identificación Visual)", points="outliers")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No hay columnas numéricas disponibles para análisis de outliers.")
            
        st.markdown("---")
        
        # ACOPLAMIENTO 5: Segmentación y Filtros Dinámicos de Control
        st.markdown("### 5. Filtros Dinámicos Globales Aplicados al Dataset")
        st.write("Configure un rango de recorte opcional para explorar el comportamiento de los datos bajo condiciones específicas.")
        
        cat_para_filtrar = df_proc.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_para_filtrar:
            filtro_col = st.selectbox("Filtrar interactivamente por columna categórica:", ["Ninguna"] + cat_para_filtrar)
            if filtro_col != "Ninguna":
                valores_unicos = df_proc[filtro_col].dropna().unique().tolist()
                seleccion_valores = st.multiselect(f"Seleccione categorías válidas de [{filtro_col}]:", opciones=valores_unicos, default=valores_unicos[:3] if len(valores_unicos)>3 else valores_unicos)
                if seleccion_valores:
                    df_filtrado_final = df_proc[df_proc[filtro_col].isin(seleccion_valores)]
                    st.write(f"Filas resultantes tras aplicar el filtro: **{df_filtrado_final.shape[0]}**")
                    st.checkbox("¿Desea guardar este sub-filtro como el dataset de análisis actual?", key="guardar_filtro")
                    if st.session_state.guardar_filtro:
                        st.session_state['df_procesado'] = df_filtrado_final
    else:
        st.error("❌ No hay datos cargados en el sistema. Diríjase al Módulo 2.")

# ==============================================================================
# MÓDULO 4: ANÁLISIS VISUAL OBLIGATORIO
# ==============================================================================
elif opcion_modulo == "4. Análisis Visual":
    st.title("📈 Núcleo Analítico Visual")
    st.markdown("Explore distribuciones, relaciones multivariadas, tendencias y obtenga conclusiones operativas efectivas.")
    st.markdown("---")
    
    if st.session_state['df_procesado'] is not None:
        df_analisis = st.session_state['df_procesado']
        
        # Clasificación en caliente de tipos de datos para construir las interfaces de gráficos
        dic_v = detectar_y_clasificar_variables(df_analisis)
        
        # Inicialización obligatoria de las 6 pestañas sugeridas por la rúbrica
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗂️ Tab 1: Resumen", 
            "📊 Tab 2: Univariado", 
            "🔄 Tab 3: Bivariado", 
            "🕸️ Tab 4: Multivariado", 
            "📅 Tab 5: Temporal", 
            "💡 Tab 6: Insights Específicos"
        ])
        
        # ----------------------------------------------------------------------
        # TAB 1: RESUMEN
        # ----------------------------------------------------------------------
        with tab1:
            st.markdown("### Resumen Estadístico y Descriptivo de las Variables")
            
            c_res1, c_res2 = st.columns(2)
            with c_res1:
                st.markdown("**Estructura Actual del Dataframe:**")
                st.write(f"- **Filas:** {df_analisis.shape[0]}")
                st.write(f"- **Columnas:** {df_analisis.shape[1]}")
            with c_res2:
                mostrar_descripcion = st.checkbox("Mostrar Descripción Estadística Detallada (Describe)", value=True)
                
            if mostrar_descripcion:
                if dic_v["num"]:
                    st.markdown("#### Variables Numéricas")
                    st.dataframe(df_analisis.describe().T, use_container_width=True)
                if [c for c in df_analisis.columns if c not in dic_v["num"]]:
                    st.markdown("#### Variables Categóricas / Objetos")
                    st.dataframe(df_analisis.describe(include=['O', 'category', 'boolean']).T, use_container_width=True)
                    
        # ----------------------------------------------------------------------
        # TAB 2: ANÁLISIS UNIVARIADO
        # ----------------------------------------------------------------------
        with tab2:
            st.markdown("### Análisis Univariado (Distribución Individual)")
            
            var_seleccionada = st.selectbox("Seleccione la variable a analizar individualmente:", df_analisis.columns.tolist())
            
            col_uni1, col_uni2 = st.columns([2, 1])
            
            with col_uni1:
                # Comportamiento dinámico según el tipo de variable
                if var_seleccionada in dic_v["num"]:
                    # Visualización Interactiva con Plotly
                    opcion_grafico = st.radio("Tipo de Gráfico Numérico:", ["Histograma de Frecuencias", "Boxplot de Dispersión"], horizontal=True)
                    if opcion_grafico == "Histograma de Frecuencias":
                        fig = px.histogram(df_analisis, x=var_seleccionada, nbins=30, title=f"Distribución de {var_seleccionada}", color_discrete_sequence=['#1f77b4'])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.box(df_analisis, y=var_seleccionada, title=f"Diagrama de Caja de {var_seleccionada}", color_discrete_sequence=['#2ca02c'])
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # Distribución Categórica
                    conteo_cats = df_analisis[var_seleccionada].value_counts().reset_index()
                    conteo_cats.columns = [var_seleccionada, 'Cantidad']
                    fig = px.bar(conteo_cats, x=var_seleccionada, y='Cantidad', title=f"Frecuencia de Categorías para {var_seleccionada}", text_auto=True, color='Cantidad')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col_uni2:
                st.markdown("#### 📝 Interpretación")
                st.write(f"Analizando la columna: **{var_seleccionada}**")
                st.write(f"- Valores únicos identificados: {df_analisis[var_seleccionada].nunique()}")
                st.write(f"- Total registros válidos (No nulos): {df_analisis[var_seleccionada].notnull().sum()}")
                st.markdown("> *Utilice esta sección para identificar asimetrías en las variables, concentraciones de datos o modas marcadas dentro de los grupos categóricos.*")

        # ----------------------------------------------------------------------
        # TAB 3: ANÁLISIS BIVARIADO
        # ----------------------------------------------------------------------
        with tab3:
            st.markdown("### Análisis Bivariado (Cruce de Variables)")
            
            biv_x = st.selectbox("Seleccione la Variable X (Independiente / Agrupadora):", df_analisis.columns.tolist(), index=0)
            biv_y = st.selectbox("Seleccione la Variable Y (Dependiente / Métrica):", df_analisis.columns.tolist(), index=min(1, len(df_analisis.columns)-1))
            
            # Determinación de flujos lógicos para graficar el cruce correcto
            if biv_x in dic_v["num"] and biv_y in dic_v["num"]:
                st.markdown("#### Gráfico de Dispersión (Numérico vs Numérico)")
                fig_bi = px.scatter(df_analisis, x=biv_x, y=biv_y, trendline="ols", title=f"Relación entre {biv_x} y {biv_y}")
                st.plotly_chart(fig_bi, use_container_width=True)
            elif biv_x in dic_v["cat"] and biv_y in dic_v["num"]:
                st.markdown("#### Gráfico de Cajas por Categoría (Categórico vs Numérico)")
                fig_bi = px.box(df_analisis, x=biv_x, y=biv_y, color=biv_x, title=f"Distribución de {biv_y} indexado por {biv_x}")
                st.plotly_chart(fig_bi, use_container_width=True)
            elif biv_x in dic_v["num"] and biv_y in dic_v["cat"]:
                st.markdown("#### Gráfico de Cajas Horizontal (Numérico vs Categórico)")
                fig_bi = px.box(df_analisis, x=biv_x, y=biv_y, color=biv_y, orientation='h', title=f"Distribución de {biv_x} por {biv_y}")
                st.plotly_chart(fig_bi, use_container_width=True)
            else:
                st.markdown("#### Barras Agrupadas (Categórico vs Categórico)")
                df_bi_cat = df_analisis.groupby([biv_x, biv_y]).size().reset_index(name='Conteo')
                fig_bi = px.bar(df_bi_cat, x=biv_x, y='Conteo', color=biv_y, barmode='group', title=f"Cruce de frecuencias entre {biv_x} y {biv_y}")
                st.plotly_chart(fig_bi, use_container_width=True)
                
            st.caption(f"Visualización bivariada dinámica calculada para {biv_x} frente a {biv_y}.")

        # ----------------------------------------------------------------------
        # TAB 4: ANÁLISIS MULTIVARIADO
        # ----------------------------------------------------------------------
        with tab4:
            st.markdown("### Análisis Multivariado y Matrices de Correlación")
            
            c_mul1, c_mul2 = st.columns([3, 1])
            
            with c_mul1:
                if len(dic_v["num"]) >= 2:
                    st.markdown("#### 🔥 Mapa de Calor de Correlaciones (Matplotlib & Seaborn)")
                    # Uso explícito de Seaborn y Matplotlib solicitado por los requisitos del proyecto
                    fig_m, ax_m = plt.subplots(figsize=(8, 5))
                    matriz_corr = df_analisis[dic_v["num"]].corr()
                    sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_m, linewidths=0.5)
                    plt.title("Matriz de Correlación de Pearson")
                    st.pyplot(fig_m)
                else:
                    st.info("Se requieren al menos 2 variables numéricas en el dataset para estructurar la matriz de correlación.")
                    
            with c_mul2:
                st.markdown("#### ⚙️ Segmentación por Color (3 Variables)")
                if len(dic_v["num"]) >= 2 and dic_v["cat"]:
                    mul_x = st.selectbox("Eje X (Num):", dic_v["num"], key="mx")
                    mul_y = st.selectbox("Eje Y (Num):", dic_v["num"], key="my")
                    mul_color = st.selectbox("Color (Cat):", dic_v["cat"], key="mc")
                    
                    fig_mul = px.scatter(df_analisis, x=mul_x, y=mul_y, color=mul_color, title="Dispersión Multivariada")
                    st.plotly_chart(fig_mul, use_container_width=True)
                else:
                    st.text("Faltan variables para habilitar la segmentación combinada de 3 variables.")

        # ----------------------------------------------------------------------
        # TAB 5: ANÁLISIS TEMPORAL
        # ----------------------------------------------------------------------
        with tab5:
            st.markdown("### Análisis de Evolución Temporal")
            
            # Buscamos de manera exhaustiva si existen columnas parseadas como fecha
            columnas_fecha_reales = df_analisis.select_dtypes(include=[np.datetime64]).columns.tolist()
            
            if columnas_fecha_reales:
                col_tiempo = st.selectbox("Seleccione la variable temporal (Eje X):", columnas_fecha_reales)
                
                if dic_v["num"]:
                    col_metrica_tiempo = st.selectbox("Seleccione la métrica numérica a evaluar en el tiempo (Eje Y):", dic_v["num"])
                    
                    # Agrupación temporal por fecha para suavizar curvas
                    df_temporal = df_analisis.groupby(col_tiempo)[col_metrica_tiempo].mean().reset_index()
                    
                    fig_linea = px.line(df_temporal, x=col_tiempo, y=col_metrica_tiempo, title=f"Evolución Promedio de {col_metrica_tiempo} a lo largo del tiempo")
                    st.plotly_chart(fig_linea, use_container_width=True)
                else:
                    st.warning("No existen variables numéricas para trazar la métrica en la línea de tiempo.")
            else:
                st.info("📅 **Análisis Temporal Desactivado:** Este dataset no cuenta actualmente con columnas en formato de fecha (Datetime).")
                st.markdown("💡 *Tip: Puedes ir al 'Módulo 3: Procesamiento de Datos' y forzar la conversión de una columna a tipo fecha utilizando el parser automático.*")

        # ----------------------------------------------------------------------
        # TAB 6: INSIGHTS ESPECÍFICOS (ENTREGABLES SUGERIDOS POR EL CASO)
        # ----------------------------------------------------------------------
        with tab6:
            st.markdown("### 💡 Insights y Gráficos Sugeridos Automáticos")
            st.markdown("Detección inteligente del dataset cargado para construir visualizaciones contextuales específicas de negocio.")
            
            nombre_file = st.session_state['nombre_dataset'].lower()
            
            # DETECCIÓN DATASET 1: AI Impact on Jobs
            if "ai_impact" in nombre_file or "jobs" in nombre_file:
                st.success("🎯 Dataset Detectado: **AI Impact on Jobs 2030**")
                
                # Caso: Salario vs Riesgo de Reemplazo
                if "Average_Salary_USD" in df_analisis.columns and "AI_Replacement_Risk" in df_analisis.columns:
                    st.markdown("#### Cruce Clave: Riesgo de Reemplazo por IA vs Salario Promedio")
                    fig_ai = px.scatter(df_analisis, x="AI_Replacement_Risk", y="Average_Salary_USD", color="Industry" if "Industry" in df_analisis.columns else None, title="Análisis de Salarios Indexados al Riesgo Tecnológico")
                    st.plotly_chart(fig_ai, use_container_width=True)
                    st.write("**Insight:** Permite identificar si los puestos con remuneraciones más competitivas se encuentran resguardados o expuestos a una automatización acelerada.")

            # DETECCIÓN DATASET 2: Superstore
            elif "superstore" in nombre_file or "store" in nombre_file:
                st.success("🎯 Dataset Detectado: **Sample - Superstore (Ventas)**")
                
                # Ventas por Categoría y Utilidad por Región
                c_store1, c_store2 = st.columns(2)
                with c_store1:
                    if "Category" in df_analisis.columns and "Sales" in df_analisis.columns:
                        fig_st1 = px.bar(df_analisis.groupby("Category")["Sales"].sum().reset_index(), x="Category", y="Sales", title="Ventas Totales por Categoría de Producto", text_auto=True)
                        st.plotly_chart(fig_st1, use_container_width=True)
                with c_store2:
                    if "Region" in df_analisis.columns and "Profit" in df_analisis.columns:
                        fig_st2 = px.box(df_analisis, x="Region", y="Profit", title="Rentabilidad y Márgenes Operativos por Región")
                        st.plotly_chart(fig_st2, use_container_width=True)
                st.write("**Insight Operativo:** Evalúa la rentabilidad real comparando si los altos volúmenes de venta en ciertas regiones se traducen verdaderamente en utilidad neta para la tienda.")

            # DETECCIÓN DATASET 3: E-commerce Risk
            elif "ecommerce" in nombre_file or "risk" in nombre_file:
                st.success("🎯 Dataset Detectado: **Synthetic E-commerce Order Risk Dataset**")
                
                if "risk_label" in df_analisis.columns or "is_fraud" in df_analisis.columns:
                    target_risk = "risk_label" if "risk_label" in df_analisis.columns else "is_fraud"
                    st.markdown(f"#### Distribución de Alertas de Fraude y Riesgo ({target_risk})")
                    
                    c_risk1, c_risk2 = st.columns([2, 1])
                    with c_risk1:
                        if "payment_method" in df_analisis.columns:
                            fig_risk = px.histogram(df_analisis, x="payment_method", color=target_risk, barmode="group", title="Nivel de Riesgo según el Método de Pago Empleado")
                            st.plotly_chart(fig_risk, use_container_width=True)
                    with c_risk2:
                        st.write("**Análisis Estratégico de Fraude:**")
                        st.dataframe(df_analisis[target_risk].value_counts(normalize=True).round(4)*100)
                        st.write("Identificar qué pasarelas de pago concentran la mayor densidad de reclamos o contracargos permite ajustar las reglas de validación en tiempo real.")

            # DETECCIÓN DATASET 4: Teen Mental Health
            elif "teen" in nombre_file or "mental" in nombre_file or "health" in nombre_file:
                st.success("🎯 Dataset Detectado: **Teen Mental Health Dataset**")
                
                if "daily_social_media_hours" in df_analisis.columns and "sleep_hours" in df_analisis.columns:
                    st.markdown("#### Relación: Horas de Redes Sociales vs Horas de Sueño")
                    fig_teen = px.scatter(df_analisis, x="daily_social_media_hours", y="sleep_hours", color="stress_level" if "stress_level" in df_analisis.columns else None, title="Exploración de Hábitos Digitales y Descanso")
                    st.plotly_chart(fig_teen, use_container_width=True)
                    st.write("**Interpretación:** Espacio orientado a observar correlaciones de comportamiento general. Evitar emitir juicios o diagnósticos de carácter clínico basados exclusivamente en esta gráfica.")
                    
            else:
                st.info("Dataset personalizado detectado. Para activar gráficos preconfigurados de negocio, cargue uno de los 4 archivos oficiales estipulados en la rúbrica del proyecto.")
                
            st.markdown("---")
            st.markdown("#### 🏁 Conclusión General para la Toma de Decisiones")
            st.markdown("> *Las visualizaciones e interacciones ejecutadas demuestran que el comportamiento de los datos varía drásticamente según el contexto del negocio. Una correcta limpieza de datos realizada previamente en el Módulo 3 asegura que los patrones gráficos visualizados aquí correspondan a realidades operativas fiables.*")
            
    else:
        st.error("❌ No se registran datos listos para procesar de forma visual. Por favor, cargue un archivo válido en el Módulo 2.")
