import joblib
import pandas as pd
import streamlit as st

# TEXTOS adaptados a Technicell
TITULO = 'Technicell · Evaluación de reparabilidad'
DESCRIPCION = 'Ingrese las características del dispositivo para estimar su reparabilidad.'
RESULTADO = 'Probabilidad de buena tasa de reparación'
SI_SUPERA = 'Alta reparabilidad'
NO_SUPERA = 'Baja reparabilidad'
ETIQUETAS = {'brand': 'Marca', 'product_type': 'Tipo de producto',
             'top_failure': 'Falla principal', 'fix_rate_sample_size': 'Tamaño de la muestra'}
OPCIONES = {'brand': {'otros': 'Otros'},
            'product_type': {'coffee maker': 'Cafetera', 'hair dryer': 'Secador de cabello', 'headphones': 'Audífonos', 'iron': 'Plancha', 'kettle': 'Hervidor', 'laptop': 'Portátil', 'printer': 'Impresora', 'vacuum': 'Aspiradora', 'smartphone': 'Celular inteligente', 'tablet': 'Tableta', 'sewing machine': 'Máquina de coser', 'sin dato': 'Sin dato'},
            'top_failure': {'screen': 'Pantalla', 'battery': 'Batería', 'connectivity': 'Conectividad', 'electrical': 'Eléctrica', 'heating': 'Sobrecalentamiento', 'mechanical': 'Mecánica', 'motor': 'Motor', 'physical': 'Daño físico', 'software': 'Software'}}
GRUPOS = {}
AVISO = 'Estimación del modelo para fines educativos. El técnico realiza la evaluación final.'

# Cargar el paquete entrenado en Colab.
st.set_page_config(page_title=TITULO)
@st.cache_resource
def cargar_paquete():
    return joblib.load('modelo.joblib')

paquete = cargar_paquete()
modelo = paquete['modelo']
TIPO = paquete.get('tipo', 'clasificacion')
COLUMNAS = paquete['columnas']
PISTAS = paquete.get('pistas', COLUMNAS)
CAMPOS = paquete.get('campos', {})
st.title(TITULO)
st.write(DESCRIPCION)

# Construir el formulario con las pistas originales.
with st.form('caso'):
    valores = {}
    for pista in PISTAS:
        campo = CAMPOS.get(pista, {'tipo': 'numero'})
        texto = ETIQUETAS.get(pista, pista.replace('_', ' ').capitalize())
        if campo['tipo'] == 'opciones':
            valores[pista] = st.selectbox(texto, campo['opciones'],
                format_func=lambda v, p=pista: str(OPCIONES.get(p, {}).get(v, v)))
        elif campo.get('entero', True):
            valores[pista] = st.number_input(texto, min_value=int(campo.get('min', 0)),
                max_value=int(campo.get('max', 10**9)), value=int(campo.get('valor', campo.get('min', 0))), step=1)
        else:
            valores[pista] = st.number_input(texto, min_value=float(campo['min']),
                max_value=float(campo['max']), value=float(campo['valor']))
    enviar = st.form_submit_button('Calcular')

# Codificar igual que en Colab y mostrar la predicción.
if enviar:
    caso = pd.get_dummies(pd.DataFrame([valores]), dtype=int)
    caso = caso.reindex(columns=COLUMNAS, fill_value=0)
    if TIPO == 'clasificacion':
        UMBRAL = paquete.get('umbral', 0.5)  # El Colab aplica 0.5, pero no lo guarda en el paquete.
        p = float(modelo.predict_proba(caso)[0, 1])
        st.metric(RESULTADO, f'{p:.0%}')
        if p > UMBRAL:
            st.success(f'{SI_SUPERA} (supera el umbral de {UMBRAL:.0%}).')
        else:
            st.warning(f'{NO_SUPERA} (no supera el umbral de {UMBRAL:.0%}).')
    elif TIPO == 'regresion':
        valor = float(modelo.predict(caso)[0])
        st.metric(RESULTADO, f'{valor:,.2f}')
    else:
        grupo = int(modelo.predict(caso)[0])
        st.metric(RESULTADO, GRUPOS.get(grupo, f'Grupo {grupo}'))
    st.caption(AVISO)
