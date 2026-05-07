import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score
from scipy import stats

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(page_title="Conductividad Térmica - FTIQ", layout="wide")

st.markdown("""
    <style>
    .intermedio { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0056b3; font-family: monospace; font-size: 14px;}
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# BASE DE DATOS DE REFERENCIA
# ==============================================================================
COMPUESTOS_INFO = {
    "Hidrógeno": {"Formula": "H2", "M": 0.002016, "M_gmol": 2.016},
    "Metano": {"Formula": "CH4", "M": 0.016043, "M_gmol": 16.043},
    "Etano": {"Formula": "C2H6", "M": 0.030070, "M_gmol": 30.070},
    "Etileno": {"Formula": "C2H4", "M": 0.028054, "M_gmol": 28.054},
    "Acetileno": {"Formula": "C2H2", "M": 0.026038, "M_gmol": 26.038},
    "Propileno": {"Formula": "C3H6", "M": 0.042081, "M_gmol": 42.081},
    "Propano": {"Formula": "C3H8", "M": 0.044097, "M_gmol": 44.097},
    "n-Butano": {"Formula": "C4H10", "M": 0.058123, "M_gmol": 58.123}
}

PROPIEDADES_CRITICAS = {
    "Hidrógeno": {"Tc": 32.98, "Pc_bar": 12.93, "w": -0.217},
    "Metano": {"Tc": 190.56, "Pc_bar": 45.98, "w": 0.011},
    "Etano": {"Tc": 282.34, "Pc_bar": 50.41, "w": 0.099},
    "Etileno": {"Tc": 305.32, "Pc_bar": 48.72, "w": 0.087},
    "Acetileno": {"Tc": 308.30, "Pc_bar": 61.14, "w": 0.189},
    "Propileno": {"Tc": 364.90, "Pc_bar": 46.00, "w": 0.142},
    "Propano": {"Tc": 369.83, "Pc_bar": 42.48, "w": 0.152},
    "n-Butano": {"Tc": 425.12, "Pc_bar": 37.96, "w": 0.200}
}

T_EBULLICION = {
    "Hidrógeno": 20.28,
    "Metano": 111.66,
    "Etano": 184.55,
    "Etileno": 169.35,
    "Acetileno": 189.15,
    "Propileno": 225.50,
    "Propano": 231.10,
    "n-Butano": 272.66
}

# Datos experimentales de la mezcla de cracking (Tabla 9 del documento)
COMPOSICION_CRACKING = {
    "Hidrógeno": 0.3961,
    "Metano": 0.1384,
    "Etano": 0.1359,
    "Etileno": 0.2593,
    "Acetileno": 0.0016,
    "Propileno": 0.0019,
    "Propano": 0.0001,
    "n-Butano": 0.0667
}

LAMBDA_EXP_PUROS = {
    "Hidrógeno": 0.1865,
    "Metano": 0.0362,
    "Etano": 0.0222,
    "Etileno": 0.02318,
    "Acetileno": 0.0213,
    "Propileno": 0.01764,
    "Propano": 0.018826,
    "n-Butano": 0.016747
}

# ==============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ==============================================================================
if 'tabla_puras' not in st.session_state:
    st.session_state.tabla_puras = pd.DataFrame(columns=[
        "Componente", "Modelo", "T (K)", "P (MPa)", "Exp (W/m·K)", 
        "Calc (W/m·K)", "Error (%)"
    ])

if 'tabla_mezclas_wass' not in st.session_state:
    st.session_state.tabla_mezclas_wass = pd.DataFrame(columns=[
        "T (K)", "lambda_calc (W/m·K)"
    ])

if 'tabla_mezclas_lindsay' not in st.session_state:
    st.session_state.tabla_mezclas_lindsay = pd.DataFrame(columns=[
        "T (K)", "lambda_calc (W/m·K)"
    ])

# ==============================================================================
# FUNCIONES DE LOS MODELOS PARA COMPUESTOS PUROS
# ==============================================================================

def eucken_modificado(eta, Cp, Cv, M, R=8.314):
    """
    Método de Eucken Modificado
    eta: viscosidad dinámica (N·s/m² = Pa·s)
    """
    lambda_calc = (eta * Cv / M) * (1.32 + 1.77 / (Cv / R))
    return lambda_calc

def chung_et_al(eta, Cv, M, T, Tc, w, R=8.314):
    """
    Método de Chung et al.
    """
    Tr = T / Tc
    alpha = Cv / R - 1.5
    beta = 0.7862 - 0.7109 * w + 1.3168 * w**2
    Z = 2.0 + 10.5 * Tr**2
    
    numerador = 0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z
    denominador = 0.6366 + beta * Z + 1.061 * alpha * beta
    Psi = 1 + alpha * (numerador / denominador)
    
    lambda_calc = (3.75 * Psi * eta * Cv) / (M * (Cv / R))
    return lambda_calc

# ==============================================================================
# FUNCIONES DE REGLAS DE MEZCLA (CORREGIDAS SEGÚN DOCUMENTO)
# ==============================================================================

def gamma_roy_thodos(Tc, Pc_bar, M_gmol):
    """
    Cálculo de Γ (inversa de conductividad reducida) - Ecuación del documento
    Γ = 210 * (Tc^2 * M^3 / Pc^4)^(1/6)
    """
    Pc_atm = Pc_bar * 0.986923  # Convertir bar a atm
    gamma = 210 * ((Tc**2 * M_gmol**3) / (Pc_atm**4))**(1/6)
    return gamma

def lambda_tr_ratio(T, Tc_i, Tc_j, gamma_i, gamma_j):
    """
    Relación de conductividades translacionales según Roy y Thodos
    λtr,i/λtr,j = (Γj/Γi) * [exp(0.0464*Tri)-exp(-0.2412*Tri)] / [exp(0.0464*Trj)-exp(-0.2412*Trj)]
    """
    Tri = T / Tc_i
    Trj = T / Tc_j
    f_i = np.exp(0.0464 * Tri) - np.exp(-0.2412 * Tri)
    f_j = np.exp(0.0464 * Trj) - np.exp(-0.2412 * Trj)
    ratio = (gamma_j * f_i) / (gamma_i * f_j)
    return ratio

def wassiljewa_mason_saxena(y, lambda_i, M_gmol, Tc, Pc_bar, T):
    """
    Regla de mezcla de Wassiljewa con modificación de Mason y Saxena
    Según ecuaciones del documento (páginas 23-25)
    """
    n = len(y)
    # Calcular Gamma para cada componente
    gamma = {}
    for i in range(n):
        gamma[i] = gamma_roy_thodos(Tc[i], Pc_bar[i], M_gmol[i])
    
    # Construir matriz de interacción A_ij
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio_lambda_tr = lambda_tr_ratio(T, Tc[i], Tc[j], gamma[i], gamma[j])
                # A_ij = [1 + (λtr,i/λtr,j)^0.5 * (Mj/Mi)^0.25]^2 / [8(1 + Mi/Mj)]^0.5
                term = (1 + np.sqrt(ratio_lambda_tr) * (M_gmol[j] / M_gmol[i])**0.25)**2
                denom = np.sqrt(8 * (1 + M_gmol[i] / M_gmol[j]))
                A[i, j] = term / denom
    
    # Calcular conductividad de la mezcla
    lambda_m = 0
    for i in range(n):
        if y[i] > 0 and lambda_i[i] > 0:
            numerador = y[i] * lambda_i[i]
            denominador = sum(y[j] * A[i, j] for j in range(n))
            if denominador > 0:
                lambda_m += numerador / denominador
    return lambda_m

def lindsay_bromley(y, lambda_i, M_gmol, Cp_i, gamma_i, Tb_lista, T):
    """
    Regla de mezcla de Lindsay y Bromley SEGÚN EL DOCUMENTO WORD
    Ecuaciones completas de la página 33-35
    """
    n = len(y)
    R = 8.314
    
    # Constante de Sutherland: S_i = 1.5 * Tb,i
    S_i = 1.5 * np.array(Tb_lista)
    
    # Calcular viscosidades usando relación inversa de Eucken modificada
    eta_i = np.zeros(n)
    for i in range(n):
        Cv_i = Cp_i[i] / gamma_i[i]
        factor_eucken = 1.32 + 1.77 / (Cv_i / R)
        M_kg = M_gmol[i] / 1000
        if lambda_i[i] > 0 and Cv_i > 0:
            eta_i[i] = (lambda_i[i] * M_kg) / (Cv_i * factor_eucken)
        else:
            eta_i[i] = 1e-6  # Valor pequeño por defecto
    
    # Construir matriz de interacción A_ij
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and eta_i[j] > 0 and lambda_i[i] > 0 and lambda_i[j] > 0:
                # Cociente de viscosidades según Eucken modificado
                # μi/μj = (λi/λj) * (Cp,j/Cp,i) * (9 - 5/γi)/(9 - 5/γj)
                visc_ratio = (lambda_i[i] / lambda_i[j]) * (Cp_i[j] / Cp_i[i])
                visc_ratio *= (9 - 5/gamma_i[i]) / (9 - 5/gamma_i[j])
                
                # Factor de masa molecular (Mj/Mi)^(3/4)
                mass_factor = (M_gmol[j] / M_gmol[i])**0.75
                
                # Factor de Sutherland [(1 + Si/T)/(1 + Sj/T)]^(1/2)
                suth_factor = ((1 + S_i[i]/T) / (1 + S_i[j]/T))**0.5
                
                # Término entre corchetes
                bracket = visc_ratio * mass_factor * suth_factor
                
                # Constante de Sutherland cruzada: Sij = sqrt(Si * Sj)
                S_ij = np.sqrt(S_i[i] * S_i[j])
                
                # A_ij final según ecuación (10.53) del documento
                A[i, j] = 0.25 * (1 + bracket**2) * ((1 + S_ij/T) / (1 + S_i[i]/T))
    
    # Calcular conductividad de la mezcla
    lambda_m = 0
    for i in range(n):
        if y[i] > 0 and lambda_i[i] > 0:
            numerador = y[i] * lambda_i[i]
            denominador = sum(y[j] * A[i, j] for j in range(n))
            if denominador > 0:
                lambda_m += numerador / denominador
    return lambda_m

# ==============================================================================
# FUNCIÓN PARA REGRESIÓN LINEAL EN GRÁFICAS
# ==============================================================================

def linear_regression_plot(x, y, xlabel, ylabel, title):
    """
    Crea un gráfico con regresión lineal y muestra la ecuación
    """
    # Regresión lineal
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r2 = r_value**2
    
    # Línea de regresión
    x_lineal = np.linspace(min(x), max(x), 100)
    y_lineal = slope * x_lineal + intercept
    
    fig = go.Figure()
    
    # Puntos experimentales
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers',
        marker=dict(size=10, color='#0056b3'),
        name='Datos'
    ))
    
    # Línea de regresión (punteada)
    fig.add_trace(go.Scatter(
        x=x_lineal, y=y_lineal,
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name=f'Regresión: y = {slope:.4f}x + {intercept:.4f}'
    ))
    
    # Línea ideal (x=y) solo si aplica
    if title.__contains__("Experimental vs Calculada"):
        val_min = min(min(x), min(y)) * 0.95
        val_max = max(max(x), max(y)) * 1.05
        fig.add_trace(go.Scatter(
            x=[val_min, val_max], y=[val_min, val_max],
            mode='lines',
            line=dict(color='green', dash='dot', width=1.5),
            name='Línea Ideal (Exp = Calc)'
        ))
    
    fig.update_layout(
        title=f"{title}<br><sub>Ecuación: y = {slope:.6f}x + {intercept:.6f} | R² = {r2:.6f}</sub>",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white"
    )
    
    return fig, slope, intercept, r2

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================
st.title("Determinación de Conductividad Térmica - FTIQ")
st.info("💡 Este software estima la conductividad térmica de gases usando modelos moleculares (Eucken Modificado y Chung et al.) y reglas de mezcla (Wassiljewa-Mason-Saxena y Lindsay-Bromley).")

st.sidebar.header("Navegación")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

# ==============================================================================
# MÓDULO 1: SUSTANCIAS PURAS
# ==============================================================================
if menu == "Sustancias Puras":
    st.header("Análisis de Componentes Puros")
    
    modelo = st.selectbox("Modelo Matemático:", ["Eucken Modificado", "Chung et al."])
    
    with st.expander("📖 Ver Ecuación y Variables del Modelo", expanded=True):
        if modelo == "Eucken Modificado":
            st.latex(r"\lambda = \frac{\eta C_v}{M}\left(1.32 + \frac{1.77}{C_v/R}\right)")
            st.markdown("""
            **Variables:**
            - **λ**: Conductividad térmica (W/m·K)
            - **η**: Viscosidad dinámica (N·s/m² o Pa·s)
            - **Cv**: Capacidad calorífica a volumen constante (J/mol·K)
            - **M**: Peso molecular (kg/mol)
            - **R**: Constante de gases (8.314 J/mol·K)
            """)
        elif modelo == "Chung et al.":
            st.latex(r"\lambda = \frac{3.75\,\Psi\,\eta C_v}{M\,(C_v/R)}")
            st.latex(r"\Psi = 1 + \alpha\left[\frac{0.215 + 0.28288\alpha - 1.061\beta + 0.26665Z}{0.6366 + \beta Z + 1.061\alpha\beta}\right]")
            st.markdown("""
            **Variables:**
            - **λ**: Conductividad térmica (W/m·K)
            - **η**: Viscosidad dinámica (N·s/m² o Pa·s)
            - **Cv**: Capacidad calorífica a volumen constante (J/mol·K)
            - **M**: Peso molecular (kg/mol)
            - **α = Cv/R - 1.5**
            - **β = 0.7862 - 0.7109ω + 1.3168ω²**
            - **Z = 2.0 + 10.5Tr²** (Tr = T/Tc)
            - **ω**: Factor acéntrico
            """)
    
    st.subheader("Entrada de Datos")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comp = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_INFO.keys()))
        T = st.number_input("Temperatura (K):", value=300.0, step=10.0)
        P = st.number_input("Presión (MPa):", value=0.1, step=0.1, format="%.3f")
        lambda_exp = st.number_input("Conductividad Experimental (W/m·K):", value=LAMBDA_EXP_PUROS[comp] if comp in LAMBDA_EXP_PUROS else 0.0000, format="%.5f")
    
    with col2:
        M_kg = st.number_input("Peso Molecular (kg/mol):", value=COMPUESTOS_INFO[comp]["M"], format="%.6f")
        eta = st.number_input("Viscosidad η (N·s/m² o Pa·s):", value=1.0e-5, format="%.2e")
        Cp = st.number_input("Cp (J/mol·K):", value=30.0, format="%.3f")
        Cv = st.number_input("Cv (J/mol·K):", value=20.0, format="%.3f")
    
    with col3:
        if modelo == "Chung et al.":
            Tc = st.number_input("Temperatura Crítica Tc (K):", value=PROPIEDADES_CRITICAS[comp]["Tc"], format="%.2f")
            w = st.number_input("Factor Acéntrico ω:", value=PROPIEDADES_CRITICAS[comp]["w"], format="%.4f")
    
    if st.button("Ejecutar Cálculo"):
        try:
            if modelo == "Eucken Modificado":
                lambda_calc = eucken_modificado(eta, Cp, Cv, M_kg)
            elif modelo == "Chung et al.":
                lambda_calc = chung_et_al(eta, Cv, M_kg, T, Tc, w)
            
            error = abs(lambda_exp - lambda_calc) / lambda_exp * 100 if lambda_exp > 0 else 0.0
            
            nuevo_registro = pd.DataFrame([{
                "Componente": comp, "Modelo": modelo, "T (K)": T, "P (MPa)": P,
                "Exp (W/m·K)": lambda_exp, "Calc (W/m·K)": round(lambda_calc, 6),
                "Error (%)": round(error, 3)
            }])
            st.session_state.tabla_puras = pd.concat([st.session_state.tabla_puras, nuevo_registro], ignore_index=True)
            st.success(f"✅ Cálculo finalizado: λ = {lambda_calc:.6f} W/m·K (Error: {error:.2f}%)")
        except Exception as e:
            st.error(f"Error en el cálculo: {e}")
    
    if not st.session_state.tabla_puras.empty:
        st.write("---")
        st.subheader("Memoria de Resultados y Validación")
        st.dataframe(st.session_state.tabla_puras, use_container_width=True)
        
        if st.button("Limpiar Memoria de Componentes"):
            st.session_state.tabla_puras = pd.DataFrame(columns=[
                "Componente", "Modelo", "T (K)", "P (MPa)", "Exp (W/m·K)", 
                "Calc (W/m·K)", "Error (%)"
            ])
            st.rerun()
        
        # Análisis de regresión para el modelo actual
        df_actual = st.session_state.tabla_puras[st.session_state.tabla_puras["Modelo"] == modelo]
        df_valid = df_actual[df_actual["Exp (W/m·K)"] > 0]
        
        if len(df_valid) > 1:
            y_exp = df_valid["Exp (W/m·K)"].astype(float).values
            y_calc = df_valid["Calc (W/m·K)"].astype(float).values
            mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
            
            col_met1, col_met2 = st.columns(2)
            col_met1.metric("Error Global MAPE", f"{mape:.3f} %")
            
            # Gráfico con regresión lineal
            fig, slope, intercept, r2 = linear_regression_plot(
                y_exp, y_calc,
                "λ Experimental (W/m·K)",
                "λ Calculada (W/m·K)",
                f"Diagrama de Dispersión: {modelo}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar estadísticas
            st.markdown(f"""
            <div class='intermedio'>
            <b>📊 Estadísticas de Regresión ({modelo}):</b><br>
            • Ecuación: λ_calc = {slope:.6f} × λ_exp + {intercept:.6f}<br>
            • Coeficiente de Determinación (R²): {r2:.6f}<br>
            • Error Global MAPE: {mape:.3f} %
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# MÓDULO 2: REGLAS DE MEZCLADO (CORREGIDO)
# ==============================================================================
elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Gaseosas Multicomponentes")
    
    regla = st.selectbox("Seleccione la Regla de Mezclado:", 
                         ["Wassiljewa-Mason-Saxena", "Lindsay-Bromley"])
    
    # Mostrar ecuación según regla seleccionada
    with st.expander("📖 Ver Ecuación de la Regla de Mezclado", expanded=True):
        if regla == "Wassiljewa-Mason-Saxena":
            st.latex(r"\lambda_m = \sum_{i=1}^{n} \frac{y_i \lambda_i}{\sum_{j=1}^{n} y_j A_{ij}}")
            st.latex(r"A_{ij} = \frac{\left[1 + \left(\frac{\lambda_{tr,i}}{\lambda_{tr,j}}\right)^{1/2}\left(\frac{M_j}{M_i}\right)^{1/4}\right]^2}{\left[8\left(1 + \frac{M_i}{M_j}\right)\right]^{1/2}}")
            st.latex(r"\frac{\lambda_{tr,i}}{\lambda_{tr,j}} = \frac{\Gamma_j\left[\exp(0.0464T_{r,i}) - \exp(-0.2412T_{r,i})\right]}{\Gamma_i\left[\exp(0.0464T_{r,j}) - \exp(-0.2412T_{r,j})\right]}")
            st.latex(r"\Gamma_i = 210\left(\frac{T_{c,i}^2\,M_i^3}{P_{c,i}^4}\right)^{1/6}")
        elif regla == "Lindsay-Bromley":
            st.latex(r"\lambda_m = \sum_{i=1}^{n} \frac{y_i \lambda_i}{\sum_{j=1}^{n} y_j A_{ij}}")
            st.latex(r"A_{ij} = \frac{1}{4}\left\{1 + \left[\frac{\mu_i}{\mu_j}\left(\frac{M_j}{M_i}\right)^{3/4}\left(\frac{1+S_i/T}{1+S_j/T}\right)^{1/2}\right]^2\right\}\left(\frac{1+S_{ij}/T}{1+S_i/T}\right)")
            st.latex(r"\frac{\mu_i}{\mu_j} = \frac{\lambda_i}{\lambda_j} \cdot \frac{C_{p,j}\left(9 - \frac{5}{\gamma_i}\right)}{C_{p,i}\left(9 - \frac{5}{\gamma_j}\right)}")
            st.latex(r"S_i = 1.5\,T_{b,i} \quad \text{y} \quad S_{ij} = \sqrt{S_i S_j}")
    
    st.subheader("Composición de la Mezcla")
    st.info("💡 Puede usar la composición estándar del gas de cracking (Tabla 9 del documento) o modificarla.")
    
    # Botón para cargar composición estándar
    if st.button("📋 Cargar composición del gas de cracking (Tabla 9)"):
        st.session_state.cargar_composicion = True
    
    # Tabla de composición
    data_mezcla = []
    for comp in COMPUESTOS_INFO:
        fraccion_default = COMPOSICION_CRACKING.get(comp, 0.0) if hasattr(st.session_state, 'cargar_composicion') and st.session_state.cargar_composicion else 0.0
        data_mezcla.append({
            "Componente": comp,
            "Fracción Molar (yᵢ)": fraccion_default,
            "λᵢ (W/m·K)": LAMBDA_EXP_PUROS.get(comp, 0.0),
            "Cp (J/mol·K)": 30.0,
            "Cv (J/mol·K)": 20.0,
            "γ (Cp/Cv)": 1.5,
            "M (g/mol)": COMPUESTOS_INFO[comp]["M_gmol"]
        })
    
    df_mezcla = st.data_editor(pd.DataFrame(data_mezcla), num_rows="fixed", use_container_width=True)
    
    # Control de temperatura
    st.write("---")
    st.subheader("🌡️ Control de Temperatura")
    
    T_operacion = st.number_input("Temperatura de operación (K):", value=300.0, step=10.0, key="temp_operacion")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🚀 Calcular y guardar resultado", type="primary", use_container_width=True):
            # Obtener datos
            y = df_mezcla["Fracción Molar (yᵢ)"].astype(float).values
            lambda_i = df_mezcla["λᵢ (W/m·K)"].astype(float).values
            M_gmol = df_mezcla["M (g/mol)"].astype(float).values
            Cp_i = df_mezcla["Cp (J/mol·K)"].astype(float).values
            Cv_i = df_mezcla["Cv (J/mol·K)"].astype(float).values
            gamma_i = df_mezcla["γ (Cp/Cv)"].astype(float).values
            
            # Normalizar fracciones molares
            suma_y = sum(y)
            if not np.isclose(suma_y, 1.0, atol=0.01) and suma_y > 0:
                st.warning(f"⚠️ La suma de fracciones molares es {suma_y:.4f}. Se normalizarán.")
                y = y / suma_y
            
            try:
                if regla == "Wassiljewa-Mason-Saxena":
                    Tc_lista = [PROPIEDADES_CRITICAS[comp]["Tc"] for comp in df_mezcla["Componente"]]
                    Pc_lista = [PROPIEDADES_CRITICAS[comp]["Pc_bar"] for comp in df_mezcla["Componente"]]
                    lambda_m = wassiljewa_mason_saxena(y, lambda_i, M_gmol, Tc_lista, Pc_lista, T_operacion)
                    
                    nuevo = pd.DataFrame([{"T (K)": T_operacion, "lambda_calc (W/m·K)": round(lambda_m, 8)}])
                    st.session_state.tabla_mezclas_wass = pd.concat([st.session_state.tabla_mezclas_wass, nuevo], ignore_index=True)
                    
                else:  # Lindsay-Bromley
                    Tb_lista = [T_EBULLICION[comp] for comp in df_mezcla["Componente"]]
                    lambda_m = lindsay_bromley(y, lambda_i, M_gmol, Cp_i, gamma_i, Tb_lista, T_operacion)
                    
                    nuevo = pd.DataFrame([{"T (K)": T_operacion, "lambda_calc (W/m·K)": round(lambda_m, 8)}])
                    st.session_state.tabla_mezclas_lindsay = pd.concat([st.session_state.tabla_mezclas_lindsay, nuevo], ignore_index=True)
                
                st.success(f"✅ λ = {lambda_m:.8f} W/m·K a T = {T_operacion} K")
            except Exception as e:
                st.error(f"Error en el cálculo: {e}")
    
    with col_btn2:
        T_extra = st.number_input("T adicional (K):", value=350.0, step=10.0, key="temp_extra")
        if st.button("➕ Agregar esta temperatura", use_container_width=True):
            # Obtener datos
            y = df_mezcla["Fracción Molar (yᵢ)"].astype(float).values
            lambda_i = df_mezcla["λᵢ (W/m·K)"].astype(float).values
            M_gmol = df_mezcla["M (g/mol)"].astype(float).values
            Cp_i = df_mezcla["Cp (J/mol·K)"].astype(float).values
            Cv_i = df_mezcla["Cv (J/mol·K)"].astype(float).values
            gamma_i = df_mezcla["γ (Cp/Cv)"].astype(float).values
            
            suma_y = sum(y)
            if not np.isclose(suma_y, 1.0, atol=0.01) and suma_y > 0:
                y = y / suma_y
            
            try:
                if regla == "Wassiljewa-Mason-Saxena":
                    Tc_lista = [PROPIEDADES_CRITICAS[comp]["Tc"] for comp in df_mezcla["Componente"]]
                    Pc_lista = [PROPIEDADES_CRITICAS[comp]["Pc_bar"] for comp in df_mezcla["Componente"]]
                    lambda_m = wassiljewa_mason_saxena(y, lambda_i, M_gmol, Tc_lista, Pc_lista, T_extra)
                    
                    nuevo = pd.DataFrame([{"T (K)": T_extra, "lambda_calc (W/m·K)": round(lambda_m, 8)}])
                    st.session_state.tabla_mezclas_wass = pd.concat([st.session_state.tabla_mezclas_wass, nuevo], ignore_index=True)
                    
                else:
                    Tb_lista = [T_EBULLICION[comp] for comp in df_mezcla["Componente"]]
                    lambda_m = lindsay_bromley(y, lambda_i, M_gmol, Cp_i, gamma_i, Tb_lista, T_extra)
                    
                    nuevo = pd.DataFrame([{"T (K)": T_extra, "lambda_calc (W/m·K)": round(lambda_m, 8)}])
                    st.session_state.tabla_mezclas_lindsay = pd.concat([st.session_state.tabla_mezclas_lindsay, nuevo], ignore_index=True)
                
                st.success(f"✅ Agregado: λ = {lambda_m:.8f} W/m·K a T = {T_extra} K")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Mostrar resultados según regla seleccionada
    st.write("---")
    st.subheader("📈 Resultados y Gráfico λ vs Temperatura")
    
    if regla == "Wassiljewa-Mason-Saxena":
        df_resultados = st.session_state.tabla_mezclas_wass
    else:
        df_resultados = st.session_state.tabla_mezclas_lindsay
    
    if not df_resultados.empty:
        st.dataframe(df_resultados.sort_values("T (K)"), use_container_width=True)
        
        # Gráfico con línea punteada y regresión
        df_sorted = df_resultados.sort_values("T (K)")
        x_temp = df_sorted["T (K)"].values
        y_lambda = df_sorted["lambda_calc (W/m·K)"].values
        
        if len(x_temp) > 1:
            # Regresión lineal
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_temp, y_lambda)
            r2 = r_value**2
            x_lineal = np.linspace(min(x_temp), max(x_temp), 100)
            y_lineal = slope * x_lineal + intercept
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_temp, y=y_lambda,
                mode='markers+lines',
                marker=dict(size=8, color='#0056b3'),
                line=dict(color='#0056b3', dash='dash', width=2),
                name=f'{regla} (datos)'
            ))
            fig.add_trace(go.Scatter(
                x=x_lineal, y=y_lineal,
                mode='lines',
                line=dict(color='red', dash='dot', width=2),
                name=f'Regresión: λ = {slope:.6f}T + {intercept:.6f}'
            ))
            
            fig.update_layout(
                title=f"Conductividad Térmica vs Temperatura<br><sub>{regla} | Ecuación: λ = {slope:.6f}·T + {intercept:.6f} | R² = {r2:.6f}</sub>",
                xaxis_title="Temperatura (K)",
                yaxis_title="λ (W/m·K)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Solo puntos, sin regresión
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_temp, y=y_lambda,
                mode='markers+lines',
                marker=dict(size=8, color='#0056b3'),
                line=dict(color='#0056b3', dash='dash', width=2),
                name=f'{regla}'
            ))
            fig.update_layout(
                title=f"Conductividad Térmica vs Temperatura<br><sub>{regla}</sub>",
                xaxis_title="Temperatura (K)",
                yaxis_title="λ (W/m·K)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Botón para limpiar memoria según regla
    if st.button("🗑️ Limpiar memoria de resultados"):
        if regla == "Wassiljewa-Mason-Saxena":
            st.session_state.tabla_mezclas_wass = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])
        else:
            st.session_state.tabla_mezclas_lindsay = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])
        st.rerun()

# ==============================================================================
# FOOTER
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Referencias**")
st.sidebar.markdown("- Poling, B. E. et al. (2001). *The Properties of Gases and Liquids* (5th ed.).")
st.sidebar.markdown("- Hopp, M. & Gross, J. (2019). *I&EC Research*, 58(45), 20816-20828.")
st.sidebar.markdown("- Lindsay, A. L. & Bromley, L. A. (1950). *Industrial & Engineering Chemistry*, 42(8), 1508-1511.")
