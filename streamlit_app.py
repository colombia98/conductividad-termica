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

# Temperaturas de ebullición (Tb en K) para Sutherland
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

# Composición estándar del gas de cracking (Tabla 9)
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

# ==============================================================================
# CORRELACIONES DE CONDUCTIVIDAD TÉRMICA k(T) SEGÚN DOCUMENTO
# ==============================================================================
def k_H2(T):
    return 5.417e-10 * T**3 - 9.068e-7 * T**2 + 8.829e-4 * T - 1.162e-2

def k_CH4(T):
    return -3.333e-11 * T**3 + 1.800e-7 * T**2 + 4.233e-5 * T + 6.400e-3

def k_C2H6(T):
    return -2.500e-10 * T**3 + 4.200e-7 * T**2 - 5.950e-5 * T + 8.300e-3

def k_C2H4(T):
    return -2.250e-10 * T**3 + 3.711e-7 * T**2 + 1.036e-5 * T + 5.980e-3

def k_C2H2(T):
    return -1.500e-10 * T**3 + 1.450e-7 * T**2 + 7.500e-5 * T + 1.800e-3

def k_C3H6(T):
    return 1.000e-7 * T**2 + 6.600e-5 * T - 1.180e-2

def k_C3H8(T):
    return -2.667e-10 * T**3 + 4.300e-7 * T**2 - 7.333e-5 * T + 8.500e-3

def k_C4H10(T):
    return 1.550e-7 * T**2 + 7.500e-6 * T + 5.000e-4

def k_compuesto(compuesto, T):
    funciones = {
        "Hidrógeno": k_H2,
        "Metano": k_CH4,
        "Etano": k_C2H6,
        "Etileno": k_C2H4,
        "Acetileno": k_C2H2,
        "Propileno": k_C3H6,
        "Propano": k_C3H8,
        "n-Butano": k_C4H10
    }
    return funciones[compuesto](T)

# ==============================================================================
# ECUACIONES DE SHOMATE PARA Cp(T)
# ==============================================================================
SHOMATE = {
    "Hidrógeno": {"A": 33.066, "B": -11.363, "C": 11.433, "D": -2.773, "E": -0.159},
    "Metano": {"A": -0.703, "B": 108.477, "C": -42.521, "D": 5.863, "E": 0.679},
    "Etano": {"A": -3.029, "B": 188.401, "C": -91.563, "D": 15.547, "E": 0.077},
    "Etileno": {"A": -6.388, "B": 184.401, "C": -112.923, "D": 28.496, "E": 0.315},
    "Acetileno": {"A": 40.685, "B": 40.026, "C": -16.154, "D": 3.670, "E": -0.658},
    "Propileno": {"A": 3.834, "B": 234.898, "C": -117.888, "D": 22.897, "E": -0.069},
    "Propano": {"A": -23.178, "B": 363.742, "C": -222.981, "D": 56.254, "E": 0.978},
    "n-Butano": {"A": -5.505, "B": 435.979, "C": -251.939, "D": 58.367, "E": 0.955}
}

def cp_shomate(compuesto, T):
    R = 8.314
    s = SHOMATE[compuesto]
    t = T / 1000
    Cp = s["A"] + s["B"]*t + s["C"]*t**2 + s["D"]*t**3 + s["E"]/t**2
    return Cp

def cv_from_cp(cp, R=8.314):
    return cp - R

# ==============================================================================
# FUNCIONES DE REGLAS DE MEZCLA (AUTOMATIZADAS)
# ==============================================================================

def gamma_roy_thodos(Tc, Pc_bar, M_gmol):
    """Γ = 210 * (Tc^2 * M^3 / Pc^4)^(1/6) donde Pc en atm"""
    Pc_atm = Pc_bar * 0.986923
    gamma = 210 * ((Tc**2 * M_gmol**3) / (Pc_atm**4))**(1/6)
    return gamma

def lambda_tr_ratio(T, Tc_i, Tc_j, gamma_i, gamma_j):
    """Relación de conductividades translacionales (Roy y Thodos)"""
    Tri = T / Tc_i
    Trj = T / Tc_j
    f_i = np.exp(0.0464 * Tri) - np.exp(-0.2412 * Tri)
    f_j = np.exp(0.0464 * Trj) - np.exp(-0.2412 * Trj)
    ratio = (gamma_j * f_i) / (gamma_i * f_j)
    return ratio

def wassiljewa_mason_saxena(y, T):
    """
    Regla de Wassiljewa con modificación de Mason y Saxena
    Calcula todo automáticamente a partir de la temperatura
    """
    n = len(y)
    comps = list(COMPOSICION_CRACKING.keys())
    
    lambda_i = []
    M_gmol = []
    Tc = []
    Pc_bar = []
    
    for i, comp in enumerate(comps):
        if y[i] > 0:
            lambda_i.append(k_compuesto(comp, T))
            M_gmol.append(COMPUESTOS_INFO[comp]["M_gmol"])
            Tc.append(PROPIEDADES_CRITICAS[comp]["Tc"])
            Pc_bar.append(PROPIEDADES_CRITICAS[comp]["Pc_bar"])
        else:
            lambda_i.append(0)
            M_gmol.append(COMPUESTOS_INFO[comp]["M_gmol"])
            Tc.append(PROPIEDADES_CRITICAS[comp]["Tc"])
            Pc_bar.append(PROPIEDADES_CRITICAS[comp]["Pc_bar"])
    
    lambda_i = np.array(lambda_i)
    M_gmol = np.array(M_gmol)
    Tc = np.array(Tc)
    Pc_bar = np.array(Pc_bar)
    
    # Calcular Gamma para cada componente
    gamma = np.zeros(n)
    for i in range(n):
        gamma[i] = gamma_roy_thodos(Tc[i], Pc_bar[i], M_gmol[i])
    
    # Construir matriz de interacción A_ij
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and lambda_i[i] > 0 and lambda_i[j] > 0:
                ratio_lambda_tr = lambda_tr_ratio(T, Tc[i], Tc[j], gamma[i], gamma[j])
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

def lindsay_bromley(y, T):
    """
    Regla de Lindsay y Bromley
    Calcula todo automáticamente a partir de la temperatura
    """
    n = len(y)
    comps = list(COMPOSICION_CRACKING.keys())
    R = 8.314
    
    lambda_i = []
    Cp_i = []
    M_gmol = []
    Tb_lista = []
    gamma_i = []
    
    for i, comp in enumerate(comps):
        if y[i] > 0:
            lambda_i.append(k_compuesto(comp, T))
            Cp_i.append(cp_shomate(comp, T))
            M_gmol.append(COMPUESTOS_INFO[comp]["M_gmol"])
            Tb_lista.append(T_EBULLICION[comp])
            Cv = cp_shomate(comp, T) - R
            gamma_i.append(cp_shomate(comp, T) / Cv)
        else:
            lambda_i.append(0)
            Cp_i.append(0)
            M_gmol.append(COMPUESTOS_INFO[comp]["M_gmol"])
            Tb_lista.append(T_EBULLICION[comp])
            gamma_i.append(1.4)
    
    lambda_i = np.array(lambda_i)
    Cp_i = np.array(Cp_i)
    M_gmol = np.array(M_gmol)
    Tb_lista = np.array(Tb_lista)
    gamma_i = np.array(gamma_i)
    
    # Constante de Sutherland: S_i = 1.5 * Tb,i
    S_i = 1.5 * Tb_lista
    
    # Estimar viscosidad a partir de λ y Cp usando relación inversa de Eucken
    eta_i = np.zeros(n)
    for i in range(n):
        if lambda_i[i] > 0 and Cp_i[i] > 0:
            Cv_i = Cp_i[i] - R
            if Cv_i > 0:
                factor_eucken = 1.32 + 1.77 / (Cv_i / R)
                M_kg = M_gmol[i] / 1000
                eta_i[i] = (lambda_i[i] * M_kg) / (Cv_i * factor_eucken)
            else:
                eta_i[i] = 1e-6
        else:
            eta_i[i] = 1e-6
    
    # Construir matriz de interacción A_ij
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and lambda_i[i] > 0 and lambda_i[j] > 0 and eta_i[j] > 0:
                visc_ratio = (lambda_i[i] / lambda_i[j]) * (Cp_i[j] / Cp_i[i])
                visc_ratio *= (9 - 5/gamma_i[i]) / (9 - 5/gamma_i[j])
                
                mass_factor = (M_gmol[j] / M_gmol[i])**0.75
                suth_factor = ((1 + S_i[i]/T) / (1 + S_i[j]/T))**0.5
                
                bracket = visc_ratio * mass_factor * suth_factor
                S_ij = np.sqrt(S_i[i] * S_i[j])
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
# FUNCIONES DE LOS MODELOS PARA COMPUESTOS PUROS (IGUAL QUE ANTES)
# ==============================================================================

def eucken_modificado(eta, Cp, Cv, M, R=8.314):
    lambda_calc = (eta * Cv / M) * (1.32 + 1.77 / (Cv / R))
    return lambda_calc

def chung_et_al(eta, Cv, M, T, Tc, w, R=8.314):
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
# INICIALIZACIÓN DE SESSION STATE
# ==============================================================================
if 'tabla_puras' not in st.session_state:
    st.session_state.tabla_puras = pd.DataFrame(columns=[
        "Componente", "Modelo", "T (K)", "P (MPa)", "Exp (W/m·K)", 
        "Calc (W/m·K)", "Error (%)"
    ])

if 'tabla_mezclas_wass' not in st.session_state:
    st.session_state.tabla_mezclas_wass = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])

if 'tabla_mezclas_lindsay' not in st.session_state:
    st.session_state.tabla_mezclas_lindsay = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================
st.title("Determinación de Conductividad Térmica - FTIQ")
st.info("💡 Este software estima la conductividad térmica de gases usando modelos moleculares (Eucken Modificado y Chung et al.) y reglas de mezcla (Wassiljewa-Mason-Saxena y Lindsay-Bromley).")

st.sidebar.header("Navegación")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

# ==============================================================================
# MÓDULO 1: SUSTANCIAS PURAS (IGUAL QUE ANTES)
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
        lambda_exp = st.number_input("Conductividad Experimental (W/m·K):", value=0.0000, format="%.5f")
    
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
        
        df_valid = st.session_state.tabla_puras[st.session_state.tabla_puras["Exp (W/m·K)"] > 0]
        if len(df_valid) > 0:
            y_exp = df_valid["Exp (W/m·K)"].astype(float).values
            y_calc = df_valid["Calc (W/m·K)"].astype(float).values
            nombres = df_valid["Componente"].values
            mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
            
            col_met1, col_met2 = st.columns(2)
            col_met1.metric("Error Global MAPE", f"{mape:.3f} %")
            if len(y_exp) > 1:
                r2 = r2_score(y_exp, y_calc)
                col_met2.metric("Coeficiente de Determinación (R²)", f"{r2:.5f}")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_exp, y=y_calc, mode='markers+text', text=nombres, textposition="top center", marker=dict(size=10, color='#0056b3'), name="Calculada"))
            val_min, val_max = min(min(y_exp), min(y_calc)) * 0.95, max(max(y_exp), max(y_calc)) * 1.05
            fig.add_trace(go.Scatter(x=[val_min, val_max], y=[val_min, val_max], mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            fig.update_layout(title="λ Experimental vs λ Calculada", xaxis_title="λ Experimental (W/m·K)", yaxis_title="λ Calculada (W/m·K)", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# MÓDULO 2: REGLAS DE MEZCLADO (AUTOMATIZADO CON CORRELACIONES)
# ==============================================================================
elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Gaseosas Multicomponentes")
    
    regla = st.selectbox("Seleccione la Regla de Mezclado:", 
                         ["Wassiljewa-Mason-Saxena", "Lindsay-Bromley"])
    
    # --- BLOQUE INFORMATIVO COMPLETO ---
    with st.expander("📖 Ver Ecuación, Nomenclatura y Procedimiento", expanded=True):
        st.markdown("### **Ecuación General de Wassiljewa**")
        st.latex(r"\lambda_m = \sum_{i=1}^{n} \frac{y_i \lambda_i}{\sum_{j=1}^{n} y_j A_{ij}}")
        
        if regla == "Wassiljewa-Mason-Saxena":
            st.markdown("#### **Coeficiente de Interacción A_ij (Mason y Saxena)**")
            st.latex(r"A_{ij} = \frac{\left[1 + \left(\frac{\lambda_{tr,i}}{\lambda_{tr,j}}\right)^{1/2}\left(\frac{M_j}{M_i}\right)^{1/4}\right]^2}{\left[8\left(1 + \frac{M_i}{M_j}\right)\right]^{1/2}}")
            
            st.markdown("#### **Relación de Conductividades Translacionales (Roy y Thodos)**")
            st.latex(r"\frac{\lambda_{tr,i}}{\lambda_{tr,j}} = \frac{\Gamma_j\left[\exp(0.0464T_{r,i}) - \exp(-0.2412T_{r,i})\right]}{\Gamma_i\left[\exp(0.0464T_{r,j}) - \exp(-0.2412T_{r,j})\right]}")
            
            st.markdown("#### **Parámetro Γ (Inversa de Conductividad Reducida)**")
            st.latex(r"\Gamma_i = 210\left(\frac{T_{c,i}^2 M_i^3}{P_{c,i}^4}\right)^{1/6}")
            
            st.markdown("#### **Nomenclatura**")
            st.markdown("""
            - **λₘ**: Conductividad térmica de la mezcla (W/m·K)
            - **yᵢ, yⱼ**: Fracciones molares de los componentes i y j
            - **λᵢ**: Conductividad térmica del componente puro i (W/m·K) - calculada con k(T)
            - **Mᵢ, Mⱼ**: Pesos moleculares (g/mol)
            - **λ_tr,i**: Conductividad translacional del componente i
            - **T_r,i = T/T_c,i**: Temperatura reducida
            - **T_c,i**: Temperatura crítica (K)
            - **P_c,i**: Presión crítica (bar)
            """)
            
        else:  # Lindsay-Bromley
            st.markdown("#### **Coeficiente de Interacción A_ij (Lindsay y Bromley)**")
            st.latex(r"A_{ij} = \frac{1}{4}\left\{1 + \left[\frac{\mu_i}{\mu_j}\left(\frac{M_j}{M_i}\right)^{3/4}\left(\frac{1+S_i/T}{1+S_j/T}\right)^{1/2}\right]^2\right\}\left(\frac{1+S_{ij}/T}{1+S_i/T}\right)")
            
            st.markdown("#### **Relación de Viscosidades (Eucken Modificado)**")
            st.latex(r"\frac{\mu_i}{\mu_j} = \frac{\lambda_i}{\lambda_j} \cdot \frac{C_{p,j}\left(9 - \frac{5}{\gamma_i}\right)}{C_{p,i}\left(9 - \frac{5}{\gamma_j}\right)}")
            
            st.markdown("#### **Constante de Sutherland**")
            st.latex(r"S_i = 1.5\,T_{b,i} \quad \text{y} \quad S_{ij} = \sqrt{S_i S_j}")
            
            st.markdown("#### **Nomenclatura**")
            st.markdown("""
            - **λₘ**: Conductividad térmica de la mezcla (W/m·K)
            - **yᵢ, yⱼ**: Fracciones molares de los componentes i y j
            - **λᵢ**: Conductividad térmica del componente puro i (W/m·K) - calculada con k(T)
            - **μᵢ, μⱼ**: Viscosidades dinámicas (Pa·s) - estimadas desde λ y Cp
            - **Mᵢ, Mⱼ**: Pesos moleculares (g/mol)
            - **C_p,i**: Capacidad calorífica a presión constante - calculada con Shomate
            - **γ_i = C_p/C_v**: Relación de capacidades caloríficas
            - **S_i**: Constante de Sutherland (K) - basada en T_b,i
            - **T_b,i**: Temperatura de ebullición normal (K)
            - **T**: Temperatura de la mezcla (K)
            """)
        
        st.markdown("---")
        st.markdown("### **📌 Procedimiento de Cálculo Automático**")
        st.markdown("""
        1. **k(T)**: Se calcula con las correlaciones polinómicas del documento (Tabla 15)
        2. **Cp(T)**: Se calcula con las ecuaciones de Shomate (Tabla 16)
        3. **Cv(T)**: Se obtiene de Cv = Cp - R (gas ideal)
        4. **λᵢ, Cpᵢ, Cvᵢ, γᵢ** se calculan automáticamente para cada temperatura
        5. Se aplica la regla de mezclado seleccionada
        6. Se genera la gráfica λ vs T con regresión lineal
        """)
    
    # Composición de la mezcla
    st.subheader("📋 Composición de la Mezcla")
    st.info("La composición corresponde al gas de cracking de etano (Tabla 9 del documento)")
    
    df_composicion = pd.DataFrame([
        {"Compuesto": comp, "Fracción Molar (yᵢ)": COMPOSICION_CRACKING[comp]}
        for comp in COMPUESTOS_INFO.keys()
    ])
    
    df_edit = st.data_editor(df_composicion, num_rows="fixed", use_container_width=True)
    
    y = df_edit["Fracción Molar (yᵢ)"].astype(float).values
    suma_y = sum(y)
    if not np.isclose(suma_y, 1.0, atol=0.01):
        st.warning(f"⚠️ La suma de fracciones molares es {suma_y:.4f}. Se normalizarán automáticamente.")
        y = y / suma_y
    
    # Control de temperatura
    st.write("---")
    st.subheader("🌡️ Rango de Temperatura")
    
    col_Tmin, col_Tmax, col_step = st.columns(3)
    with col_Tmin:
        T_min = st.number_input("Temperatura mínima (K):", value=290.0, step=10.0)
    with col_Tmax:
        T_max = st.number_input("Temperatura máxima (K):", value=410.0, step=10.0)
    with col_step:
        T_step = st.number_input("Incremento (K):", value=10.0, step=5.0)
    
    if st.button("🚀 Calcular Conductividad de la Mezcla", type="primary", use_container_width=True):
        if T_max <= T_min:
            st.error("La temperatura máxima debe ser mayor que la mínima")
        else:
            temperaturas = np.arange(T_min, T_max + T_step/2, T_step)
            
            resultados_wass = []
            resultados_lindsay = []
            
            with st.spinner("Calculando..."):
                for T_val in temperaturas:
                    try:
                        lambda_wass = wassiljewa_mason_saxena(y, T_val)
                        lambda_lindsay = lindsay_bromley(y, T_val)
                        resultados_wass.append({"T (K)": T_val, "lambda_calc (W/m·K)": lambda_wass})
                        resultados_lindsay.append({"T (K)": T_val, "lambda_calc (W/m·K)": lambda_lindsay})
                    except Exception as e:
                        st.error(f"Error a T={T_val} K: {e}")
            
            st.session_state.tabla_mezclas_wass = pd.DataFrame(resultados_wass)
            st.session_state.tabla_mezclas_lindsay = pd.DataFrame(resultados_lindsay)
            st.success(f"✅ Cálculo completado para {len(temperaturas)} temperaturas")
    
    # Mostrar resultados
    st.write("---")
    st.subheader("📈 Resultados y Gráfico λ vs Temperatura")
    
    if regla == "Wassiljewa-Mason-Saxena":
        df_resultados = st.session_state.tabla_mezclas_wass
        color = '#0056b3'
    else:
        df_resultados = st.session_state.tabla_mezclas_lindsay
        color = '#e67e22'
    
    if not df_resultados.empty:
        st.dataframe(df_resultados.sort_values("T (K)").round(8), use_container_width=True)
        
        df_sorted = df_resultados.sort_values("T (K)")
        x_temp = df_sorted["T (K)"].values
        y_lambda = df_sorted["lambda_calc (W/m·K)"].values
        
        if len(x_temp) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_temp, y_lambda)
            r2 = r_value**2
            x_lineal = np.linspace(min(x_temp), max(x_temp), 100)
            y_lineal = slope * x_lineal + intercept
            
            fig = go.Figure()
            
            # Datos calculados (puntos y línea punteada)
            fig.add_trace(go.Scatter(
                x=x_temp, y=y_lambda,
                mode='markers+lines',
                marker=dict(size=8, color=color),
                line=dict(color=color, dash='dash', width=2),
                name=f'{regla} (calculado)'
            ))
            
            # Regresión lineal
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
            
            st.markdown(f"""
            <div class='intermedio'>
            <b>📊 Resultados de Regresión Lineal ({regla}):</b><br>
            • Ecuación: λ = {slope:.8f} × T + {intercept:.8f}<br>
            • Coeficiente de Determinación (R²): {r2:.6f}
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("🗑️ Limpiar todos los resultados"):
        st.session_state.tabla_mezclas_wass = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])
        st.session_state.tabla_mezclas_lindsay = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])
        st.rerun()
    
    # Tabla de resultados esperados
    with st.expander("📚 Ver resultados esperados según documento (Tabla 17)", expanded=False):
        datos_esperados = {
            "T (K)": [290, 310, 330, 350, 370, 390, 410],
            "Wassiljewa (W/m·K)": [0.0245, 0.0271, 0.0298, 0.0326, 0.0356, 0.0386, 0.0417],
            "Lindsay-Bromley (W/m·K)": [0.0224, 0.0247, 0.0271, 0.0296, 0.0322, 0.0349, 0.0376]
        }
        df_esperados = pd.DataFrame(datos_esperados)
        st.dataframe(df_esperados, use_container_width=True)

# ==============================================================================
# FOOTER
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Referencias**")
st.sidebar.markdown("- Poling, B. E. et al. (2001). *The Properties of Gases and Liquids* (5th ed.).")
st.sidebar.markdown("- Lindsay, A. L. & Bromley, L. A. (1950). *Industrial & Engineering Chemistry*, 42(8), 1508-1511.")
st.sidebar.markdown("- Hopp, M. & Gross, J. (2019). *I&EC Research*, 58(45), 20816-20828.")
