import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

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

# Temperaturas de ebullición (Tb en K)
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

# ==============================================================================
# INICIALIZACIÓN DE SESSION STATE
# ==============================================================================
if 'tabla_puras' not in st.session_state:
    st.session_state.tabla_puras = pd.DataFrame(columns=[
        "Componente", "Modelo", "T (K)", "P (MPa)", "Exp (W/m·K)", 
        "Calc (W/m·K)", "Error (%)"
    ])

if 'tabla_mezclas' not in st.session_state:
    st.session_state.tabla_mezclas = pd.DataFrame(columns=[
        "T (K)", "Regla_Mezcla", "lambda_calc (W/m·K)"
    ])

# ==============================================================================
# FUNCIONES DE LOS MODELOS PARA COMPUESTOS PUROS
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
# FUNCIONES DE REGLAS DE MEZCLA
# ==============================================================================

def gamma_roy_thodos(Tc, Pc_bar, M_gmol):
    Pc_atm = Pc_bar * 0.986923
    gamma = 210 * ((Tc**2 * M_gmol**3) / (Pc_atm**4))**(1/6)
    return gamma

def lambda_tr_ratio(T, Tc_i, Tc_j, gamma_i, gamma_j):
    Tri = T / Tc_i
    Trj = T / Tc_j
    f_i = np.exp(0.0464 * Tri) - np.exp(-0.2412 * Tri)
    f_j = np.exp(0.0464 * Trj) - np.exp(-0.2412 * Trj)
    ratio = (gamma_j * f_i) / (gamma_i * f_j)
    return ratio

def wassiljewa_mason_saxena(y, lambda_i, M_gmol, Tc, Pc_bar, T):
    n = len(y)
    gamma = {}
    for i in range(n):
        gamma[i] = gamma_roy_thodos(Tc[i], Pc_bar[i], M_gmol[i])
    
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                ratio_lambda_tr = lambda_tr_ratio(T, Tc[i], Tc[j], gamma[i], gamma[j])
                term = (1 + np.sqrt(ratio_lambda_tr) * (M_gmol[j] / M_gmol[i])**0.25)**2
                denom = np.sqrt(8 * (1 + M_gmol[i] / M_gmol[j]))
                A[i, j] = term / denom
    
    lambda_m = 0
    for i in range(n):
        numerador = y[i] * lambda_i[i]
        denominador = sum(y[j] * A[i, j] for j in range(n))
        if denominador > 0:
            lambda_m += numerador / denominador
    return lambda_m

def lindsay_bromley(y, lambda_i, Cp_i, gamma_i, M_gmol, Tb_lista, T):
    """
    Regla de Lindsay y Bromley simplificada
    La viscosidad se estima a partir de λ y Cp usando la relación de Eucken
    """
    n = len(y)
    # Constante de Sutherland
    S_i = 1.5 * np.array(Tb_lista)
    
    # Estimar viscosidad a partir de λ y Cp usando relación inversa de Eucken
    R = 8.314
    eta_i = np.zeros(n)
    for i in range(n):
        # Despejar η de λ = (η*Cv/M)*(1.32 + 1.77/(Cv/R))
        Cv_i = Cp_i[i] / gamma_i[i]
        factor_eucken = 1.32 + 1.77 / (Cv_i / R)
        M_kg = M_gmol[i] / 1000
        eta_i[i] = (lambda_i[i] * M_kg) / (Cv_i * factor_eucken)
    
    # Construir matriz de interacción A_ij
    A = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # Relación de viscosidades
                term_visc = (eta_i[i] / eta_i[j]) * (Cp_i[j] / Cp_i[i])
                term_visc *= (9 - 5/gamma_i[i]) / (9 - 5/gamma_i[j])
                
                # Factor de masa molecular
                mass_factor = (M_gmol[j] / M_gmol[i])**0.75
                
                # Factor de Sutherland
                sutherland_factor = ((1 + S_i[i]/T) / (1 + S_i[j]/T))**0.5
                
                bracket = term_visc * mass_factor * sutherland_factor
                S_ij = np.sqrt(S_i[i] * S_i[j])
                A[i, j] = 0.25 * (1 + bracket**2) * ((1 + S_ij/T) / (1 + S_i[i]/T))
    
    # Calcular conductividad de la mezcla
    lambda_m = 0
    for i in range(n):
        numerador = y[i] * lambda_i[i]
        denominador = sum(y[j] * A[i, j] for j in range(n))
        if denominador > 0:
            lambda_m += numerador / denominador
    return lambda_m

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
        elif modelo == "Chung et al.":
            st.latex(r"\lambda = \frac{3.75\,\Psi\,\eta C_v}{M\,(C_v/R)}")
    
    st.subheader("Entrada de Datos")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comp = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_INFO.keys()))
        T = st.number_input("Temperatura (K):", value=300.0, step=10.0)
        P = st.number_input("Presión (MPa):", value=0.1, step=0.1, format="%.3f")
        lambda_exp = st.number_input("Conductividad Experimental (W/m·K):", value=0.0000, format="%.5f")
    
    with col2:
        M_kg = st.number_input("Peso Molecular (kg/mol):", value=COMPUESTOS_INFO[comp]["M"], format="%.6f")
        eta = st.number_input("Viscosidad η (Pa·s):", value=1.0e-5, format="%.2e")
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
# MÓDULO 2: REGLAS DE MEZCLADO (VERSIÓN SIMPLIFICADA)
# ==============================================================================
elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Gaseosas Multicomponentes")
    
    regla = st.selectbox("Seleccione la Regla de Mezclado:", 
                         ["Wassiljewa-Mason-Saxena", "Lindsay-Bromley"])
    
    st.subheader("Composición de la Mezcla")
    st.info("Ingrese las fracciones molares y las conductividades individuales de cada componente.")
    
    # Tabla simplificada - SOLO columnas necesarias
    data_mezcla = []
    for comp in COMPUESTOS_INFO:
        data_mezcla.append({
            "Componente": comp,
            "Fracción Molar (yᵢ)": 0.0,
            "λᵢ (W/m·K)": 0.0,
            "Cp (J/mol·K)": 30.0,
            "Cv (J/mol·K)": 20.0,
            "M (g/mol)": COMPUESTOS_INFO[comp]["M_gmol"]
        })
    
    df_mezcla = st.data_editor(pd.DataFrame(data_mezcla), num_rows="fixed", use_container_width=True)
    
    # Control de temperatura - MÁS VISIBLE
    st.write("---")
    st.subheader("🌡️ Control de Temperatura")
    
    col_temp1, col_temp2, col_temp3 = st.columns([2, 1, 1])
    with col_temp1:
        T_operacion = st.number_input("Temperatura de operación (K):", value=300.0, step=10.0, key="temp_principal")
    with col_temp2:
        if st.button("📊 Calcular a esta T", use_container_width=True):
            st.session_state.temp_actual = T_operacion
    with col_temp3:
        nueva_temp = st.number_input("Nueva T (K) para agregar:", value=350.0, step=10.0, key="temp_nueva")
        if st.button("➕ Agregar temperatura", use_container_width=True):
            st.session_state.temp_agregar = nueva_temp
    
    # Inicializar variable de temperatura a usar
    if 'temp_actual' not in st.session_state:
        st.session_state.temp_actual = 300.0
    
    # Botón principal de cálculo
    if st.button("🚀 Calcular Conductividad de la Mezcla", type="primary"):
        T_usar = st.session_state.get('temp_actual', 300.0)
        
        y = df_mezcla["Fracción Molar (yᵢ)"].astype(float).values
        lambda_i = df_mezcla["λᵢ (W/m·K)"].astype(float).values
        M_gmol = df_mezcla["M (g/mol)"].astype(float).values
        Cp_i = df_mezcla["Cp (J/mol·K)"].astype(float).values
        Cv_i = df_mezcla["Cv (J/mol·K)"].astype(float).values
        
        # Calcular gamma internamente
        gamma_i = Cp_i / Cv_i
        
        suma_y = sum(y)
        if not np.isclose(suma_y, 1.0, atol=0.01) and suma_y > 0:
            st.warning(f"⚠️ La suma de fracciones molares es {suma_y:.4f}. Se normalizarán.")
            y = y / suma_y
        
        try:
            if regla == "Wassiljewa-Mason-Saxena":
                Tc_lista = [PROPIEDADES_CRITICAS[comp]["Tc"] for comp in df_mezcla["Componente"]]
                Pc_lista = [PROPIEDADES_CRITICAS[comp]["Pc_bar"] for comp in df_mezcla["Componente"]]
                lambda_m = wassiljewa_mason_saxena(y, lambda_i, M_gmol, Tc_lista, Pc_lista, T_usar)
            else:  # Lindsay-Bromley
                Tb_lista = [T_EBULLICION[comp] for comp in df_mezcla["Componente"]]
                lambda_m = lindsay_bromley(y, lambda_i, Cp_i, gamma_i, M_gmol, Tb_lista, T_usar)
            
            nuevo = pd.DataFrame([{"T (K)": T_usar, "Regla_Mezcla": regla, "lambda_calc (W/m·K)": round(lambda_m, 6)}])
            st.session_state.tabla_mezclas = pd.concat([st.session_state.tabla_mezclas, nuevo], ignore_index=True)
            st.success(f"✅ λ = {lambda_m:.6f} W/m·K a T = {T_usar} K")
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Botón para agregar temperatura adicional
    if 'temp_agregar' in st.session_state:
        T_extra = st.session_state.temp_agregar
        y = df_mezcla["Fracción Molar (yᵢ)"].astype(float).values
        lambda_i = df_mezcla["λᵢ (W/m·K)"].astype(float).values
        M_gmol = df_mezcla["M (g/mol)"].astype(float).values
        Cp_i = df_mezcla["Cp (J/mol·K)"].astype(float).values
        Cv_i = df_mezcla["Cv (J/mol·K)"].astype(float).values
        gamma_i = Cp_i / Cv_i
        
        suma_y = sum(y)
        if not np.isclose(suma_y, 1.0, atol=0.01) and suma_y > 0:
            y = y / suma_y
        
        try:
            if regla == "Wassiljewa-Mason-Saxena":
                Tc_lista = [PROPIEDADES_CRITICAS[comp]["Tc"] for comp in df_mezcla["Componente"]]
                Pc_lista = [PROPIEDADES_CRITICAS[comp]["Pc_bar"] for comp in df_mezcla["Componente"]]
                lambda_m = wassiljewa_mason_saxena(y, lambda_i, M_gmol, Tc_lista, Pc_lista, T_extra)
            else:
                Tb_lista = [T_EBULLICION[comp] for comp in df_mezcla["Componente"]]
                lambda_m = lindsay_bromley(y, lambda_i, Cp_i, gamma_i, M_gmol, Tb_lista, T_extra)
            
            nuevo = pd.DataFrame([{"T (K)": T_extra, "Regla_Mezcla": regla, "lambda_calc (W/m·K)": round(lambda_m, 6)}])
            st.session_state.tabla_mezclas = pd.concat([st.session_state.tabla_mezclas, nuevo], ignore_index=True)
            st.success(f"✅ Agregado: λ = {lambda_m:.6f} W/m·K a T = {T_extra} K")
            del st.session_state.temp_agregar
        except Exception as e:
            st.error(f"Error: {e}")
    
    # Tabla de resultados y gráfico
    if not st.session_state.tabla_mezclas.empty:
        st.write("---")
        st.subheader("📈 Historial y Gráfico λ vs Temperatura")
        
        df_actual = st.session_state.tabla_mezclas[st.session_state.tabla_mezclas["Regla_Mezcla"] == regla]
        if not df_actual.empty:
            st.dataframe(df_actual.sort_values("T (K)"), use_container_width=True)
            
            df_sorted = df_actual.sort_values("T (K)")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(x=df_sorted["T (K)"], y=df_sorted["lambda_calc (W/m·K)"], mode='lines+markers', marker=dict(size=8, color='#0056b3'), line=dict(width=2), name=f"{regla}"))
            fig_temp.update_layout(title=f"Conductividad Térmica vs Temperatura<br><sub>{regla}</sub>", xaxis_title="Temperatura (K)", yaxis_title="λ (W/m·K)", template="plotly_white")
            st.plotly_chart(fig_temp, use_container_width=True)
        
        if st.button("🗑️ Limpiar Memoria de Mezclas"):
            st.session_state.tabla_mezclas = pd.DataFrame(columns=["T (K)", "Regla_Mezcla", "lambda_calc (W/m·K)"])
            st.rerun()

# ==============================================================================
# FOOTER
# ==============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("📚 **Referencias**")
st.sidebar.markdown("- Poling, B. E. et al. (2001). *The Properties of Gases and Liquids* (5th ed.).")
st.sidebar.markdown("- Hopp, M. & Gross, J. (2019). *I&EC Research*, 58(45), 20816-20828.")
