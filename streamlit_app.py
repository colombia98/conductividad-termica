import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
COMPUESTOS = ["Hidrógeno", "Metano", "Etano", "Etileno", "Acetileno", "Propileno", "Propano", "n-Butano"]

COMPUESTOS_INFO = {
    "Hidrógeno": {"Formula": "H2", "M_gmol": 2.016, "Tc": 32.98, "Pc_bar": 12.93, "w": -0.217},
    "Metano": {"Formula": "CH4", "M_gmol": 16.043, "Tc": 190.56, "Pc_bar": 45.98, "w": 0.011},
    "Etano": {"Formula": "C2H6", "M_gmol": 30.070, "Tc": 282.34, "Pc_bar": 50.41, "w": 0.099},
    "Etileno": {"Formula": "C2H4", "M_gmol": 28.054, "Tc": 305.32, "Pc_bar": 48.72, "w": 0.087},
    "Acetileno": {"Formula": "C2H2", "M_gmol": 26.038, "Tc": 308.30, "Pc_bar": 61.14, "w": 0.189},
    "Propileno": {"Formula": "C3H6", "M_gmol": 42.081, "Tc": 364.90, "Pc_bar": 46.00, "w": 0.142},
    "Propano": {"Formula": "C3H8", "M_gmol": 44.097, "Tc": 369.83, "Pc_bar": 42.48, "w": 0.152},
    "n-Butano": {"Formula": "C4H10", "M_gmol": 58.123, "Tc": 425.12, "Pc_bar": 37.96, "w": 0.200}
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
    """Hidrógeno - polinomio cúbico"""
    return 5.417e-10 * T**3 - 9.068e-7 * T**2 + 8.829e-4 * T - 1.162e-2

def k_CH4(T):
    """Metano - polinomio cúbico"""
    return -3.333e-11 * T**3 + 1.800e-7 * T**2 + 4.233e-5 * T + 6.400e-3

def k_C2H6(T):
    """Etano - polinomio cúbico"""
    return -2.500e-10 * T**3 + 4.200e-7 * T**2 - 5.950e-5 * T + 8.300e-3

def k_C2H4(T):
    """Etileno - polinomio cúbico"""
    return -2.250e-10 * T**3 + 3.711e-7 * T**2 + 1.036e-5 * T + 5.980e-3

def k_C2H2(T):
    """Acetileno - polinomio cúbico"""
    return -1.500e-10 * T**3 + 1.450e-7 * T**2 + 7.500e-5 * T + 1.800e-3

def k_C3H6(T):
    """Propileno - polinomio cuadrático"""
    return 1.000e-7 * T**2 + 6.600e-5 * T - 1.180e-2

def k_C3H8(T):
    """Propano - polinomio cúbico"""
    return -2.667e-10 * T**3 + 4.300e-7 * T**2 - 7.333e-5 * T + 8.500e-3

def k_C4H10(T):
    """n-Butano - polinomio cuadrático"""
    return 1.550e-7 * T**2 + 7.500e-6 * T + 5.000e-4

def k_compuesto(compuesto, T):
    """Función general para obtener k(T) de cualquier compuesto"""
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
# Constantes de Shomate (Tabla 16)
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
    """Capacidad calorífica a presión constante usando ecuación de Shomate"""
    R = 8.314
    s = SHOMATE[compuesto]
    t = T / 1000
    Cp = s["A"] + s["B"]*t + s["C"]*t**2 + s["D"]*t**3 + s["E"]/t**2
    return Cp  # J/mol·K

def cv_from_cp(cp, R=8.314):
    """Cv = Cp - R para gas ideal"""
    return cp - R

# ==============================================================================
# FUNCIONES DE REGLAS DE MEZCLA (SEGÚN DOCUMENTO)
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
    
    # Obtener propiedades a la temperatura T
    lambda_i = []
    M_gmol = []
    Tc = []
    Pc_bar = []
    
    for i, comp in enumerate(comps):
        if y[i] > 0:
            lambda_i.append(k_compuesto(comp, T))
            M_gmol.append(COMPUESTOS_INFO[comp]["M_gmol"])
            Tc.append(COMPUESTOS_INFO[comp]["Tc"])
            Pc_bar.append(COMPUESTOS_INFO[comp]["Pc_bar"])
        else:
            lambda_i.append(0)
            M_gmol.append(COMPUESTOS_INFO[comp]["M_gmol"])
            Tc.append(COMPUESTOS_INFO[comp]["Tc"])
            Pc_bar.append(COMPUESTOS_INFO[comp]["Pc_bar"])
    
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
    
    # Obtener propiedades a la temperatura T
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
            # γ = Cp/Cv, pero Cv = Cp - R
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
                # Cociente de viscosidades según Eucken modificado
                visc_ratio = (lambda_i[i] / lambda_i[j]) * (Cp_i[j] / Cp_i[i])
                visc_ratio *= (9 - 5/gamma_i[i]) / (9 - 5/gamma_i[j])
                
                # Factor de masa molecular (Mj/Mi)^(3/4)
                mass_factor = (M_gmol[j] / M_gmol[i])**0.75
                
                # Factor de Sutherland [(1 + Si/T)/(1 + Sj/T)]^(1/2)
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
# INICIALIZACIÓN DE SESSION STATE
# ==============================================================================
if 'tabla_mezclas_wass' not in st.session_state:
    st.session_state.tabla_mezclas_wass = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])

if 'tabla_mezclas_lindsay' not in st.session_state:
    st.session_state.tabla_mezclas_lindsay = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])

# ==============================================================================
# INTERFAZ PRINCIPAL
# ==============================================================================
st.title("Determinación de Conductividad Térmica - FTIQ")
st.info("💡 Este software estima la conductividad térmica de mezclas gaseosas usando las reglas de Wassiljewa-Mason-Saxena y Lindsay-Bromley. Las propiedades k(T), Cp(T) y Cv(T) se calculan automáticamente con las correlaciones del documento.")

st.sidebar.header("Navegación")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

# ==============================================================================
# MÓDULO 1: SUSTANCIAS PURAS (k vs T)
# ==============================================================================
if menu == "Sustancias Puras":
    st.header("Conductividad Térmica de Componentes Puros vs Temperatura")
    st.info("📊 Las siguientes gráficas muestran la conductividad térmica de los componentes puros en función de la temperatura según las correlaciones del documento.")
    
    # Selección de compuestos
    compuestos_seleccionados = st.multiselect(
        "Seleccione los compuestos a graficar:",
        COMPUESTOS,
        default=["Hidrógeno", "Metano", "Etano", "Etileno"]
    )
    
    # Rango de temperatura
    col_min, col_max = st.columns(2)
    with col_min:
        T_min = st.number_input("Temperatura mínima (K):", value=290.0, step=10.0)
    with col_max:
        T_max = st.number_input("Temperatura máxima (K):", value=410.0, step=10.0)
    
    if T_max <= T_min:
        st.error("La temperatura máxima debe ser mayor que la mínima")
    else:
        # Generar datos
        T_range = np.linspace(T_min, T_max, 50)
        
        # Crear gráfico
        fig = go.Figure()
        
        for comp in compuestos_seleccionados:
            k_vals = [k_compuesto(comp, T) for T in T_range]
            fig.add_trace(go.Scatter(
                x=T_range, y=k_vals,
                mode='lines',
                name=comp,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title="Conductividad Térmica de Componentes Puros vs Temperatura",
            xaxis_title="Temperatura (K)",
            yaxis_title="λ (W/m·K)",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar valores a una temperatura específica
        st.subheader("Valores a temperatura específica")
        T_esp = st.number_input("Temperatura (K):", value=300.0, step=10.0, key="T_esp")
        
        datos_compuestos = []
        for comp in COMPUESTOS:
            k_val = k_compuesto(comp, T_esp)
            Cp_val = cp_shomate(comp, T_esp)
            Cv_val = cp_shomate(comp, T_esp) - 8.314
            datos_compuestos.append({
                "Compuesto": comp,
                "λ (W/m·K)": f"{k_val:.6f}",
                "Cp (J/mol·K)": f"{Cp_val:.3f}",
                "Cv (J/mol·K)": f"{Cv_val:.3f}"
            })
        
        st.dataframe(pd.DataFrame(datos_compuestos), use_container_width=True)

# ==============================================================================
# MÓDULO 2: REGLAS DE MEZCLADO (AUTOMATIZADO)
# ==============================================================================
elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Gaseosas Multicomponentes")
    
    # Mostrar ecuaciones
    with st.expander("📖 Ver Ecuaciones de las Reglas de Mezclado", expanded=False):
        st.latex(r"\lambda_m = \sum_{i=1}^{n} \frac{y_i \lambda_i}{\sum_{j=1}^{n} y_j A_{ij}}")
        st.markdown("**Wassiljewa-Mason-Saxena:**")
        st.latex(r"A_{ij} = \frac{\left[1 + \left(\frac{\lambda_{tr,i}}{\lambda_{tr,j}}\right)^{1/2}\left(\frac{M_j}{M_i}\right)^{1/4}\right]^2}{\left[8\left(1 + \frac{M_i}{M_j}\right)\right]^{1/2}}")
        st.markdown("**Lindsay-Bromley:**")
        st.latex(r"A_{ij} = \frac{1}{4}\left\{1 + \left[\frac{\mu_i}{\mu_j}\left(\frac{M_j}{M_i}\right)^{3/4}\left(\frac{1+S_i/T}{1+S_j/T}\right)^{1/2}\right]^2\right\}\left(\frac{1+S_{ij}/T}{1+S_i/T}\right)")
    
    # Composición de la mezcla
    st.subheader("📋 Composición de la Mezcla")
    st.info("La composición corresponde al gas de cracking de etano (Tabla 9 del documento)")
    
    # Mostrar composición estándar
    df_composicion = pd.DataFrame([
        {"Compuesto": comp, "Fracción Molar (yᵢ)": COMPOSICION_CRACKING[comp]}
        for comp in COMPUESTOS
    ])
    
    # Permitir editar composición
    df_edit = st.data_editor(df_composicion, num_rows="fixed", use_container_width=True)
    
    # Normalizar fracciones molares
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
        T_min = st.number_input("Temperatura mínima (K):", value=290.0, step=10.0, key="mix_Tmin")
    with col_Tmax:
        T_max = st.number_input("Temperatura máxima (K):", value=410.0, step=10.0, key="mix_Tmax")
    with col_step:
        T_step = st.number_input("Incremento (K):", value=10.0, step=5.0, key="mix_Tstep")
    
    if st.button("🚀 Calcular Conductividad de la Mezcla en el Rango de Temperatura", type="primary", use_container_width=True):
        if T_max <= T_min:
            st.error("La temperatura máxima debe ser mayor que la mínima")
        else:
            temperaturas = np.arange(T_min, T_max + T_step/2, T_step)
            
            resultados_wass = []
            resultados_lindsay = []
            
            with st.spinner("Calculando..."):
                for T in temperaturas:
                    try:
                        lambda_wass = wassiljewa_mason_saxena(y, T)
                        lambda_lindsay = lindsay_bromley(y, T)
                        resultados_wass.append({"T (K)": T, "lambda_calc (W/m·K)": lambda_wass})
                        resultados_lindsay.append({"T (K)": T, "lambda_calc (W/m·K)": lambda_lindsay})
                    except Exception as e:
                        st.error(f"Error a T={T} K: {e}")
            
            # Guardar resultados
            st.session_state.tabla_mezclas_wass = pd.DataFrame(resultados_wass)
            st.session_state.tabla_mezclas_lindsay = pd.DataFrame(resultados_lindsay)
            
            st.success(f"✅ Cálculo completado para {len(temperaturas)} temperaturas")
    
    # Mostrar resultados y gráficos
    st.write("---")
    st.subheader("📈 Resultados y Gráfico λ vs Temperatura")
    
    regla_seleccionada = st.radio(
        "Seleccione la regla de mezclado para visualizar:",
        ["Wassiljewa-Mason-Saxena", "Lindsay-Bromley"],
        horizontal=True
    )
    
    if regla_seleccionada == "Wassiljewa-Mason-Saxena":
        df_resultados = st.session_state.tabla_mezclas_wass
        color = '#0056b3'
    else:
        df_resultados = st.session_state.tabla_mezclas_lindsay
        color = '#e67e22'
    
    if not df_resultados.empty:
        st.dataframe(df_resultados.sort_values("T (K)").round(6), use_container_width=True)
        
        # Gráfico con puntos y línea punteada
        df_sorted = df_resultados.sort_values("T (K)")
        x_temp = df_sorted["T (K)"].values
        y_lambda = df_sorted["lambda_calc (W/m·K)"].values
        
        # Regresión lineal
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
                name=f'{regla_seleccionada} (calculado)'
            ))
            
            # Regresión lineal
            fig.add_trace(go.Scatter(
                x=x_lineal, y=y_lineal,
                mode='lines',
                line=dict(color='red', dash='dot', width=2),
                name=f'Regresión: λ = {slope:.6f}T + {intercept:.6f}'
            ))
            
            fig.update_layout(
                title=f"Conductividad Térmica vs Temperatura<br><sub>{regla_seleccionada} | Ecuación: λ = {slope:.6f}·T + {intercept:.6f} | R² = {r2:.6f}</sub>",
                xaxis_title="Temperatura (K)",
                yaxis_title="λ (W/m·K)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar ecuación y R²
            st.markdown(f"""
            <div class='intermedio'>
            <b>📊 Resultados de Regresión Lineal ({regla_seleccionada}):</b><br>
            • Ecuación: λ = {slope:.8f} × T + {intercept:.8f}<br>
            • Coeficiente de Determinación (R²): {r2:.6f}
            </div>
            """, unsafe_allow_html=True)
    
    # Botón para limpiar
    if st.button("🗑️ Limpiar todos los resultados"):
        st.session_state.tabla_mezclas_wass = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])
        st.session_state.tabla_mezclas_lindsay = pd.DataFrame(columns=["T (K)", "lambda_calc (W/m·K)"])
        st.rerun()
    
    # Mostrar tabla de resultados esperados del documento (Tabla 17)
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
