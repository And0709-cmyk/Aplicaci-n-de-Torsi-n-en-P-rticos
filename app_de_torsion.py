import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
import string

# CLASES Y FUNCIONES 
@dataclass
class Columna:
    id: str
    x: float
    y: float
    kx: float
    ky: float

class AnalisisTorsion:
    def __init__(self, columnas, Hx, Hy, Vx, Vy, Dx, Dy, factor_amplificacion=1.5):
        self.columnas = columnas
        self.Hx, self.Hy = Hx, Hy
        self.Vx, self.Vy = Vx, Vy  
        self.Dx, self.Dy = Dx, Dy
        self.fa = factor_amplificacion
        self.df = pd.DataFrame([c.__dict__ for c in columnas])
        self._centro_rigidez()

    def _centro_rigidez(self):
        df = self.df
        self.sum_kx = df.kx.sum()
        self.sum_ky = df.ky.sum()
        self.X_R = (df.ky * df.x).sum() / self.sum_ky if self.sum_ky != 0 else 0
        self.Y_R = (df.kx * df.y).sum() / self.sum_kx if self.sum_kx != 0 else 0

    def excentricidad_teorica(self):
        return self.Vx - self.X_R, self.Vy - self.Y_R

    def excentricidad_accidental(self):
        return 0.05 * self.Dx, 0.05 * self.Dy

    def momento_polar(self):
        df = self.df
        return (df.kx * (df.y - self.Y_R) ** 2).sum() + \
               (df.ky * (df.x - self.X_R) ** 2).sum()

    def _cortante_por_torsion(self, Mt, kx, ky, x, y, J):
        if J == 0: return 0, 0
        return Mt * kx * (y - self.Y_R) / J, Mt * ky * (x - self.X_R) / J

    def calcular(self):
        df = self.df.copy()
        J = self.momento_polar()
        e_x_teor, e_y_teor = self.excentricidad_teorica()
        e_x_acc, e_y_acc = self.excentricidad_accidental()

        Mtx_teor, Mty_teor = self.Hx * e_y_teor, self.Hy * e_x_teor
        Mtx_acc, Mty_acc = self.Hx * e_y_acc, self.Hy * e_x_acc

        df["Vx_directo"] = (df.kx / self.sum_kx * self.Hx) if self.sum_kx != 0 else 0
        vx_t, _ = self._cortante_por_torsion(Mtx_teor, df.kx, df.ky, df.x, df.y, J)
        vx_a, _ = self._cortante_por_torsion(Mtx_acc, df.kx, df.ky, df.x, df.y, J)
        df["Vx_teor"], df["Vx_acc"] = vx_t, vx_a.abs()
        df["Vx_diseno"] = self._envolvente(df.Vx_directo, df.Vx_teor, df.Vx_acc)

        df["Vy_directo"] = (df.ky / self.sum_ky * self.Hy) if self.sum_ky != 0 else 0
        _, vy_t = self._cortante_por_torsion(Mty_teor, df.kx, df.ky, df.x, df.y, J)
        _, vy_a = self._cortante_por_torsion(Mty_acc, df.kx, df.ky, df.x, df.y, J)
        df["Vy_teor"], df["Vy_acc"] = vy_t, vy_a.abs()
        df["Vy_diseno"] = self._envolvente(df.Vy_directo, df.Vy_teor, df.Vy_acc)

        self.resultado = df
        self.J = J
        self.momentos = dict(Mtx_teor=Mtx_teor, Mty_teor=Mty_teor, Mtx_acc=Mtx_acc, Mty_acc=Mty_acc)
        return df

    def _envolvente(self, v_directo, v_teor, v_acc):
        amplificado = v_directo + self.fa * v_teor + v_acc
        reducido = v_directo + v_teor + v_acc
        envolvente = np.where(v_teor >= 0, amplificado, reducido)
        return np.maximum(envolvente, v_directo)

def calcular_centros_cortante(datos_pisos, total_pisos):
    cortantes_acumulados = {}
    Hx_acum, Hy_acum = 0.0, 0.0
    sum_Fy_Xg, sum_Fx_Yg = 0.0, 0.0
    
    for p in range(total_pisos, 0, -1):
        if p not in datos_pisos:
            continue
        
        datos = datos_pisos[p]
        Fx, Fy = datos['Fx'], datos['Fy']
        CMx, CMy = datos['CMx'], datos['CMy']
        
        Hx_acum += Fx
        Hy_acum += Fy
        
        sum_Fy_Xg += (Fy * CMx)
        sum_Fx_Yg += (Fx * CMy)
        
        Vx = sum_Fy_Xg / Hy_acum if Hy_acum != 0 else 0
        Vy = sum_Fx_Yg / Hx_acum if Hx_acum != 0 else 0
        
        cortantes_acumulados[p] = {
            'Hx': Hx_acum, 'Hy': Hy_acum, 
            'Vx': Vx, 'Vy': Vy
        }
        
    return cortantes_acumulados


# INTERFAZ (STREAMLIT)

st.set_page_config(page_title="Torsión en Planta - Multi-piso", layout="wide")
st.title("Análisis de Torsión en Planta (Edificios Multi-piso)")

# BARRA LATERAL
st.sidebar.header("1. Geometría Global")
num_pisos = st.sidebar.number_input("Número de pisos", min_value=1, max_value=10, value=2)
num_x = st.sidebar.number_input("Cantidad de Ejes X", min_value=2, max_value=10, value=3)
num_y = st.sidebar.number_input("Cantidad de Ejes Y", min_value=2, max_value=10, value=2)

ejes_x_names = [str(i) for i in range(1, num_x + 1)]
ejes_y_names = list(string.ascii_uppercase)[:num_y]

st.sidebar.subheader("Coordenadas de Ejes X (m)")
df_ejes_x = pd.DataFrame({"Eje": ejes_x_names, "Coord X": [i*8.0 for i in range(num_x)]})
df_ejes_x = st.sidebar.data_editor(df_ejes_x, hide_index=True, use_container_width=True)

st.sidebar.subheader("Coordenadas de Ejes Y (m)")
df_ejes_y = pd.DataFrame({"Eje": ejes_y_names, "Coord Y": [i*8.0 for i in range(num_y)]})
df_ejes_y = st.sidebar.data_editor(df_ejes_y, hide_index=True, use_container_width=True)

coord_x_dict = dict(zip(df_ejes_x["Eje"], df_ejes_x["Coord X"]))
coord_y_dict = dict(zip(df_ejes_y["Eje"], df_ejes_y["Coord Y"]))

Dx = max(df_ejes_x["Coord X"]) - min(df_ejes_x["Coord X"])
Dy = max(df_ejes_y["Coord Y"]) - min(df_ejes_y["Coord Y"])

st.sidebar.divider()
fa = st.sidebar.number_input("Factor de Amplificación Torsional", value=1.0, step=0.1)

# AREA PRINCIPAL: CONFIGURACION POR PISOS
st.header("2. Configuración por Nivel")
matriz_planta_inicial = pd.DataFrame(True, index=ejes_y_names[::-1], columns=ejes_x_names)
tabs = st.tabs([f"Piso {i}" for i in range(1, num_pisos + 1)])
resultados_por_piso = {}

for i, tab in enumerate(tabs):
    piso = i + 1
    with tab:
        col_datos, col_planta = st.columns([1, 2])
        
        with col_datos:
            st.subheader("Datos Sísmicos de la Planta")
            
            Fx = st.number_input(f"Fuerza Sismica Piso Fx (ton)", value=10.0, key=f"Fx_{piso}")
            Fy = st.number_input(f"Fuerza Sismica Piso Fy (ton)", value=10.0, key=f"Fy_{piso}")
            CMx = st.number_input(f"Centro de Masas CMx (m)", value=8.0 if piso==1 else 12.0, key=f"CMx_{piso}")
            CMy = st.number_input(f"Centro de Masas CMy (m)", value=4.0, key=f"CMy_{piso}")
        
        with col_planta:
            st.subheader("Vista en Planta (Asignar Columnas)")
            matriz_planta_editada = st.data_editor(
                matriz_planta_inicial, key=f"planta_{piso}", use_container_width=True
            )
        
        columnas_activas = []
        for y_eje in matriz_planta_editada.index:
            for x_eje in matriz_planta_editada.columns:
                if matriz_planta_editada.loc[y_eje, x_eje]: 
                    columnas_activas.append({
                        "id": f"{y_eje}{x_eje}",
                        "x": coord_x_dict[x_eje],
                        "y": coord_y_dict[y_eje],
                        "kx": 12.0, "ky": 12.0
                    })
        
        if len(columnas_activas) > 0:
            st.subheader("Rigideces de columnas")
            df_cols_editadas = st.data_editor(
                pd.DataFrame(columnas_activas), key=f"rigideces_{piso}", 
                hide_index=True, use_container_width=True
            )
        
            resultados_por_piso[piso] = {
                "cols": df_cols_editadas, "Fx": Fx, "Fy": Fy, "CMx": CMx, "CMy": CMy
            }


# EJECUCIÓN Y RESULTADOS

st.divider()
if st.button("Procesar Análisis Torsional para todos los pisos", type="primary", use_container_width=True):
    
    st.header("3. Resultados de Diseño (Desglose Paso a Paso)")
    
    cortantes_globales = calcular_centros_cortante(resultados_por_piso, num_pisos)
    
    for piso, datos in resultados_por_piso.items():
        st.markdown(f"## Resumen del Piso {piso}")
        
        c_piso = cortantes_globales[piso]
        
        df_cols_actual = datos["cols"]
        
        Dx_local = df_cols_actual['x'].max() - df_cols_actual['x'].min() if not df_cols_actual.empty else 0
        Dy_local = df_cols_actual['y'].max() - df_cols_actual['y'].min() if not df_cols_actual.empty else 0
        
        columnas_obj = [Columna(**row) for row in df_cols_actual.to_dict('records')]
        
        try:
            analisis = AnalisisTorsion(
                columnas=columnas_obj, Hx=c_piso['Hx'], Hy=c_piso['Hy'], 
                Vx=c_piso['Vx'], Vy=c_piso['Vy'], Dx=Dx_local, Dy=Dy_local, factor_amplificacion=fa
            )
            df_res = analisis.calcular()

            st.markdown("Cálculos Previos")
            c_m, c_c, c_r = st.columns(3)
            c_m.info(f"**Centro de Masas (CM):**\nCM_x = {datos['CMx']:.3f} m\nCM_y = {datos['CMy']:.3f} m")
            st.markdown("Cálculos Previos")
            c_m, c_c, c_r = st.columns(3)
            c_m.info(f"**Centro de Masas (CM):**\nCM_x = {datos['CMx']:.3f} m\nCM_y = {datos['CMy']:.3f} m")
            c_c.success(f"**Centro de Cortante (V):**\nV_x = {analisis.Vx:.3f} m\nV_y = {analisis.Vy:.3f} m")
            c_r.error(f"**Centro de Rigidez (R):**\nR_x = {analisis.X_R:.3f} m\nR_y = {analisis.Y_R:.3f} m")
            
            c1, c2 = st.columns(2)
            c1.success(f"**Momento Polar de Inercia:**\nJ = {analisis.J:.3f}")
            m = analisis.momentos
            c2.warning(f"**Momentos Torsores:**\nTeóricos: Mtx = {m['Mtx_teor']:.2f}, Mty = {m['Mty_teor']:.2f}\nAccidentales: Mtx = ±{m['Mtx_acc']:.2f}, Mty = ±{m['Mty_acc']:.2f}")
            
            # TABLA 1: CORTANTE DIRECTO
            st.subheader("A) Fuerza Cortante Directo ($V^\Delta$)")
            df_directo = df_res[["id", "kx", "ky", "Vx_directo", "Vy_directo"]].copy()
            st.dataframe(df_directo.round(3), use_container_width=True)
            
            # TABLA 2: TORSIÓN TEÓRICA
            st.subheader("B) Fuerza Cortante por Torsión Teórica ($V^\theta$)")
            df_teorico = df_res[["id", "x", "y", "Vx_teor", "Vy_teor"]].copy()
            st.dataframe(df_teorico.round(3), use_container_width=True)
            
            # TABLA 3: TORSIÓN ACCIDENTAL
            st.subheader("C) Fuerza Cortante por Torsión Accidental ($V^{acc}$)")
            df_acc = df_res[["id", "x", "y", "Vx_acc", "Vy_acc"]].copy()
            df_acc.rename(columns={"Vx_acc": "Vx_acc (±)", "Vy_acc": "Vy_acc (±)"}, inplace=True)
            st.dataframe(df_acc.round(3), use_container_width=True)
            
            # TABLA 4: DISEÑO FINAL 
            st.subheader("D) Fuerza Cortante de Diseño Final ($V^{diseño}$)")
            cols_mostrar = ["id", "Vx_directo", "Vx_teor", "Vx_acc", "Vx_diseno", "Vy_directo", "Vy_teor", "Vy_acc", "Vy_diseno"]
            st.dataframe(df_res[cols_mostrar].round(3), use_container_width=True)
            
            st.divider()
            
        except Exception as e:
            st.error(f"Error procesando el Piso {piso}: {e}")
