import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector.constants import ClientFlag
import os, certifi
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import uuid
from datetime import datetime
import pytz
import joblib
import matplotlib.pyplot as plt

import sklearn.compose._column_transformer as _ct

class _RemainderColsList(list):
    """Dummy stub para compatibilidad al cargar el ColumnTransformer serializado."""
    pass

# Registrar nuestra stub en el módulo original
_ct._RemainderColsList = _RemainderColsList

# ——————————————————————————————————————————————————————————————————————
# 1) Carga de .env
load_dotenv(find_dotenv())

# 1b) Carga del modelo entrenado
@st.cache_resource
def load_model():
    # __file__ apunta a app.py
    base_dir   = Path(__file__).parent
    model_path = base_dir / "models" / "modelo_imputacion_uene.pkl"

    if not model_path.exists():
        st.error(f"❌ Modelo no encontrado en:\n{model_path}")
        st.stop()

    return joblib.load(model_path)

model = load_model()  # ← Cargamos el pipeline

# 2) Configuración de Streamlit
st.set_page_config(
    page_title="Presupuesto EMCALI",
    page_icon="images/LOGO-EMCALI-vertical-color_1.png",
    layout="centered"
)
st.sidebar.image("images/LOGO-EMCALI-vertical-color.png", use_container_width=True)

st.markdown("""
<style>
:root{
  --brand:#ef5f17;   /* primario */
  --accent:#f8cf3f;  /* acento   */
  --ink:#1f2937;
  --bg:#fffef7;
}

/* Fondo general y títulos */
body{ background:var(--bg) }
h1,h2,h3,h4{ color:var(--ink); letter-spacing:.2px }
h3{ font-size:1.08rem; margin:.35rem 0 .4rem; font-weight:700 }

/* Sidebar con acento */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#fff 0%,#fffef8 100%);
  border-right:6px solid var(--brand);
}
[data-testid="stSidebar"]::before{
  content:""; display:block; height:6px; background:linear-gradient(90deg,var(--accent),var(--brand));
}

/* Formulario con franja acento */
[data-testid="stForm"]{
  border:1px solid #f2f2f2; background:#fff; border-radius:14px;
  padding:1rem 1rem .8rem; position:relative;
}
[data-testid="stForm"]::before{
  content:""; position:absolute; left:0; top:0; right:0; height:4px;
  background:linear-gradient(90deg,var(--accent),var(--brand));
  border-top-left-radius:14px; border-top-right-radius:14px;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] .css-1dimb5e{ border-radius:10px !important }
input[type="radio"], input[type="checkbox"]{ accent-color: var(--brand); }

/* Botón primario con gradiente brand→accent */
.stButton > button{
  background:linear-gradient(135deg,var(--brand),var(--accent));
  color:#fff; font-weight:700; border:0; border-radius:12px;
  padding:.62rem 1.1rem; box-shadow:0 3px 10px rgba(239,95,23,.18);
  transition:transform .06s ease, filter .15s ease;
}
.stButton > button:hover:enabled{ filter:brightness(.98); transform:translateY(-1px) }
.stButton > button:disabled{ opacity:.45; cursor:not-allowed }

/* Métricas (tarjetas) con borde acento */
[data-testid="stMetric"]{
  border:1px solid #f1f1f1; border-left:5px solid var(--accent);
  border-radius:12px; background:#fff; padding:.75rem 1rem;
}
[data-testid="stMetricValue"]{ font-size:1.08rem }
[data-testid="stMetricLabel"]{ font-size:.86rem; color:#6b7280 }

/* Alertas */
[data-testid="stAlert"]{ border-radius:10px }
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong{
  color:var(--brand);
}

/* Tablas */
[data-testid="stDataFrame"] table{ font-size:.92rem }
[data-testid="stDataFrame"] thead th{
  position:sticky; top:0; background:#fff; box-shadow:inset 0 -1px 0 #eee;
}

/* Links y pequeños detalles */
a{ color:var(--brand) }
.small-note{ color:#6b7280; font-size:.85rem }

/* Responsive */
@media (max-width: 900px){
  [data-testid="column"]{ width:100% !important; flex:1 1 100% !important }
}
</style>
""", unsafe_allow_html=True)

# 3) Inicializar session_state
if "logueado" not in st.session_state:
    st.session_state["logueado"]         = False
    st.session_state["usuario"]          = None
    st.session_state["session_id"]       = None
    st.session_state["nombre_completo"]  = ""   # cadena vacía por defecto

# justo donde chequeas session_state al arrancar la app
if "datos" not in st.session_state:
    st.session_state["datos"] = pd.DataFrame(columns=[
        "id","item","categoria","grupo","centro_gestor","unidad_codigo",
        "concepto_gasto","descripcion","cantidad","valor_unitario",
        "fecha_inicio","created_by","imputacion"
    ])

if "contador_item" not in st.session_state:
    st.session_state["contador_item"] = 1

# ——————————————————————————————————————————————————————————————————————
def conectar_db():
    """
    Abre una conexión a PlanetScale con SSL, autocommit y timeout.
    Lee:
      - PLANETSCALE_HOST
      - PLANETSCALE_PORT (default 3306)
      - PLANETSCALE_USER
      - PLANETSCALE_PASSWORD
      - PLANETSCALE_DATABASE
      - PLANETSCALE_SSL_CA (opcional; si no existe, usa certifi.where())
    """
    host     = os.getenv("PLANETSCALE_HOST")
    port     = int(os.getenv("PLANETSCALE_PORT", 3306))
    user     = os.getenv("PLANETSCALE_USER")
    pwd      = os.getenv("PLANETSCALE_PASSWORD")
    db       = os.getenv("PLANETSCALE_DATABASE")
    ssl_ca   = os.getenv("PLANETSCALE_SSL_CA", "")

    # Si la ruta a PLANETSCALE_SSL_CA no existe, usamos el bundle de certifi
    if not ssl_ca or not Path(ssl_ca).is_file():
        ssl_ca = certifi.where()

    try:
        conn = mysql.connector.connect(
            host               = host,
            port               = port,
            user               = user,
            password           = pwd,
            database           = db,
            ssl_ca             = ssl_ca,
            client_flags       = [ClientFlag.SSL],
            connection_timeout = 30,      # tiempo máximo para conectar
            use_pure           = True
        )
        # Habilitamos autocommit para no tener que llamar a commit() manualmente
        conn.autocommit = True
        return conn

    except mysql.connector.Error as err:
        # Manejo de errores más explícito
        st.error(f"🔌 Error al conectar a la base de datos: {err}")
        raise

# ——————————————————————————————————————————————————————————————————————
# 5) Validación de credenciales
def validar_credenciales(username, password):
    try:
        cn  = conectar_db()
        cur = cn.cursor()
        cur.execute(
            "SELECT 1 FROM usuarios WHERE username=%s AND password=%s",
            (username, password)
        )
        ok = cur.fetchone() is not None
        cur.close(); cn.close()
        return ok
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return False

# ——————————————————————————————————————————————————————————————————————
# 6) Control de sesiones únicas
def is_user_active(username):
    cn  = conectar_db(); cur = cn.cursor()
    cur.execute("SELECT 1 FROM active_sessions WHERE username=%s", (username,))
    activo = cur.fetchone() is not None
    cur.close(); cn.close()
    return activo

def create_session(username, session_id):
    cn  = conectar_db(); cur = cn.cursor()
    cur.execute(
        "INSERT INTO active_sessions(username, session_id) "
        "VALUES (%s,%s) ON DUPLICATE KEY UPDATE session_id=VALUES(session_id)",
        (username, session_id)
    )
    cur.close(); cn.close()

def end_session(username):
    cn  = conectar_db(); cur = cn.cursor()
    cur.execute("DELETE FROM active_sessions WHERE username=%s", (username,))
    cur.close(); cn.close()

@st.cache_data(ttl=300)
def load_user_records(username: str) -> pd.DataFrame:
    """
    Carga todos los registros del usuario desde presupuesto_registros.
    """
    cn = conectar_db()
    # Query parametrizada para evitar inyección
    df = pd.read_sql(
        "SELECT id, item, categoria, grupo, centro_gestor, unidad_codigo, "
        "concepto_gasto, descripcion, cantidad, valor_unitario, fecha_inicio, "
        "created_by, imputacion "
        "FROM presupuesto_registros "
        "WHERE created_by = %s "
        "ORDER BY id DESC",
        cn,
        params=(username,)
    )
    cn.close()
    return df

#Cuadro Lateral
def load_ingresos():
    cn = conectar_db()
    df = pd.read_sql(
        "SELECT centro_gestor AS centro, ingreso_asignado AS ingreso FROM ingresos_asignados",
        cn
    )
    cn.close()
    return df

def load_gastos():
    cn = conectar_db()
    df = pd.read_sql(
        """
        SELECT unidad_codigo       AS centro,
               SUM(cantidad * valor_unitario) AS gastos
        FROM presupuesto_registros
        GROUP BY unidad_codigo
        """,
        cn
    )
    cn.close()
    return df

# Mapea los prefijos de centro a ger-neg
GER_NEG_MAP = {
    "D52": "5203",
    "C52": "5204",
    "G52": "5205",
    # añade más reglas si es necesario
}

def parse_imputacion(raw: str) -> dict:
    """
    Parse cualquier imputación estilo "D52000.F00C00G0P0FNPNTO.10005.EM9900.0.2120201003.3331101"
    y extrae:
      - centro           (segmento 0)
      - pec              (segmento 1)
      - id_recurso       (segmento 2)
      - proyecto         (segmento 3)
      - pospre           (últimos dos segmentos unidos con '.')
      - grupo            (dos primeros dígitos de pospre)
      - ger_neg          (según prefijo de 'centro' usando GER_NEG_MAP)
    Cualquier parte faltante quedará en None.
    """
    # Limpia puntos sobrantes y espacios
    cleaned = raw.strip().strip(".")
    # Separa y quita segmentos vacíos
    segs = [s for s in cleaned.split(".") if s]

    # Indexación segura
    centro       = segs[0] if len(segs) > 0 else None
    pec          = segs[1] if len(segs) > 1 else None
    id_recurso   = segs[2] if len(segs) > 2 else None
    proyecto     = segs[3] if len(segs) > 3 else None

    # Construye pospre uniendo los dos últimos segmentos
    if len(segs) >= 2:
        pospre = f"{segs[-2]}.{segs[-1]}"
    else:
        pospre = segs[-1] if segs else None

    # Extrae grupo: dos primeros dígitos del primer subsegmento de pospre
    grupo = None
    if pospre:
        primera_parte = pospre.split(".")[0]
        grupo = primera_parte[:2] if len(primera_parte) >= 2 else primera_parte

    # Determina ger-neg a partir de los tres primeros caracteres de centro
    ger_neg = None
    if centro and len(centro) >= 3:
        pref = centro[:3]
        ger_neg = GER_NEG_MAP.get(pref)

    return {
        "centro":      centro,
        "pec":         pec,
        "id_recurso":  id_recurso,
        "proyecto":    proyecto,
        "pospre":      pospre,
        "grupo":       grupo,
        "ger_neg":     ger_neg
    }



# ——————————————————————————————————————————————————————————————————————
# 4) Master data: usuarios_unidades y unidad_concepto_gasto
@st.cache_data(ttl=600)
def load_master_data():
    cn = conectar_db()
    df_u = pd.read_sql("""
        SELECT
          uu.username,
          TRIM(uu.unidad_codigo)   AS unidad_codigo,
          TRIM(uu.unidad_nombre)   AS unidad_nombre,
          COALESCE(u.nombre_completo, uu.unidad_nombre) AS nombre_completo
        FROM usuarios_unidades uu
        LEFT JOIN usuarios u USING(username)
    """, cn)
    df_c = pd.read_sql("""
        SELECT TRIM(unidad_codigo) AS unidad_codigo,
               TRIM(concepto_gasto) AS concepto_gasto
        FROM unidad_concepto_gasto
    """, cn)
    cn.close()
    return df_u, df_c

def get_fullname(username: str) -> str:
    df_u, _ = load_master_data()
    sel = df_u.loc[df_u["username"] == username, "nombre_completo"]
    return sel.iloc[0] if not sel.empty else username

def obtener_unidades(username: str) -> list[str]:
    df_u, _ = load_master_data()
    df_user = df_u[df_u["username"] == username]
    # 2) Devuelvo lista de "codigo + nombre"
    return df_user.apply(lambda row: f"{row['unidad_codigo']} {row['unidad_nombre']}", axis=1).tolist()

def obtener_conceptos(unidad_codigo: str) -> list[str]:
    _, df_c = load_master_data()
    # 3) Filtro sólo los conceptos de esa unidad
    return df_c[df_c["unidad_codigo"] == unidad_codigo]["concepto_gasto"].tolist()

# 8) UI de Login
def mostrar_login():
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("images/Pajaro-Tangara-3.png", width=80)
    with col_title:
        st.title("Inicio de Sesión")

    user = st.text_input("Usuario", key="login_user")
    pwd  = st.text_input("Contraseña", type="password", key="login_pwd")

    if st.button("Iniciar sesión", key="login_btn"):
        if not user or not pwd:
            st.warning("Usuario y contraseña son obligatorios.")
            return

        # Aquí envolvemos la validación en un spinner
        with st.spinner("⏳ Validando credenciales, por favor espera..."):
            # 1) Validar credenciales
            if not validar_credenciales(user, pwd):
                st.error("❌ Usuario o contraseña incorrectos.")
                return

            # 2) Si tiene sesión activa anterior, la cerramos
            if is_user_active(user):
                end_session(user)

            # 3) Creamos nueva sesión
            sid = str(uuid.uuid4())
            create_session(user, sid)

            # 4) Actualizamos el estado y forzamos rerun
            st.session_state["logueado"]   = True
            st.session_state["usuario"]    = user
            st.session_state["session_id"] = sid
            st.session_state["nombre_completo"] = get_fullname(user)

        # Una vez fuera del spinner, ya logueado
        st.success(f"✅ Bienvenido, {st.session_state['nombre_completo']}!")
        st.rerun()

# ——————————————————————————————————————————————————————————————————————
# 9) Sidebar + Menú
def mostrar_sidebar():
    ingresos_df = load_ingresos()
    gastos_df   = load_gastos()
    # Sólo mostramos las Unidades que tiene el usuario
    mis_unidades = obtener_unidades(st.session_state['usuario'])

    # Mezclamos y calculamos saldo
    df_flow = (
        ingresos_df
        .merge(gastos_df, on="centro", how="left")
        .loc[lambda d: d['centro'].isin(mis_unidades)]
    )
    df_flow['gastos'] = df_flow['gastos'].fillna(0)
    df_flow['saldo']  = df_flow['ingreso'] - df_flow['gastos']

    # Renderizamos una “card” por Unidad
    for _, row in df_flow.iterrows():
        # separamos código y descripción de la unidad para mostrar bonito
        partes = row['centro'].split(' ', 1)
        codigo = partes[0]
        nombre = partes[1] if len(partes)>1 else ''
        
        ingreso = f"{row['ingreso']:,.0f}"
        gastos  = f"{row['gastos']:,.0f}"
        saldo   = f"{row['saldo']:,.0f}"
        color   = "#228B22" if row['saldo'] >= 0 else "red"
        
        st.sidebar.markdown(f"""
        <div style="
            border: 2px solid #ef5f17;
            border-radius: 8px;
            padding: 0.75em;
            margin-bottom: 0.5em;
            background: #fff;
        ">
        <div style="font-weight:bold; margin-bottom:0.5em;">
            🏷️ {codigo} — {nombre}
        </div>
        <div>Ingreso: <strong>${ingreso}</strong></div>
        <div>Gastos:  <strong>${gastos}</strong></div>
        <div>Saldo:   <strong style="color:{color}">${saldo}</strong></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Si el saldo es negativo, mostramos una alerta
        if row['saldo'] < 0:
            st.sidebar.error(f"⚠️ Atención: gastos de {nombre} superan el ingreso asignado.")

    st.sidebar.markdown(f"👤 **{st.session_state['nombre_completo']}**")
    if st.sidebar.button("Cerrar sesión", key="logout_btn_sidebar"):
        end_session(st.session_state["usuario"])
        for k in ("logueado","usuario","session_id"):
            st.session_state.pop(k, None)
        st.rerun()

    st.sidebar.markdown("---")
    opciones = ["Agregar","Buscar","Editar","Eliminar","Ver Todo"]
    if st.session_state["usuario"] == "admin":
        opciones.append("Descargar")
    return st.sidebar.selectbox("🔧 Menú", opciones, key="sidebar_menu")

# ——————————————————————————————————————————————————————————————————————
# 10) Flujo principal
if not st.session_state["logueado"]:
    mostrar_login()
    st.stop()

accion = mostrar_sidebar()



# — Título y tabs
col_logo, col_titulo = st.columns([1, 6])  # Ajusta proporciones según el tamaño del logo

with col_logo:
    st.image("images/icono-energia.png", width=80)
    
with col_titulo:
    st.markdown("<h3 style='margin-bottom: 0;'>Gestión Presupuestal UENE 2026</h3>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["📑 Presupuesto", "📊 Dashboard"])

with tab1:
    if accion == "Agregar":
        # —————————————————— STORY / INTRO ——————————————————
        st.markdown("<h3 style='margin-bottom:0.25rem'>➕ Agregar Registro Presupuestal</h3>", unsafe_allow_html=True)
        st.error("⚠️ Tener encuenta LEY DE GARANTIAS inicia el 01 de febrero 2026 y termina 30 de junio 2026.")

        # — Datos base
        ingresos_df = load_ingresos()
        gastos_df   = load_gastos()
        unidades    = obtener_unidades(st.session_state["usuario"])
        if not unidades:
            st.warning("No tienes unidades asignadas.")
            st.stop()

        # Valores 'ocultos' (no se muestran en el formulario)
        grupo  = "21 Funcionamiento"
        centro = unidades[0]
        unidad = unidades[0]

        ingreso_sel = float(ingresos_df.loc[ingresos_df['centro'].eq(centro), 'ingreso'].iloc[0]) if centro in ingresos_df['centro'].values else 0.0
        gasto_sel   = float(gastos_df.loc[gastos_df['centro'].eq(centro),   'gastos' ].iloc[0]) if centro in gastos_df['centro'].values   else 0.0
        saldo_disp  = ingreso_sel - gasto_sel

        # —————————————————— PASO 1: FECHA ——————————————————
        #st.markdown("#### 1) Fecha de inicio del proceso")
        fecha_inicio = st.date_input("Fecha Inicio de Proceso", key="fecha_inicio")

        # Advertencia y confirmación si es enero
        if fecha_inicio.month == 1:
            st.warning(
                "⚠️ **ADVERTENCIA:** Esta es la fecha límite para presentar a la UGA "
                "os documentos requeridos para iniciar la gestión contractual. La no presentación en el plazo será causal de INCUMPLIMIENTO"
            )
            confirm_enero = st.checkbox("Sí, confirmo que inicio en enero", key="confirm_enero")
        else:
            confirm_enero = True  # no aplica confirmación

        

        categoria = st.radio(
            "Categoría",
            ["AGOP", "Excepciones 3.1", "Contratos", "Vigencias Futuras"],
            horizontal=True,
            help="Seleccion Categoria de Contratación"
        )


        # —————————————————— PASO 2: DETALLE ——————————————————
        #st.markdown("#### 2) Detalle del gasto")
        with st.form("form_add", clear_on_submit=True):
            c1, c2 = st.columns(2)

            with c1:
                item = f"GE{st.session_state['contador_item']:04d}"
                st.text_input("Item", value=item, disabled=True)

                # Guarda '3.1' en BD si el radio dijo "Excepciones 3.1"
                categoria_db = "3.1" if categoria == "Excepciones 3.1" else categoria
                st.caption(f"Categoría seleccionada: **{categoria_db}**")

                conceptos = obtener_conceptos(unidad) or ["(sin conceptos disponibles)"]
                concepto  = st.selectbox("Concepto de Gasto", conceptos, help="Elegir el Concepto de Acuerdo a su Gasto.")
                descripcion = st.text_area("Descripción", placeholder="Escribe una descripción breve y clara del gasto…", height=110)

            with c2:
                cantidad   = st.number_input("Cantidad", min_value=1, value=1, key="cant_add")
                valor_unit = st.number_input("Valor Unitario", min_value=0.0, format="%.2f", key="vu_add")
                # OJO: dentro del form esto no se actualizará en vivo (sólo tras submit)
                valor_total = cantidad * valor_unit
                st.markdown(f"**Total calculado:** ${valor_total:,.0f}")
                # No muestres aquí el error de saldo; valida tras submit.

            # Habilita/deshabilita SOLO con la confirmación de enero
            btn_disabled = (fecha_inicio.month == 1 and not confirm_enero)
            st.caption("Verifica que todo esté correcto. Si es enero, marca la confirmación. Luego guarda.")
            enviar = st.form_submit_button("Guardar e Imputar", disabled=btn_disabled)

        # — GUARDAR —
        if enviar:
            # Recalcula con los valores enviados
            valor_total = cantidad * valor_unit

            # Validaciones server-side
            errores = []
            if concepto == "(sin conceptos disponibles)":
                errores.append("Debes seleccionar un concepto válido.")
            if valor_total <= 0:
                errores.append("El total debe ser mayor que 0.")
            if valor_total > saldo_disp:
                errores.append("El total excede tu saldo disponible.")

            if errores:
                st.error(" • ".join(errores))
                st.stop()

            # Predicción
            df_in = pd.DataFrame([{
                "Unidad": unidad,
                "Concepto de Gasto": concepto,
                "Descripcion del Gasto": descripcion
            }])
            pred = model.predict(df_in)[0]

            # INSERT en BD (usando categoria_db)
            cn1, cur1 = conectar_db(), None
            try:
                cur1 = cn1.cursor()
                cur1.execute("""
                    INSERT INTO presupuesto_registros
                    (item,categoria,grupo,centro_gestor,unidad_codigo,
                    concepto_gasto,descripcion,cantidad,valor_unitario,
                    fecha_inicio,created_by,imputacion,accion)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    item, categoria_db, grupo, centro, unidad,
                    concepto, descripcion, cantidad, valor_unit,
                    fecha_inicio, st.session_state["usuario"], pred, "Agregar"
                ))
                inserted_id = cur1.lastrowid
                cn1.commit()
            finally:
                if cur1: cur1.close()
                cn1.close()

            # Historizar en registros_usuarios
            p = parse_imputacion(pred)
            cn2, cur2 = conectar_db(), None
            try:
                cur2 = cn2.cursor()
                cur2.execute("""
                    INSERT INTO registros_usuarios
                    (item,grupo,concepto_gasto,imputacion_raw,descripcion,
                    ger_neg,centro_gestor,pec,id_recurso,proyecto,pospre,
                    cantidad,valor_unitario,total)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    item, p["grupo"], concepto, pred, descripcion,
                    p["ger_neg"], p["centro"], p["pec"],
                    p["id_recurso"], p["proyecto"], p["pospre"],
                    cantidad, valor_unit, valor_total
                ))
                cn2.commit()
            finally:
                if cur2: cur2.close()
                cn2.close()

            # Añadir a la sesión (para la grilla de abajo)
            nueva = pd.DataFrame([{
                "id":             inserted_id,
                "item":           item,
                "categoria":      categoria,
                "grupo":          grupo,
                "centro_gestor":  centro,
                "unidad_codigo":  unidad,
                "concepto_gasto": concepto,
                "descripcion":    descripcion,
                "cantidad":       cantidad,
                "valor_unitario": valor_unit,
                "fecha_inicio":   str(fecha_inicio),
                "created_by":     st.session_state["usuario"],
                "imputacion":     pred
            }])
            st.session_state["datos"] = pd.concat([st.session_state["datos"], nueva], ignore_index=True)
            st.session_state["contador_item"] += 1

            st.success(f"✅ Registro #{inserted_id} guardado. Imputación sugerida: **{pred}**")
            st.rerun()

        # —————————————————— TABLA CON FECHA ESTIMADA ——————————————————
        if not st.session_state["datos"].empty:
            #st.markdown("#### 5) Registros de esta sesión")
            df_s = st.session_state["datos"].copy()
            df_s["fecha_inicio"] = pd.to_datetime(df_s["fecha_inicio"])

            def estimar_fecha_contratacion(row):
                base = row["fecha_inicio"]
                dias = 0
                # por grupo
                if str(row["grupo"]).startswith("21"):
                    dias += 90
                elif str(row["grupo"]).startswith("23"):
                    dias += 150
                # por categoría
                if row.get("categoria") == "AGOP":
                    dias += 15
                # por concepto específico
                if str(row.get("concepto_gasto", "")).startswith("074"):
                    dias += 15
                return base + pd.Timedelta(days=dias)

            df_s["fecha_contratacion"] = df_s.apply(estimar_fecha_contratacion, axis=1).dt.date

            st.dataframe(
                df_s[[
                    "id","item","categoria","grupo","centro_gestor",
                    "concepto_gasto","fecha_inicio","fecha_contratacion",
                    "cantidad","valor_unitario","imputacion"
                ]].rename(columns={
                    "fecha_inicio":"Fecha inicio",
                    "fecha_contratacion":"Fecha contratación",
                    "valor_unitario":"Valor unitario"
                }),
                use_container_width=True,
                height=240
            )


        # — BUSCAR —
    elif accion == "Buscar":
            st.header("🔍 Buscar Registro")
            id_buscar = st.text_input("Ingrese ID de registro")
            if st.button("Buscar"):
                try:
                    cn=conectar_db(); cur=cn.cursor()
                    cur.execute("SELECT * FROM presupuesto_registros WHERE id=%s",(id_buscar,))
                    fila=cur.fetchone(); cn.close()
                    if fila:
                        df = pd.DataFrame([fila], columns=[d[0] for d in cur.description])
                        st.dataframe(df)
                    else:
                        st.warning("No encontrado.")
                except Exception as e:
                    st.error(f"Error: {e}")    
        
    elif accion == "Editar":
            st.header("✏️ Editar Registro")
            id_ed = st.text_input("ID a editar", key="id_ed")
        
            load_flag = f"loaded_{id_ed}"
            data_key  = f"data_{id_ed}"
        
            # 1) Botón para cargar datos
            if st.button("Cargar datos", key="load_edit"):
                cn  = conectar_db()
                cur = cn.cursor()
                cur.execute("""
                    SELECT 
                    id, item, categoria, grupo, centro_gestor, unidad_codigo,
                    concepto_gasto, descripcion, cantidad, valor_unitario,
                    fecha_inicio, imputacion
                    FROM presupuesto_registros
                    WHERE id = %s
                """, (id_ed,))
                row = cur.fetchone()
                cols = [d[0] for d in cur.description]
                cur.close(); cn.close()
        
                if not row:
                    st.warning("❌ El ID no existe.")
                else:
                    st.session_state[data_key] = dict(zip(cols, row))
                    st.session_state[load_flag] = True
        
            # 2) Si ya cargamos los datos, mostramos el formulario
            if st.session_state.get(load_flag, False):
                data = st.session_state[data_key]
        
                with st.form("form_edit", clear_on_submit=True):
                    item2           = st.text_input("Item", value=data["item"], key="item2")
                    categoria2      = st.text_input("Categoría", value=data["categoria"], key="categoria2")
                    grupo2          = st.text_input("Grupo", value=data["grupo"], key="grupo2")
                    centro2         = st.text_input("Centro Gestor", value=data["centro_gestor"], key="centro2")
                    unidad2         = st.text_input("Unidad", value=data["unidad_codigo"], key="unidad2")
                    conc2           = st.text_input("Concepto de Gasto", value=data["concepto_gasto"], key="conc2")
                    desc2           = st.text_area("Descripción del Gasto", value=data["descripcion"], key="desc2")
                    cantidad2       = st.number_input("Cantidad", min_value=0, value=int(data["cantidad"]), key="cantidad2")
                    valor_unit2     = st.number_input("Valor Unitario", min_value=0.0, format="%.2f", value=float(data["valor_unitario"]), key="valor2")
                    fecha_i2        = st.date_input("Fecha Inicio de Proceso", value=pd.to_datetime(data["fecha_inicio"]).date(), key="fecha2")
                    enviar2         = st.form_submit_button("Actualizar")
        
                # 3) Al enviar, actualizamos DB y refrescamos caches
                if enviar2:
                    cn3, cur3 = conectar_db(), None
                    try:
                        cur3 = cn3.cursor()
                        cur3.execute("""
                            UPDATE presupuesto_registros
                            SET item=%s,
                                categoria=%s,
                                grupo=%s,
                                centro_gestor=%s,
                                unidad_codigo=%s,
                                concepto_gasto=%s,
                                descripcion=%s,
                                cantidad=%s,
                                valor_unitario=%s,
                                fecha_inicio=%s,
                                accion='Editar'
                            WHERE id=%s
                        """, (
                            item2, categoria2, grupo2, centro2, unidad2,
                            conc2, desc2, cantidad2, valor_unit2, fecha_i2,
                            id_ed
                        ))
                        cn3.commit()
                        st.success("✅ Registro actualizado correctamente.")
                    except Exception as e:
                        st.error(f"❌ Error al actualizar: {e}")
                    finally:
                        if cur3: cur3.close()
                        cn3.close()
        
                    # 4) Actualizo el registro en session_state
                    st.session_state[data_key] = {
                        **data,
                        "item":           item2,
                        "categoria":      categoria2,
                        "grupo":          grupo2,
                        "centro_gestor":  centro2,
                        "unidad_codigo":  unidad2,
                        "concepto_gasto": conc2,
                        "descripcion":    desc2,
                        "cantidad":       cantidad2,
                        "valor_unitario": valor_unit2,
                        "fecha_inicio":   fecha_i2.strftime("%Y-%m-%d")
                    }
        
                    # 5) Limpio caché de ingresos/gastos y de registros de usuario
                    for fn in (load_ingresos, load_gastos, load_user_records):
                        try:
                            fn.clear()
                        except Exception:
                            pass
        
                    # 6) Forzar recarga completa (sidebar, Ver Todo y Descargar)
                    st.rerun() 
        

            # — ELIMINAR —
            elif accion == "Eliminar":
                st.header("🗑️ Eliminar Registro")
                id_del = st.text_input("ID a eliminar", key="id_del")
                if st.button("Eliminar", key="btn_eliminar"):
                    try:
                        # 1) Borrar del presupuesto
                        cn = conectar_db()
                        cur = cn.cursor()
                        cur.execute("DELETE FROM presupuesto_registros WHERE id=%s", (id_del,))
                        cn.commit()
                        cur.close()
                        cn.close()

                        st.success("✅ Registro eliminado correctamente.")

                        # 2) Limpiar caché para recargar ingresos/gastos
                        try:
                            load_ingresos.clear()
                            load_gastos.clear()
                        except Exception:
                            pass

                        # 3) Forzar recarga completa (actualiza sidebar)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error al eliminar: {e}")

        # — VER TODO —
    elif accion == "Ver Todo":
            if st.session_state["usuario"] == "admin":
                st.header("📋 Todos los Registros")
                cn = conectar_db()
                df = pd.read_sql("SELECT * FROM presupuesto_registros ORDER BY id DESC", cn)
                cn.close()
            else:
                st.header("📋 Mis Registros")
                df = load_user_records(st.session_state["usuario"])
            
            if df.empty:
                st.info("No tienes registros aún.")
            else:
                st.dataframe(df, use_container_width=True)        
        
    if accion == "Descargar":
            st.header("🔽 Descargar todos los registros de usuarios")

            # 1) Traer los datos
            cn = conectar_db()
            df = pd.read_sql("SELECT * FROM registros_usuarios", cn)
            cn.close()

            # 2) Renombrar columnas para que queden con tus títulos
            df = df.rename(columns={
                "item":                  "Item (1)",
                "grupo":                 "Grupo (2)",
                "categoria":             "Categoría (3)",
                "agrupador":             "Agrupador (4)",
                "no_vig_futura":         "No. Vig Futura (5)",
                "posicion_vf":           "Posicion VF (6)",
                "clasif_activos":        "Clasif_Activos (7)",
                "regla_activos":         "Regla Activos (8)",
                "concepto_gasto":        "Concepto de Gasto",
                "imputacion":            "Imputacion (9)",
                "detalle_objeto_gasto":  "Detalle Objeto de Gasto (10)",
                "ger_neg":               "Ger-neg (11)",
                "centro_gestor":         "Centro Gestor (12)",
                "descr_centro_gestor":   "Descr_Centro Gestor (13)",
                "pec":                   "PEC (14)",
                "desc_pec":              "Desc_PEC (15)",
                "id_recurso":            "ID Recurso (16)",
                "nombre_id_recurso":     "Nombre_ID Recurso (17)",
                "proyecto":              "Proyecto (18)",
                "descr_proyecto":        "Descr_Proyecto (19)",
                "pospre":                "Pospre (20)",
                "desc_pospre":           "Desc_Pospre (21)",
                "cantidad":              "Cantidad (22)",
                "unidad_medida":         "Unidad Medida (23)",
                "valor_unitario":        "Valor Unitario (24)",
                "total_proyectado":      "Total Proyectado (25)",
                # agrega aquí más mapeos si faltara alguna columna
            })

            # 3) Mostrar y descargar
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name="registros_usuarios.csv",
                mime="text/csv"
            )    

# — Tab Dashboard —
with tab2:
    st.markdown("<h3>📈 Visualización Dashboard</h3>", unsafe_allow_html=True)
    # Introducción breve
    st.markdown(
        """
        Bienvenido al **Dashboard Presupuestal UENE 2026**.  
        Aquí verás tus KPIs en miles y, además, un análisis de tus gastos 
        agrupados por concepto para tomar decisiones más informadas.
        """
    )

    # 1) Datos de flujo por unidad (ingresos, gastos, saldo)
    ingresos_df = load_ingresos().rename(columns={'centro':'Unidad','ingreso':'Ingresos'})
    gastos_df   = load_gastos().rename(columns={'centro':'Unidad','gastos':'Gastos'})
    df_flow     = ingresos_df.merge(gastos_df, on='Unidad', how='left').fillna(0)
    df_flow['Saldo'] = df_flow['Ingresos'] - df_flow['Gastos']

    # Filtra únicamente tus unidades
    mis_unidades = obtener_unidades(st.session_state["usuario"])
    df_mia       = df_flow[df_flow['Unidad'].isin(mis_unidades)]
    if df_mia.empty:
        st.info("Aún no tienes registros de ingresos/gastos para tus unidades.")
        st.stop()

    # 2) KPIs en miles (k)
    total_ing = df_mia['Ingresos'].sum()
    total_gas = df_mia['Gastos'].sum()
    total_sal = df_mia['Saldo'].sum()

    st.markdown("---")
    st.subheader("🔑 Tus Indicadores Clave (en miles)")
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Ingresos Asignados",
        f"{total_ing/1_000:,.0f}",
        help="Suma de ingresos asignados (en miles)"
    )
    c2.metric(
        "Gastos Registrados",
        f"{total_gas/1_000:,.0f}",
        help="Suma de gastos registrados (en miles)"
    )
    c3.metric(
        "Saldo Disponible",
        f"{total_sal/1_000:,.0f}",
        f"{'👍 Positivo' if total_sal>=0 else '⚠️ Negativo'}",
        help="Balance entre ingresos y gastos (en miles)"
    )

    if total_sal < 0:
        st.warning("⚠️ Tu saldo es negativo. Revisa tus gastos por concepto.")
    else:
        st.success("✅ Tu saldo es positivo. ¡Buen control del presupuesto!")

    st.markdown("---")

    # 3) Gráfica de barras de Ingresos vs Gastos
    st.subheader("📊 Ingresos vs Gastos por Unidad")
 
    import numpy as np

    fig1, ax1 = plt.subplots()
    x     = np.arange(len(df_mia))
    ancho = 0.35

    ax1.bar(x - ancho/2, df_mia['Ingresos'] / 1_000, ancho, label="Ingresos")
    ax1.bar(x + ancho/2, df_mia['Gastos']   / 1_000, ancho, label="Gastos")

    ax1.set_xticks(x)
    ax1.set_xticklabels(df_mia['Unidad'], rotation=0, ha='center')
    ax1.set_ylabel("Miles de pesos")
    ax1.legend()

    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("---")

    
    # Gráfico 2: Por Concepto de Gasto — barras **horizontales**
    #st.subheader("📊 Por Concepto de Gasto")
    df_regs = load_user_records(st.session_state["usuario"])
    df_regs["Gasto"] = df_regs["cantidad"] * df_regs["valor_unitario"]
    df_concept = (
        df_regs.groupby("concepto_gasto")["Gasto"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig2, ax2 = plt.subplots()
    ax2.barh(
        df_concept["concepto_gasto"],
        df_concept["Gasto"] / 1_000
    )
    ax2.set_xlabel("Gasto (miles de pesos)")
    ax2.set_ylabel("Concepto de Gasto")
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown(
        """
        Aquí puedes ver en qué conceptos se está concentrando tu gasto. 
        Ordenados de mayor a menor, estos te ayudan a identificar partidas clave.
        """
    )
