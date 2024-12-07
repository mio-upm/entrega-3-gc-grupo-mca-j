import pandas as pd
import pulp as lp

# Carga de los datos desde los archivos proporcionados
df_operaciones = pd.read_excel("241204_datos_operaciones_programadas.xlsx", index_col=0)
df_costes = pd.read_excel("241204_costes.xlsx", index_col=0)

# Se limpian los nombres de columnas
df_operaciones.columns = df_operaciones.columns.str.strip()

# Se crea lista de todas las operaciones y quirófanos
operaciones = df_operaciones.index.tolist()
quirofanos = df_costes.index.tolist()

# Función para calcular incompatibilidades
def calcular_incompatibilidades(operaciones_df):
    incompatibilidades = {op: set() for op in operaciones_df.index.tolist()}
    n = 0
    for codigo_a, a in operaciones_df.iterrows():
        a_inicio = pd.to_datetime(a['Hora inicio'])
        a_fin = pd.to_datetime(a['Hora fin'])
        n = n + 1
        for num in range(n, len(operaciones_df)):
            b = operaciones_df.iloc[num]
            b_inicio = pd.to_datetime(b['Hora inicio'])
            b_fin = pd.to_datetime(b['Hora fin'])
            codigo_b = operaciones_df.index[num]
            if a_inicio <= b_inicio < a_fin or b_inicio <= a_inicio < b_fin:
                incompatibilidades[codigo_a].add(codigo_b)
                incompatibilidades[codigo_b].add(codigo_a)
    return incompatibilidades

incompatibilidades = calcular_incompatibilidades(df_operaciones)

# Función para generar planificaciones iniciales
def generar_planificaciones_optimizadas(operaciones, incompatibilidades):
    planificaciones = []
    operaciones_pendientes = set(operaciones)
    while operaciones_pendientes:
        planificacion = set()
        for op in sorted(operaciones_pendientes):
            if not any(op_incompatible in planificacion for op_incompatible in incompatibilidades[op]):
                planificacion.add(op)
        planificaciones.append(planificacion)
        operaciones_pendientes -= planificacion
    return planificaciones

planificaciones_iniciales = generar_planificaciones_optimizadas(operaciones, incompatibilidades)

# Función para resolver el modelo
def resolver_modelo3(operaciones, planificaciones):
    planificaciones_indices = list(range(len(planificaciones)))

    # Crea el modelo
    modelo3 = lp.LpProblem(name="Minimizar_Quirófanos", sense=lp.LpMinimize)

    # Variables de decisión: y[k] indica si la planificación k es seleccionada
    y = lp.LpVariable.dicts("y", planificaciones_indices, cat=lp.LpBinary)

    # Función objetivo: minimizar el número de quirófanos seleccionados
    modelo3 += lp.lpSum(y[k] for k in planificaciones_indices)

    # Restricción: cada operación debe estar cubierta por al menos una planificación
    for op in operaciones:
        modelo3 += lp.lpSum(y[k] for k in planificaciones_indices if op in planificaciones[k]) >= 1

    # Resuelve el modelo
    modelo3.solve()

    # Extrae las planificaciones seleccionadas
    planificaciones_seleccionadas = [
        planificaciones[k] for k in planificaciones_indices if y[k].value() == 1
    ]

    return planificaciones_seleccionadas, len(planificaciones_seleccionadas)

# Se resuelve el modelo con planificaciones iniciales
planificaciones_seleccionadas, num_quirofanos_utilizados = resolver_modelo3(operaciones, planificaciones_iniciales)

# Se caracteriza la solución
print("\nPlanificaciones seleccionadas (Modelo 3):")
for idx, planificacion in enumerate(planificaciones_seleccionadas, start=1):
    print(f"Planificación {idx}: {planificacion}")

print(f"\nNúmero mínimo de quirófanos necesarios: {num_quirofanos_utilizados}")
