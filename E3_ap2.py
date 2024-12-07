import pandas as pd
import pulp as lp

# Carga de los datos desde los archivos proporcionados
df_operaciones = pd.read_excel("241204_datos_operaciones_programadas.xlsx", index_col=0)
df_costes = pd.read_excel("241204_costes.xlsx", index_col=0)

# Se limpian los nombres de columnas
df_operaciones.columns = df_operaciones.columns.str.strip()

# Servicios relevantes para el modelo
servicios_relevantes = [
    "Cardiología Pediátrica",
    "Cirugía Cardíaca Pediátrica",
    "Cirugía Cardiovascular",
    "Cirugía General y del Aparato Digestivo"
]

# Se filtran operaciones de los servicios relevantes para el modelo
operaciones_relevantes = df_operaciones[df_operaciones["Especialidad quirúrgica"].isin(servicios_relevantes)]

# Se crea una lista de operaciones y quirófanos
operaciones = operaciones_relevantes.index.tolist()
quirofanos = df_costes.index.tolist()

# Se calcula el coste medio por operación
costes_medios_operaciones = df_costes.mean(axis=0)

# Se crea una función para calcular incompatibilidades considerando todas las operaciones relevantes
def calcular_incompatibilidades(operaciones_df):
    incompatibilidades = {op: set() for op in operaciones_df.index.tolist()}
    n = 0  # Contador para evitar comparaciones repetidas

    for codigo_a, a in operaciones_df.iterrows():
        a_inicio = pd.to_datetime(a['Hora inicio'])
        a_fin = pd.to_datetime(a['Hora fin'])
        n = n + 1

        for num in range(n, len(operaciones_df)):
            b = operaciones_df.iloc[num]
            b_inicio = pd.to_datetime(b['Hora inicio'])
            b_fin = pd.to_datetime(b['Hora fin'])
            codigo_b = operaciones_df.index[num]

            # Comprobar solapamiento
            if a_inicio <= b_inicio < a_fin or b_inicio <= a_inicio < b_fin:
                incompatibilidades[codigo_a].add(codigo_b)
                incompatibilidades[codigo_b].add(codigo_a)
    return incompatibilidades

incompatibilidades = calcular_incompatibilidades(operaciones_relevantes)

# Se crea una función para generar planificaciones optimizadas
def generar_planificaciones_optimizadas(operaciones, incompatibilidades):
    planificaciones = []
    operaciones_pendientes = set(operaciones)

    while operaciones_pendientes:
        planificacion = set()
        for op in sorted(operaciones_pendientes):  # Ordenar para mantener consistencia
            if not any(op_incompatible in planificacion for op_incompatible in incompatibilidades[op]):
                planificacion.add(op)
        planificaciones.append(planificacion)
        operaciones_pendientes -= planificacion  # Elimina operaciones ya asignadas
    return planificaciones

planificaciones_optimizadas = generar_planificaciones_optimizadas(operaciones, incompatibilidades)

# Se crea una función para resolver el modelo
def resolver_modelo2(operaciones, planificaciones, costes_medios):
    planificaciones_indices = list(range(len(planificaciones)))

    # Calcular el coste de cada planificación basado en costes medios
    costes_planificaciones = {
        k: sum(costes_medios[op] for op in planificaciones[k])
        for k in planificaciones_indices
    }

    # Creación del modelo
    modelo2 = lp.LpProblem(name="Set_Covering_Quirófanos", sense=lp.LpMinimize)

    # Variables de decisión: y[k] indica si la planificación k es seleccionada
    y = lp.LpVariable.dicts("y", planificaciones_indices, cat=lp.LpBinary)

    # Función objetivo: minimizar el coste total de las planificaciones seleccionadas
    modelo2 += lp.lpSum(costes_planificaciones[k] * y[k] for k in planificaciones_indices)

    # Restricción: cada operación debe estar cubierta por al menos una planificación
    for op in operaciones:
        modelo2 += lp.lpSum(y[k] for k in planificaciones_indices if op in planificaciones[k]) >= 1

    # Resolver el modelo
    modelo2.solve()

    # Extracción de las planificaciones seleccionadas
    planificaciones_seleccionadas = [
        planificaciones[k] for k in planificaciones_indices if y[k].value() == 1
    ]

    coste_total = modelo2.objective.value()

    return planificaciones_seleccionadas, coste_total

# Resolver el modelo con las planificaciones optimizadas
planificaciones_seleccionadas_opt, coste_total_opt = resolver_modelo2(operaciones, planificaciones_optimizadas, costes_medios_operaciones)

# Caracterizar la solución
planificaciones_seleccionadas_count_opt = len(planificaciones_seleccionadas_opt)
operaciones_totales_cubiertas_opt = sum(len(planificacion) for planificacion in planificaciones_seleccionadas_opt)

# Mostrar los resultados optimizados
print("\nPlanificaciones seleccionadas (Optimizadas):")
for idx, planificacion in enumerate(planificaciones_seleccionadas_opt, start=1):
    print(f"Planificación {idx}: {planificacion}")

print(f"\nCoste total de la asignación: {coste_total_opt}")
print("\nCaracterización de la solución óptima:")
print(f"- Total de planificaciones seleccionadas: {planificaciones_seleccionadas_count_opt}")
print(f"- Total de operaciones cubiertas: {operaciones_totales_cubiertas_opt}")
print(f"- Coste total mínimo: {coste_total_opt}")
