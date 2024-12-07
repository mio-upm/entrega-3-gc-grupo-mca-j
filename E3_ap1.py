# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:18:12 2024

@author: jovan
"""

import pandas as pd
import pulp as lp

# Carga de datos de los archivos proporcionados
df_costes = pd.read_excel("241204_costes.xlsx", index_col=0)
df_operaciones = pd.read_excel("241204_datos_operaciones_programadas.xlsx", index_col=0)

# Vista previa de los datos cargados para entender su estructura
df_operaciones.head(), df_costes.head()


# Se filtran las operaciones por Cardiología Pediátrica
equipos_cardiologia_pediatrica = df_operaciones[df_operaciones["Especialidad quirúrgica"] == "Cardiología Pediátrica"]
operaciones = equipos_cardiologia_pediatrica.index.tolist()
quirofanos = df_costes.index.tolist()

# Se crea diccionario de incompatibilidades utilizando conjuntos
L = {op: set() for op in operaciones}
n = 0  # Contador para evitar comparaciones repetidas

for codigo_a, a in equipos_cardiologia_pediatrica.iterrows():
    a_inicio = a['Hora inicio ']
    a_fin = a['Hora fin']
    n = n+1
    
    for num in range(n, len(equipos_cardiologia_pediatrica)):
        b = equipos_cardiologia_pediatrica.iloc[num]
        b_inicio = b['Hora inicio ']
        b_fin = b['Hora fin']
        codigo_b = equipos_cardiologia_pediatrica.index[num]

        # Se comprueba el solapamiento
        if a_inicio <= b_inicio < a_fin or b_inicio <= a_inicio < b_fin:
            L[codigo_a].add(codigo_b)
            L[codigo_b].add(codigo_a)

# Se crea una función para resolver el modelo
def resolver_modelo1(operaciones, quirofanos, df_costes, incompatibilidades):
    # Definir el modelo
    modelo1 = lp.LpProblem(name="Asignacion_Quirófanos", sense=lp.LpMinimize)
    
    # Variables de decisión
    x = lp.LpVariable.dicts("x", [(i, j) for i in operaciones for j in quirofanos], cat=lp.LpBinary)
    
    # Función objetivo: minimizar el coste total
    modelo1 += lp.lpSum(df_costes.loc[j, i] * x[i, j] for i in operaciones for j in quirofanos)
    
    # Restricción 1: Cada operación debe ser asignada a un quirófano
    for i in operaciones:
        modelo1 += lp.lpSum(x[i, j] for j in quirofanos) >= 1
    
    # Restricción 2: Operaciones incompatibles no pueden compartir quirófano
    for i in operaciones:
        for h in incompatibilidades[i]:
            for j in quirofanos:
                modelo1 += x[i, j] + x[h, j] <= 1
    
    # Solución del modelo
    modelo1.solve()
    
    # Se extraen las asignaciones óptimas y el coste total
    asignaciones = []
    for i in operaciones:
        for j in quirofanos:
            if x[i, j].value() == 1:
                asignaciones.append({"Operacion": i, "Quirofano": j})
    
    coste_total = modelo1.objective.value()
    
    return asignaciones, coste_total

# Se resuelve el modelo con los datos preparados
asignaciones, coste_total = resolver_modelo1(operaciones, quirofanos, df_costes, L)

# Se muestra la solución óptima
print("\nAsignaciones óptimas:")
print(pd.DataFrame(asignaciones))
print(f"\nCoste total de la asignación: {coste_total}")

# Se caracteriza la solución óptima
def caracterizar_solucion(asignaciones, coste_total):
    print("\nCaracterización de la solución óptima:")
    print(f"Total de operaciones asignadas: {len(asignaciones)}")
    print(f"Total de quirófanos utilizados: {len(set(a['Quirofano'] for a in asignaciones))}")
    print(f"Coste total mínimo: {coste_total}")

caracterizar_solucion(asignaciones, coste_total)