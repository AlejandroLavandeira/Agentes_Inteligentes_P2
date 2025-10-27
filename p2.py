#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Motor de Búsqueda A* para la Práctica 2: Búsqueda de Agentes Inteligentes.

Este script implementa el algoritmo A* para resolver el problema de planificación
del robot Kiva en un almacén. Utiliza un modelo de estados complejo y una
heurística admisible (distancia Manhattan) para encontrar el plan de coste mínimo.
"""

import heapq
import time

# --- 1. CLASE NODO ---
# Representa un nodo en el árbol de búsqueda de A*

class Nodo:
    """
    Un nodo en el árbol de búsqueda A*.
    Contiene el estado, su nodo padre (para reconstruir el camino),
    la acción que llevó a este estado, y los costes g, h, y f.
    """
    def __init__(self, estado, padre, accion, g, h, f):
        self.estado = estado
        self.padre = padre
        self.accion = accion
        self.g = g  # Coste real desde el inicio hasta este nodo
        self.h = h  # Coste heurístico estimado desde este nodo hasta el final
        self.f = f  # Coste total estimado (f = g + h)

    def __lt__(self, otro):
        """
        Comparador "menor que" para la cola de prioridad.
        Ordena por f, y en caso de empate, por h (para desempates).
        """
        if self.f == otro.f:
            return self.h < otro.h
        return self.f < otro.f

    def __repr__(self):
        return f"Nodo(f={self.f}, g={self.g}, h={self.h}, estado={self.estado})"

# --- 2. CLASE DE BÚSQUEDA A* ---

class BusquedaKiva:
    """
    Implementa el motor de búsqueda A* para el problema del Kiva.
    """

    def __init__(self, obstaculos_fijos, pos_inicial_pallets, pos_inicial_robot, request):
        """
        Inicializa el problema de búsqueda.

        Args:
            obstaculos_fijos (set): Un set de tuplas (x, y) de paredes y obstáculos.
            pos_inicial_pallets (frozenset): Un frozenset de tuplas (id, (x, y), o),
                                            donde 'id' es la pos original (Px, Py).
            pos_inicial_robot (tuple): Tupla (x, y, o) de la pose inicial del robot.
            request (list): Lista de tuplas de tarea [((Px,Py), (Ex,Ey), O_final), ...].
        """
        self.obstaculos_fijos = obstaculos_fijos
        self.pos_inicial_robot = pos_inicial_robot
        
        # El estado inicial se construye a partir de los datos
        self.estado_inicial = (
            pos_inicial_robot,  # (robot_x, robot_y, robot_o)
            None,               # pallet_cargado_id
            pos_inicial_pallets,# frozenset( (id, (x,y), o), ... )
            tuple(request)      # tuple( ((Px,Py), (Ex,Ey), O), ... )
        )

        # Mapeo de (Px,Py) -> (Ex,Ey,O) para el test de meta
        self.pallets_meta_pos = { tarea[0]: (tarea[1], tarea[2]) for tarea in request }

        # --- Definiciones de Movimiento (0=N, 1=E, 2=S, 3=O) ---
        self.movimientos = {
            0: (0, 1),   # Norte (sube Y)
            1: (1, 0),   # Este  (sube X)
            2: (0, -1),  # Sur   (baja Y)
            3: (-1, 0)   # Oeste (baja X)
        }
        self.rotar_derecha = {0: 1, 1: 2, 2: 3, 3: 0}
        self.rotar_izquierda = {0: 3, 3: 2, 2: 1, 1: 0}

    # --- 2.1. Funciones de Ayuda ---

    def manhattan(self, pos1, pos2):
        """Calcula la distancia Manhattan (coste mínimo de movimiento)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_pallet_at(self, pos, pos_pallets):
        """
        Devuelve el pallet (id, (x,y), o) en una posición dada, o None.
        """
        for pallet in pos_pallets:
            if pallet[1] == pos:
                return pallet
        return None

    def es_valido(self, x, y, pos_pallets):
        """
        Comprueba si una celda (x, y) es transitable.
        """
        pos = (x, y)
        # 1. Comprobar obstáculos fijos (paredes)
        if pos in self.obstaculos_fijos:
            return False
        # 2. Comprobar pallets en el suelo
        if self.get_pallet_at(pos, pos_pallets):
            return False
        return True

    def zona_giro_libre(self, x, y, pos_pallets):
        """
        Comprueba si las 8 celdas adyacentes están libres para girar cargado.
        """
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if not self.es_valido(x + i, y + j, pos_pallets):
                    return False
        return True

    # --- 2.2. Funciones Clave de A* (Meta, Heurística, Sucesores) ---

    def es_meta(self, estado):
        """
        Comprueba si el estado actual es un estado final (objetivo).
        """
        robot_pose, pallet_cargado, pos_pallets, tareas_pendientes = estado

        # Condición 1: No debe haber tareas pendientes
        if tareas_pendientes:
            return False
        
        # Condición 2: El robot no debe estar cargando nada
        if pallet_cargado is not None:
            return False

        # Condición 3: El robot debe estar en su posición inicial
        if robot_pose != self.pos_inicial_robot:
            return False

        # Condición 4: Todos los pallets de la request deben estar en su destino
        for id_pallet_original, (meta_pos, meta_ori) in self.pallets_meta_pos.items():
            pallet_encontrado = self.get_pallet_at(meta_pos, pos_pallets)
            
            if not pallet_encontrado:
                return False # El pallet no está en la posición de entrega
            
            if pallet_encontrado[0] != id_pallet_original or pallet_encontrado[2] != meta_ori:
                # El pallet equivocado está aquí, o la orientación es incorrecta
                return False

        # Si todo se cumple, es un estado meta
        return True

    def calcular_heuristica(self, estado):
        """
        Calcula la heurística admisible h(n) para un estado dado.
        Suma los costes fijos (elevar/bajar) y las distancias Manhattan.
        """
        robot_pose, pallet_cargado, pos_pallets, tareas_pendientes = estado
        coste_h = 0
        pos_actual_robot = (robot_pose[0], robot_pose[1])
        pos_casa = (self.pos_inicial_robot[0], self.pos_inicial_robot[1])

        if not tareas_pendientes:
            # No hay tareas, solo coste de volver a casa
            return self.manhattan(pos_actual_robot, pos_casa)

        # --- 1. Coste de la TAREA ACTIVA (tareas_pendientes[0]) ---
        tarea_activa = tareas_pendientes[0]
        id_pallet_activo = tarea_activa[0]
        pos_pallet_activo = tarea_activa[0]
        pos_destino_activo = tarea_activa[1]
        
        pos_ultimo_destino = pos_destino_activo # Para calcular la vuelta a casa

        if pallet_cargado == id_pallet_activo:
            # 1a. Ya tiene el pallet: Coste(Robot -> Destino) + Bajar
            coste_h += self.manhattan(pos_actual_robot, pos_destino_activo) + 3
        else:
            # 1b. Va vacío: Coste(Robot -> Pallet) + Elevar + Coste(Pallet -> Destino) + Bajar
            coste_h += self.manhattan(pos_actual_robot, pos_pallet_activo) + 3
            coste_h += self.manhattan(pos_pallet_activo, pos_destino_activo) + 3

        # --- 2. Coste del RESTO de tareas (tareas_pendientes[1:]) ---
        for i in range(1, len(tareas_pendientes)):
            tarea = tareas_pendientes[i]
            pos_pallet = tarea[0]
            pos_destino = tarea[1]
            pos_ultimo_destino = pos_destino
            
            # Coste(Pallet -> Destino) + Elevar + Bajar
            coste_h += self.manhattan(pos_pallet, pos_destino) + 6

        # --- 3. Coste de REGRESO A CASA (desde el último destino) ---
        coste_h += self.manhattan(pos_ultimo_destino, pos_casa)

        return coste_h


    def get_sucesores(self, estado_actual):
        """
        Genera todos los estados sucesores válidos desde el estado actual.
        Devuelve una lista de tuplas: (accion, nuevo_estado, coste_accion)
        """
        sucesores = []
        robot_pose, pallet_cargado, pos_pallets, tareas_pendientes = estado_actual
        (rx, ry, rθ) = robot_pose

        coste_extra = 1 if pallet_cargado else 0 # Coste extra por ir cargado

        # --- Operador 1: 'mover_adelante' ---
        dx, dy = self.movimientos[rθ]
        (nx, ny) = (rx + dx, ry + dy)

        if self.es_valido(nx, ny, pos_pallets):
            nuevo_robot_pose = (nx, ny, rθ)
            nuevo_estado = (nuevo_robot_pose, pallet_cargado, pos_pallets, tareas_pendientes)
            coste_accion = 1 + coste_extra
            sucesores.append(('mover_adelante', nuevo_estado, coste_accion))

        # --- Operador 2 y 3: 'girar_derecha' y 'girar_izquierda' ---
        for accion, rotacion in [('girar_derecha', self.rotar_derecha), 
                                 ('girar_izquierda', self.rotar_izquierda)]:
            # Precondición de giro cargado
            if pallet_cargado and not self.zona_giro_libre(rx, ry, pos_pallets):
                continue
                
            nueva_ori = rotacion[rθ]
            nuevo_robot_pose = (rx, ry, nueva_ori)
            nuevo_estado = (nuevo_robot_pose, pallet_cargado, pos_pallets, tareas_pendientes)
            coste_accion = 2 + coste_extra
            sucesores.append((accion, nuevo_estado, coste_accion))

        # --- Operador 4: 'elevar_pallet' ---
        if pallet_cargado is None: # Precondición: no llevar nada
            pallet_debajo = self.get_pallet_at((rx, ry), pos_pallets)
            if pallet_debajo: # Precondición: estar sobre un pallet
                id_pallet_elevado = pallet_debajo[0]
                
                nuevo_pos_pallets = pos_pallets - {pallet_debajo}
                nuevo_estado = (robot_pose, id_pallet_elevado, nuevo_pos_pallets, tareas_pendientes)
                coste_accion = 3
                sucesores.append((f'elevar {id_pallet_elevado}', nuevo_estado, coste_accion))

        # --- Operador 5: 'bajar_pallet' ---
        if pallet_cargado is not None: # Precondición: llevar un pallet
            # Precondición: el sitio está libre (ya lo comprueba es_valido)
            # Nota: es_valido comprueba si la celda está libre de *otros* pallets.
            # Como el que llevamos no está en pos_pallets, la celda (rx,ry)
            # siempre estará "libre" para esta comprobación.
            
            id_pallet_bajado = pallet_cargado
            pallet_a_bajar = (id_pallet_bajado, (rx, ry), rθ) # (id, pos, ori)
            
            nuevo_pos_pallets = pos_pallets | {pallet_a_bajar}
            
            # Comprobar si esta bajada completa una tarea
            nuevo_tareas_pendientes = tareas_pendientes
            if tareas_pendientes:
                tarea_activa = tareas_pendientes[0]
                id_tarea, pos_tarea, ori_tarea = tarea_activa[0], tarea_activa[1], tarea_activa[2]
                
                if (id_pallet_bajado == id_tarea and 
                    (rx, ry) == pos_tarea and 
                    rθ == ori_tarea):
                    # ¡Tarea completada! Eliminarla de la lista
                    nuevo_tareas_pendientes = tareas_pendientes[1:]

            nuevo_estado = (robot_pose, None, nuevo_pos_pallets, nuevo_tareas_pendientes)
            coste_accion = 3
            sucesores.append((f'bajar {id_pallet_bajado}', nuevo_estado, coste_accion))
        
        return sucesores

    # --- 2.3. El Motor A* ---
    
    def reconstruir_camino(self, nodo_final):
        """
        Recorre los punteros 'padre' para construir el plan final.
        """
        camino = []
        coste_total = nodo_final.g
        actual = nodo_final
        while actual.padre is not None:
            camino.append(actual.accion)
            actual = actual.padre
        camino.reverse()
        return camino, coste_total

    def resolver(self):
        """
        Ejecuta el bucle principal del algoritmo A*.
        """
        print("Iniciando búsqueda A*...")
        start_time = time.time()

        # Usamos un dict para rastrear el coste 'g' más bajo a cada estado
        # Esto combina las listas ABIERTA y CERRADA 
        g_costs = { self.estado_inicial: 0 }

        # Cola de prioridad (ABIERTA) [cite: 1]
        # Almacena (f, Nodo)
        h_inicial = self.calcular_heuristica(self.estado_inicial)
        f_inicial = h_inicial
        nodo_inicial = Nodo(self.estado_inicial, None, None, 0, h_inicial, f_inicial)
        
        abierta = []
        heapq.heappush(abierta, (nodo_inicial.f, nodo_inicial))
        
        nodos_expandidos = 0

        # Bucle principal de A* [cite: 2]
        while abierta:
            # Quitar el primer nodo (el mejor) de ABIERTA [cite: 3]
            f_actual, nodo_actual = heapq.heappop(abierta)

            # Optimización: Si encontramos un camino peor, lo ignoramos
            if nodo_actual.g > g_costs.get(nodo_actual.estado, float('inf')):
                continue

            # Comprobar si es un estado final [cite: 4]
            if self.es_meta(nodo_actual.estado):
                end_time = time.time()
                camino, coste = self.reconstruir_camino(nodo_actual)
                print("¡ÉXITO! Solución encontrada.")
                return {
                    "camino": camino,
                    "coste_total": coste,
                    "nodos_expandidos": nodos_expandidos,
                    "tiempo_total": end_time - start_time
                }
            
            # Expandir N y meterlo en CERRADA (implícito en g_costs) [cite: 5]
            nodos_expandidos += 1
            
            # Generar sucesores
            for accion, estado_sucesor, coste_accion in self.get_sucesores(nodo_actual.estado):
                nuevo_g = nodo_actual.g + coste_accion

                # Comprobar si este es un camino mejor al sucesor [cite: 6]
                if nuevo_g < g_costs.get(estado_sucesor, float('inf')):
                    g_costs[estado_sucesor] = nuevo_g
                    h = self.calcular_heuristica(estado_sucesor)
                    f = nuevo_g + h
                    
                    padre = nodo_actual
                    nuevo_nodo = Nodo(estado_sucesor, padre, accion, nuevo_g, h, f)
                    
                    # Insertar 's' en orden en ABIERTA [cite: 6]
                    heapq.heappush(abierta, (f, nuevo_nodo))

        # Si ABIERTA se vacía, no hay solución [cite: 2, 9]
        end_time = time.time()
        print("FRACASO. No se encontró solución.")
        return {
            "camino": None,
            "tiempo_total": end_time - start_time
        }


# --- 3. EJECUCIÓN DEL MUNDO DE PRUEBA (warehouse0.world) ---

if __name__ == "__main__":
    
    print("Configurando el problema 'warehouse0.world'...")

    # 1. OBSTÁCULOS FIJOS (Paredes y Obstáculos del .world)
    # Interpretación manual del fichero 'warehouse0.world'
    # Las poses (x, y) son los centros. Asumimos que bloquean esa celda.
    # Los objetos rotados (yaw=1.57) o largos (ej. -6.5) bloquean 2 celdas.
    OBSTACULOS = set([
        # Obstacles
        (-4, 2), (-1, 4), (-1, 3), (-4, 1), (-4, 0),
        # Walls (interpretando poses .5 como bloqueo de 2 celdas)
        (-7, 3), (-7, 4), (-7, 2), (-7, 1), (-7, 0),
        (-7, 0), (-6, 0), (-7, 4), (-6, 4),
        (-6, 0), (-5, 0), (-6, 4), (-5, 4),
        (-5, 4), (-5, 5), (-5, 5), (-5, 6),
        (-5, 0), (-5, -1),
        (-1, 5), (-1, 6),
        (-1, 5), (0, 5), (0, 5), (1, 5), (1, 5), (2, 5),
        (2, 4), (2, 5), (2, 3), (2, 4), (2, 2), (2, 3),
        (2, 1), (2, 2), (2, 0), (2, 1),
        (1, 0), (2, 0), (0, 0), (1, 0), (-1, 0), (0, 0),
        (-1, 0), (-1, -1),
        (-2, -1), (-1, -1), (-3, -1), (-2, -1), (-4, -1), (-3, -1), (-5, -1), (-4, -1),
        (-2, 6), (-1, 6), (-3, 6), (-2, 6), (-4, 6), (-3, 6), (-5, 6), (-4, 6)
    ])

    # 2. POSICIÓN INICIAL DEL ROBOT
    # <pose>-6 2 0 0 0 0</pose> (Yaw=0 es Este)
    POS_INICIAL_ROBOT = (-6, 2, 1) # (x, y, o) -> o=1 (Este)

    # 3. POSICIÓN INICIAL DE PALLETS
    # <pose>-3 4 0 0 0 0</pose> (Yaw=0 es Este)
    ID_PALLET_0 = (-3, 4) # Usamos su pos original como ID
    POS_INICIAL_PALLETS = frozenset([
        (ID_PALLET_0, (-3, 4), 1) # (id, (x,y), o) -> o=1 (Este)
    ])
    
    # 4. PETICIÓN (Nuestra 'request' inventada)
    # Mover pallet de (-3, 4) al destino (0, 5) con orientación Norte (0)
    REQUEST = [
        ( (-3, 4), (0, 5), 0 ) 
    ]

    # --- Ejecutar la Búsqueda ---
    problema = BusquedaKiva(OBSTACULOS, POS_INICIAL_PALLETS, POS_INICIAL_ROBOT, REQUEST)
    
    # Imprimir un resumen del estado inicial
    print(f"Estado Inicial: {problema.estado_inicial}")
    print(f"Heurística Inicial: {problema.calcular_heuristica(problema.estado_inicial)}")
    print("--------------------------------------------------")

    resultado = problema.resolver()
    
    print("--------------------------------------------------")
    
    if resultado["camino"]:
        print(f"Coste Total: {resultado['coste_total']}")
        print(f"Longitud del Plan: {len(resultado['camino'])}")
        print(f"Nodos Expandidos: {resultado['nodos_expandidos']}")
        print(f"Tiempo Total: {resultado['tiempo_total']:.4f} seg")
        
        print("\n--- PLAN ENCONTRADO ---")
        for i, accion in enumerate(resultado["camino"]):
            print(f"Paso {i+1}: {accion}")
    else:
        print("No se pudo encontrar un plan.")
        print(f"Tiempo Total: {resultado['tiempo_total']:.4f} seg")
