#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Motor de Búsqueda A* para la Práctica 2: Búsqueda de Agentes Inteligentes.
Versión 2:
- Corregido set de OBSTACULOS (solo incluye <obstacle>, no <wall>).
- Compatible con Python 3.5 (sin f-strings).
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
        # Compatible con Python 3.5
        return "Nodo(f={}, g={}, h={}, estado={})".format(self.f, self.g, self.h, self.estado)

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
            # Comparamos solo la posición (x,y)
            if pallet[1] == pos:
                return pallet
        return None

    def es_valido(self, x, y, pos_pallets):
        """
        Comprueba si una celda (x, y) es transitable.
        """
        pos = (x, y)
        
        # 1. Comprobar obstáculos fijos (los <obstacle> del .world)
        if pos in self.obstaculos_fijos:
            return False
            
        # 2. Comprobar pallets en el suelo (que no son parte de la request)
        if self.get_pallet_at(pos, pos_pallets):
            return False
            
        # 3. Comprobar límites del mapa (inferidos del .world)
        # Asumimos que el área jugable está contenida y no necesitamos
        # un chequeo explícito de límites si el mapa está "cerrado" por paredes.
        # Si la búsqueda se "escapa", añadiremos límites aquí.
            
        return True

    def zona_giro_libre(self, x, y, pos_pallets):
        """
        Comprueba si las 8 celdas adyacentes están libres para girar cargado.
        """
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                # NOTA: Usamos una versión simplificada de es_valido
                # que solo comprueba las 2 condiciones principales.
                pos = (x + i, y + j)
                if pos in self.obstaculos_fijos:
                    return False
                if self.get_pallet_at(pos, pos_pallets):
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
            pallet_encontrado = None
            for pallet in pos_pallets:
                if pallet[1] == meta_pos:
                    pallet_encontrado = pallet
                    break
            
            if not pallet_encontrado:
                return False # El pallet no está en la posición de entrega
            
            # Comparamos id, pos (ya hecho) y orientación
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
            # Buscamos la posición actual del pallet (por si se ha movido)
            pos_actual_pallet = None
            for p in pos_pallets:
                if p[0] == id_pallet_activo:
                    pos_actual_pallet = p[1]
                    break
            # Si el pallet no está en el suelo, es que lo llevamos (caso 1a)
            # o algo va mal, pero para la heurística asumimos que está donde debería
            if pos_actual_pallet is None:
                # Esto puede pasar si el pallet_cargado es OTRO pallet
                # En ese caso, el coste es:
                # Robot -> pos_temp -> Pallet_activo -> Destino_activo
                # Lo simplificamos (admisible) a:
                # Robot -> Pallet_activo -> Destino_activo
                pos_actual_pallet = pos_pallet_activo # Asumimos su pos original

            coste_h += self.manhattan(pos_actual_robot, pos_actual_pallet) + 3
            coste_h += self.manhattan(pos_actual_pallet, pos_destino_activo) + 3

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
                
                # Creamos un nuevo frozenset sin el pallet elevado
                nuevo_pos_pallets = pos_pallets - {pallet_debajo}
                nuevo_estado = (robot_pose, id_pallet_elevado, nuevo_pos_pallets, tareas_pendientes)
                coste_accion = 3
                sucesores.append(('elevar {}'.format(id_pallet_elevado), nuevo_estado, coste_accion))

        # --- Operador 5: 'bajar_pallet' ---
        if pallet_cargado is not None: # Precondición: llevar un pallet
            
            # (rx, ry) es la celda donde se bajará
            # Comprobamos que esté libre (debe estarlo, si no, 'mover' no nos habría traído aquí)
            
            id_pallet_bajado = pallet_cargado
            pallet_a_bajar = (id_pallet_bajado, (rx, ry), rθ) # (id, pos, ori)
            
            # Creamos un nuevo frozenset añadiendo el pallet bajado
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
            sucesores.append(('bajar {}'.format(id_pallet_bajado), nuevo_estado, coste_accion))
        
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
        # Esto combina las listas ABIERTA y CERRADA [cite: 1-9]
        g_costs = { self.estado_inicial: 0 }

        # Cola de prioridad (ABIERTA)
        # Almacena (f, Nodo)
        h_inicial = self.calcular_heuristica(self.estado_inicial)
        f_inicial = h_inicial
        nodo_inicial = Nodo(self.estado_inicial, None, None, 0, h_inicial, f_inicial)
        
        abierta = []
        heapq.heappush(abierta, (nodo_inicial.f, nodo_inicial))
        
        nodos_expandidos = 0

        # Bucle principal de A*
        while abierta:
            # Quitar el primer nodo (el mejor) de ABIERTA
            f_actual, nodo_actual = heapq.heappop(abierta)

            # Optimización: Si encontramos un camino peor, lo ignoramos
            if nodo_actual.g > g_costs.get(nodo_actual.estado, float('inf')):
                continue

            # Comprobar si es un estado final
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
            
            # Expandir N y meterlo en CERRADA (implícito en g_costs)
            nodos_expandidos += 1
            
            # Generar sucesores
            for accion, estado_sucesor, coste_accion in self.get_sucesores(nodo_actual.estado):
                nuevo_g = nodo_actual.g + coste_accion

                # Comprobar si este es un camino mejor al sucesor
                if nuevo_g < g_costs.get(estado_sucesor, float('inf')):
                    g_costs[estado_sucesor] = nuevo_g
                    h = self.calcular_heuristica(estado_sucesor)
                    f = nuevo_g + h
                    
                    padre = nodo_actual
                    nuevo_nodo = Nodo(estado_sucesor, padre, accion, nuevo_g, h, f)
                    
                    # Insertar 's' en orden en ABIERTA
                    heapq.heappush(abierta, (f, nuevo_nodo))

        # Si ABIERTA se vacía, no hay solución
        end_time = time.time()
        print("FRACASO. No se encontró solución.")
        return {
            "camino": None,
            "tiempo_total": end_time - start_time
        }


# --- 3. EJECUCIÓN DEL MUNDO DE PRUEBA (warehouse0.world) ---

if __name__ == "__main__":
    
    print("Configurando el problema 'warehouse0.world'...")

    # 1. OBSTÁCULOS FIJOS (Solo los <obstacle> del .world)
    #    Las <wall> definen el perímetro, que la búsqueda no alcanzará.
    #    <obstacle_0> <pose>-4 2 0 0 0 1.57</pose> (2m long, vertical, center -4, 2)
    #    <obstacle_1> <pose>-1 4 0 0 0 1.57</pose> (2m long, vertical, center -1, 4)
    #    <obstacle_2> <pose>-1 3 0 0 0 1.57</pose> (2m long, vertical, center -1, 3)
    #    <obstacle_3> <pose>-4 1 0 0 0 1.57</pose> (2m long, vertical, center -4, 1)
    #    <obstacle_4> <pose>-4 0 0 0 0 1.57</pose> (2m long, vertical, center -4, 0)
    #
    #    ¡CORRECCIÓN IMPORTANTE! Un obstáculo vertical de 2m centrado en (-4, 2)
    #    ocupa las celdas (-4, 1) y (-4, 2).
    
    OBSTACULOS = set([
        (-4, 1), (-4, 2), # obstacle_0
        (-1, 3), (-1, 4), # obstacle_1
        (-1, 2), (-1, 3), # obstacle_2
        (-4, 0), (-4, 1), # obstacle_3
        (-4, -1), (-4, 0) # obstacle_4
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
    print("Estado Inicial: {}".format(problema.estado_inicial))
    print("Heurística Inicial: {}".format(problema.calcular_heuristica(problema.estado_inicial)))
    print("--------------------------------------------------")

    resultado = problema.resolver()
    
    print("--------------------------------------------------")
    
    if resultado["camino"]:
        print("Coste Total: {}".format(resultado['coste_total']))
        print("Longitud del Plan: {}".format(len(resultado['camino'])))
        print("Nodos Expandidos: {}".format(resultado['nodos_expandidos']))
        print("Tiempo Total: {:.4f} seg".format(resultado['tiempo_total']))
        
        print("\n--- PLAN ENCONTRADO ---")
        for i, accion in enumerate(resultado["camino"]):
            print("Paso {}: {}".format(i+1, accion))
    else:
        print("No se pudo encontrar un plan.")
        print("Tiempo Total: {:.4f} seg".format(resultado['tiempo_total']))
