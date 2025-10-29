"""
Lógica principal de la simulación y funciones de análisis.

Contiene las funciones para inicializar el sistema, manejar las colisiones
(con paredes y entre partículas), ejecutar el bucle de simulación y
calcular las propiedades macroscópicas (energía, temperatura).
"""

import numpy as np
from typing import List
from gas_ideal.particula import Particula

# Usaremos unidades reducidas donde la constante de Boltzmann k_B = 1.
# Esto es común en física computacional.
K_BOLTZMANN = 1.0 


def inicializar_particulas(N: int, L: float, v_max: float, 
                           m: float, r: float) -> List[Particula]:
    """
    Crea una lista de N partículas en posiciones aleatorias sin superposición.

    Args:
        N: Número de partículas.
        L: Longitud del lado de la caja cuadrada (de 0 a L).
        v_max: Velocidad máxima inicial (componentes en [-v_max, v_max]).
        m: Masa de cada partícula.
        r: Radio de cada partícula.

    Returns:
        Una lista de N instancias de Particula.
    """
    particulas = []
    rng = np.random.default_rng()
    
    while len(particulas) < N:
        # Posición aleatoria dentro de los límites (considerando el radio)
        pos = rng.uniform(r, L - r, size=2)
        
        # Verificar superposición con partículas existentes
        superpuesta = False
        nueva_particula_temp = Particula(pos, np.zeros(2), m, r)
        for p_existente in particulas:
            if p_existente.superposicion(nueva_particula_temp):
                superpuesta = True
                break
        
        if not superpuesta:
            # Velocidad aleatoria
            vel = rng.uniform(-v_max, v_max, size=2)
            particulas.append(Particula(pos, vel, m, r))
            
    return particulas


def manejar_colisiones_pared(p: Particula, L: float) -> None:
    """
    Maneja las colisiones elásticas de una partícula con las paredes de la caja.

    Args:
        p: La partícula a verificar.
        L: Longitud del lado de la caja.
    """
    # Colisión con pared izquierda (x=0) o derecha (x=L)
    if p.pos[0] <= p.radio:
        p.pos[0] = p.radio  # Corregir posición para evitar que se pegue
        p.vel[0] = -p.vel[0]  # Invertir velocidad en x
    elif p.pos[0] >= L - p.radio:
        p.pos[0] = L - p.radio
        p.vel[0] = -p.vel[0]

    # Colisión con pared inferior (y=0) o superior (y=L)
    if p.pos[1] <= p.radio:
        p.pos[1] = p.radio
        p.vel[1] = -p.vel[1]  # Invertir velocidad en y
    elif p.pos[1] >= L - p.radio:
        p.pos[1] = L - p.radio
        p.vel[1] = -p.vel[1]


def manejar_colisiones_particulas(particulas: List[Particula]) -> None:
    """
    Maneja colisiones elásticas 2D entre todas las partículas.
    
    Utiliza un bucle O(N^2) para verificar todos los pares únicos.
    Aplica la conservación del momento y la energía para calcular
    las velocidades post-colisión.

    Args:
        particulas: Lista de todas las partículas en la simulación.
    """
    N = len(particulas)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = particulas[i]
            p2 = particulas[j]

            if p1.superposicion(p2):
                # Implementación de la colisión elástica 2D
                # Referencia: https://en.wikipedia.org/wiki/Elastic_collision
                
                # Vector normal (de p1 a p2)
                v_rel = p1.vel - p2.vel
                r_rel = p1.pos - p2.pos
                
                # Distancia (norma de r_rel)
                dist_sq = np.dot(r_rel, r_rel)
                
                # Evitar división por cero si están perfectamente superpuestas
                if dist_sq < 1e-12:
                    continue

                # Producto punto de v_rel y r_rel
                v_r_dot = np.dot(v_rel, r_rel)

                # Solo colisionan si se están moviendo una hacia la otra
                if v_r_dot < 0:
                    # Asumimos masas iguales (m1 = m2 = m)
                    # La fórmula se simplifica:
                    # v1' = v1 - (v_rel . r_rel) / |r_rel|^2 * r_rel
                    # v2' = v2 + (v_rel . r_rel) / |r_rel|^2 * r_rel
                    
                    impulso = (v_r_dot / dist_sq) * r_rel
                    
                    p1.vel = p1.vel - impulso
                    p2.vel = p2.vel + impulso

                    # (Opcional pero recomendado) Corregir superposición
                    # Moverlas ligeramente para que dejen de tocarse
                    overlap = (p1.radio + p2.radio) - np.sqrt(dist_sq)
                    correction = (overlap / 2 + 1e-6) * (r_rel / np.sqrt(dist_sq))
                    p1.pos = p1.pos + correction
                    p2.pos = p2.pos - correction


def paso_simulacion(particulas: List[Particula], L: float, dt: float) -> None:
    """
    Ejecuta un único paso de la simulación (mover + colisionar).

    Args:
        particulas: La lista de partículas.
        L: Tamaño de la caja.
        dt: Paso de tiempo.
    """
    for p in particulas:
        p.mover(dt)
        manejar_colisiones_pared(p, L)
    
    manejar_colisiones_particulas(particulas)


# --- Funciones de Análisis ---

def calcular_energia_total(particulas: List[Particula]) -> float:
    """
    Calcula la energía cinética total del sistema.
    
    Returns:
        La suma de las energías cinéticas de todas las partículas.
    """
    return sum(p.energia_cinetica() for p in particulas)


def calcular_temperatura_cinetica(particulas: List[Particula]) -> float:
    """
    Calcula la temperatura "cinética" del sistema.

    Basado en el Teorema de Equipartición para un gas ideal 2D:
    <E_k_particula> = (f/2) * k_B * T
    Aquí, grados de libertad f=2 (traslación en x, y).
    <E_k_particula> = k_B * T
    
    La energía cinética promedio es E_total / N.
    Por lo tanto, T = (E_total / N) / k_B

    Returns:
        La temperatura escalar del sistema (asumiendo k_B=1).
    """
    if not particulas:
        return 0.0
        
    energia_promedio = calcular_energia_total(particulas) / len(particulas)
    
    # T = <E_k> / k_B
    return energia_promedio / K_BOLTZMANN


def obtener_distribucion_velocidades(particulas: List[Particula]) -> np.ndarray:
    """
    Recopila las magnitudes de velocidad (rapidez) de todas las partículas.

    Returns:
        Un array de NumPy con la rapidez de cada partícula.
    """
    return np.array([np.linalg.norm(p.vel) for p in particulas])
