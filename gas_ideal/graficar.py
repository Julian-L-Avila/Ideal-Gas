"""
Funciones de visualización para la simulación de gas ideal.

Utiliza matplotlib para:
1. Graficar la conservación de la energía a lo largo del tiempo.
2. Graficar la distribución de velocidades (rapidez) y compararla
   con la distribución teórica de Maxwell-Boltzmann 2D.
3. (Opcional) Animar las trayectorias de las partículas.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
from gas_ideal.particula import Particula

# Asumimos las mismas constantes que en simulacion.py
K_BOLTZMANN = 1.0


def graficar_conservacion_energia(energia_hist: List[float], dt: float):
    """
    Grafica la energía cinética total del sistema vs. tiempo.
    
    Args:
        energia_hist: Lista de valores de energía total en cada paso.
        dt: El paso de tiempo usado en la simulación.
    """
    pasos = len(energia_hist)
    tiempo = np.arange(pasos) * dt
    
    # Calcular la fluctuación relativa (desviación de la media)
    energia_media = np.mean(energia_hist)
    fluctuacion_rel = (energia_hist - energia_media) / energia_media
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tiempo, fluctuacion_rel)
    ax.set_title("Conservación de Energía (Fluctuación Relativa)")
    ax.set_xlabel("Tiempo (unidades reducidas)")
    ax.set_ylabel(r"$(E(t) - \langle E \rangle) / \langle E \rangle$")
    ax.grid(True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    print(f"Energía media: {energia_media:.5e}")
    print(f"Desviación estándar de energía (relativa): "
          f"{np.std(fluctuacion_rel):.5e}")
    
    plt.tight_layout()
    plt.show()


def maxwell_boltzmann_2d(v: np.ndarray, m: float, T: float) -> np.ndarray:
    """
    Distribución de rapidez de Maxwell-Boltzmann 2D (teórica).
    
    P(v) = (m*v / (k_B*T)) * exp(-m*v^2 / (2*k_B*T))
    
    Args:
        v: Array de valores de rapidez (magnitud de velocidad).
        m: Masa de la partícula.
        T: Temperatura del sistema.
        
    Returns:
        La densidad de probabilidad P(v) para cada v.
    """
    return (m * v / (K_BOLTZMANN * T)) * \
           np.exp(-m * v**2 / (2 * K_BOLTZMANN * T))


def graficar_distribucion_velocidades(particulas: List[Particula], T_media: float):
    """
    Grafica la distribución de velocidades (rapidez) del sistema
    y la compara con la predicción teórica de Maxwell-Boltzmann.
    
    Args:
        particulas: Lista de partículas (en su estado final).
        T_media: La temperatura promedio (calculada) del sistema.
    """
    rapideces = np.array([np.linalg.norm(p.vel) for p in particulas])
    masa = particulas[0].masa if particulas else 1.0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Graficar el histograma de los datos (normalizado)
    ax.hist(rapideces, bins=30, density=True, 
            label="Datos de Simulación", alpha=0.7)
    
    # Graficar la curva teórica
    v_teorico = np.linspace(0, np.max(rapideces), 200)
    P_v = maxwell_boltzmann_2d(v_teorico, m=masa, T=T_media)
    
    ax.plot(v_teorico, P_v, 'r-', lw=2, 
            label=f"Maxwell-Boltzmann 2D (T={T_media:.2f})")
    
    ax.set_xlabel("Rapidez (v)")
    ax.set_ylabel("Densidad de Probabilidad P(v)")
    ax.set_title("Distribución de Velocidades del Sistema")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()


def animar_simulacion(pos_historia: List[np.ndarray], L: float, r: float):
    """
    Crea una animación de las trayectorias de las partículas.
    
    Args:
        pos_historia: Lista donde cada elemento es un array (N, 2)
                      de posiciones en un paso de tiempo.
        L: Tamaño de la caja.
        r: Radio de la partícula (para escalar el tamaño del punto).
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_aspect('equal')
    ax.set_title("Simulación de Gas Ideal 2D")
    
    # Factor de escala para que los puntos se vean bien
    # (El tamaño en 'scatter' está en puntos^2, no en unidades de datos)
    fig_width_pixels = fig.get_window_extent().width
    scale_factor = (2 * r / L * fig_width_pixels)
    
    # Inicializar el scatter plot (estará vacío)
    scat = ax.scatter([], [], s=scale_factor**2)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    def update(frame):
        # 'frame' es el índice del paso de tiempo
        posiciones = pos_historia[frame]
        scat.set_offsets(posiciones)
        ax.set_title(f"Simulación de Gas Ideal 2D (Paso {frame})")
        return scat,

    # Crear la animación
    # 'frames' es el número total de pasos de tiempo
    # 'interval' es el tiempo en ms entre frames (ej. 20ms -> 50 fps)
    # 'blit=True' optimiza el redibujado
    ani = FuncAnimation(fig, update, frames=len(pos_historia),
                        init_func=init, blit=True, interval=20, repeat=False)
    
    plt.show()
