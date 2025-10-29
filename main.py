"""
Archivo principal para ejecutar la simulación de gas ideal.

1. Importa los módulos necesarios.
2. Define los parámetros de la simulación.
3. Ejecuta el bucle de simulación.
4. Llama a las funciones de análisis y visualización.
"""

import numpy as np
import time
from gas_ideal import simulacion as sim
from gas_ideal import graficar as graf

# --- Parámetros de la Simulación ---
N_PARTICULAS = 200       # Número de partículas
L_CAJA = 10.0           # Tamaño de la caja (L x L)
MASA = 1.0              # Masa (unidades reducidas)
RADIO = 0.05             # Radio de la partícula
V_MAX_INICIAL = 2.0     # Velocidad inicial máxima

# Parámetros de tiempo
DT = 0.001              # Paso de tiempo (clave para la estabilidad)
PASOS_SIMULACION = 5000 # Número total de pasos
PASOS_HISTORIA = 10     # Guardar datos cada X pasos (para animación)
PASOS_EQUILIBRIO = 1000 # Pasos a ignorar para el cálculo de T media

def ejecutar():
    """Función principal de la simulación."""
    
    print("Iniciando simulación de gas ideal...")
    print(f"Parámetros: N={N_PARTICULAS}, L={L_CAJA}, dt={DT}, "
          f"Pasos={PASOS_SIMULACION}")
    
    # 1. Inicialización
    start_time = time.time()
    particulas = sim.inicializar_particulas(
        N=N_PARTICULAS, L=L_CAJA, v_max=V_MAX_INICIAL, m=MASA, r=RADIO
    )
    print(f"Inicialización completada en {time.time() - start_time:.2f} s")

    # Listas para guardar los resultados
    energia_historia = []
    posicion_historia = [] # Para la animación
    temperatura_historia = []

    # 2. Bucle de Simulación
    start_time = time.time()
    for i in range(PASOS_SIMULACION):
        sim.paso_simulacion(particulas, L_CAJA, DT)
        
        # Guardar datos
        energia_total = sim.calcular_energia_total(particulas)
        energia_historia.append(energia_total)
        
        temp_actual = sim.calcular_temperatura_cinetica(particulas)
        temperatura_historia.append(temp_actual)

        if i % PASOS_HISTORIA == 0:
            # Guardar posiciones (copia profunda)
            pos_actuales = np.array([p.pos for p in particulas])
            posicion_historia.append(pos_actuales)
            
        if (i+1) % 1000 == 0:
            print(f"Paso {i+1}/{PASOS_SIMULACION} completado...")
            
    print(f"Bucle de simulación completado en {time.time() - start_time:.2f} s")

    # 3. Análisis y Visualización
    print("\n--- Análisis de Resultados ---")
    
    # 3.1. Verificar Conservación de Energía
    # (Se espera una línea casi plana)
    graf.graficar_conservacion_energia(energia_historia, DT)
    
    # 3.2. Relación Velocidad-Temperatura
    # Calcular la T promedio después de que el sistema se estabilice
    T_media_equilibrio = np.mean(temperatura_historia[PASOS_EQUILIBRIO:])
    
    print(f"\nTemperatura cinética promedio (post-equilibrio): "
          f"{T_media_equilibrio:.3f}")
          
    # Verificar la distribución de velocidades (rapidez)
    # (Se espera que coincida con Maxwell-Boltzmann 2D)
    graf.graficar_distribucion_velocidades(particulas, T_media_equilibrio)

    # 3.3. Animación de Trayectorias
    print("\nIniciando animación (cierre la ventana para continuar)...")
    graf.animar_simulacion(posicion_historia, L_CAJA, RADIO)
    
    print("\nSimulación finalizada.")

if __name__ == "__main__":
    ejecutar()
