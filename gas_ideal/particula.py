"""
Define la clase Particula para la simulación del gas ideal.

Esta clase representa una única partícula en un espacio 2D, caracterizada
por su posición, velocidad, masa y radio.
"""

import numpy as np
from typing import Type

# Definimos un tipo para vectores 2D para mejorar la legibilidad
Vector2D = np.ndarray


class Particula:
    """
    Representa una partícula puntual con propiedades físicas en 2D.

    Atributos:
        pos (Vector2D): Arreglo de NumPy [x, y] que representa la posición.
        vel (Vector2D): Arreglo de NumPy [vx, vy] que representa la velocidad.
        masa (float): Masa de la partícula.
        radio (float): Radio de la partícula (usado para colisiones).
    """

    def __init__(self, 
                 pos: Vector2D, 
                 vel: Vector2D, 
                 masa: float, 
                 radio: float):
        """
        Inicializa una nueva instancia de Particula.

        Args:
            pos: Posición inicial como un np.ndarray de shape (2,).
            vel: Velocidad inicial como un np.ndarray de shape (2,).
            masa: Valor de la masa (escalar).
            radio: Valor del radio (escalar).
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.masa = float(masa)
        self.radio = float(radio)

    def __repr__(self) -> str:
        """Representación de string para debugging."""
        return (f"Particula(pos={self.pos}, vel={self.vel}, "
                f"m={self.masa}, r={self.radio})")

    def mover(self, dt: float) -> None:
        """
        Actualiza la posición de la partícula basándose en su velocidad.

        Aplica la dinámica balística simple: r(t + dt) = r(t) + v(t) * dt.

        Args:
            dt: El paso de tiempo (delta t) para la actualización.
        """
        self.pos += self.vel * dt

    def energia_cinetica(self) -> float:
        """
        Calcula la energía cinética de la partícula.

        La energía cinética se define como E_k = 1/2 * m * |v|^2.

        Returns:
            El valor escalar de la energía cinética.
        """
        # np.dot(v, v) es equivalente a |v|^2 o (vx^2 + vy^2)
        return 0.5 * self.masa * np.dot(self.vel, self.vel)

    def distancia(self, otra: Type['Particula']) -> float:
        """
        Calcula la distancia euclidiana a otra partícula.

        Args:
            otra: La otra instancia de Particula.

        Returns:
            La distancia escalar entre los centros de las dos partículas.
        """
        # np.linalg.norm calcula la magnitud de un vector
        return np.linalg.norm(self.pos - otra.pos)

    def superposicion(self, otra: Type['Particula']) -> bool:
        """
        Verifica si esta partícula se superpone con otra.

        Args:
            otra: La otra instancia de Particula.

        Returns:
            True si las partículas se superponen, False en caso contrario.
        """
        # La superposición ocurre si la distancia entre centros es
        # menor que la suma de sus radios.
        return self.distancia(otra) < (self.radio + otra.radio)
