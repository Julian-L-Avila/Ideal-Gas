"""
Pruebas unitarias para la simulación de gas ideal.

Verifica la lógica de la clase Particula y la física de colisiones
(conservación de energía y momento) en la simulación.
"""

import unittest
import numpy as np
from gas_ideal.particula import Particula
from gas_ideal.simulacion import (
    manejar_colisiones_pared, 
    manejar_colisiones_particulas,
    calcular_energia_total
)

# Usamos np.testing.assert_allclose para comparar arrays de punto flotante
from numpy.testing import assert_allclose, assert_almost_equal


class TestParticula(unittest.TestCase):
    """Pruebas para la clase Particula."""

    def test_movimiento(self):
        """Verifica que la partícula se mueve correctamente."""
        p = Particula(pos=[0.0, 0.0], vel=[1.0, 2.0], masa=1.0, radio=0.1)
        p.mover(dt=0.5)
        assert_allclose(p.pos, [0.5, 1.0])

    def test_energia_cinetica(self):
        """Verifica el cálculo de la energía cinética."""
        # E_k = 0.5 * m * (vx^2 + vy^2) = 0.5 * 2 * (3^2 + 4^2) = 1 * 25 = 25
        p = Particula(pos=[0, 0], vel=[3.0, 4.0], masa=2.0, radio=0.1)
        self.assertAlmostEqual(p.energia_cinetica(), 25.0)


class TestSimulacion(unittest.TestCase):
    """Pruebas para la lógica de simulación y física."""

    def setUp(self):
        """Configuración común para las pruebas de simulación."""
        self.L = 10.0  # Caja de 10x10

    def test_colision_pared_horizontal(self):
        """Verifica la colisión elástica con una pared horizontal (superior)."""
        p = Particula(pos=[5.0, 9.95], vel=[0.0, 1.0], masa=1.0, radio=0.1)
        # La partícula está a 0.05 de la pared (radio=0.1, pos.y=9.95)
        # Se moverá a y=10.05 en dt=0.1, lo que activa la colisión
        p.mover(dt=0.1)
        manejar_colisiones_pared(p, self.L)
        
        # La posición debe corregirse a L - radio
        self.assertAlmostEqual(p.pos[1], 9.9) 
        # La velocidad en y debe invertirse
        self.assertAlmostEqual(p.vel[1], -1.0) 

    def test_colision_pared_vertical(self):
        """Verifica la colisión elástica con una pared vertical (izquierda)."""
        p = Particula(pos=[0.05, 5.0], vel=[-2.0, 0.0], masa=1.0, radio=0.1)
        p.mover(dt=0.1) # Se moverá a x = -0.15
        manejar_colisiones_pared(p, self.L)
        
        self.assertAlmostEqual(p.pos[0], 0.1) # Corregida a radio
        self.assertAlmostEqual(p.vel[0], 2.0) # Velocidad en x invertida

    def test_colision_particulas_conservacion_energia_momento(self):
        """
        Verifica la conservación de energía y momento en una colisión
        simple de dos partículas.
        """
        # Colisión frontal
        p1 = Particula(pos=[4.0, 5.0], vel=[1.0, 0.0], masa=1.0, radio=0.1)
        p2 = Particula(pos=[4.1, 5.0], vel=[-1.0, 0.0], masa=1.0, radio=0.1)
        particulas = [p1, p2]
        
        # Calcular estados iniciales
        E_inicial = calcular_energia_total(particulas)
        P_inicial = p1.masa * p1.vel + p2.masa * p2.vel

        # Están superpuestas, la función debe detectar la colisión
        manejar_colisiones_particulas(particulas)
        
        # Calcular estados finales
        E_final = calcular_energia_total(particulas)
        P_final = p1.masa * p1.vel + p2.masa * p2.vel
        
        # En una colisión frontal 1D con masas iguales, intercambian velocidades
        assert_allclose(p1.vel, [-1.0, 0.0], atol=1e-7)
        assert_allclose(p2.vel, [1.0, 0.0], atol=1e-7)
        
        # Verificar conservación
        assert_almost_equal(E_inicial, E_final, decimal=7)
        assert_allclose(P_inicial, P_final, atol=1e-7)

    def test_colision_particulas_caso_general(self):
        """Prueba una colisión 2D más general."""
        p1 = Particula(pos=[1.0, 1.0], vel=[1.0, 1.0], masa=2.0, radio=0.1)
        p2 = Particula(pos=[1.1, 1.1], vel=[-1.0, -1.0], masa=2.0, radio=0.1)
        particulas = [p1, p2]
        
        E_inicial = calcular_energia_total(particulas)
        P_inicial = p1.masa * p1.vel + p2.masa * p2.vel
        
        manejar_colisiones_particulas(particulas)
        
        E_final = calcular_energia_total(particulas)
        P_final = p1.masa * p1.vel + p2.masa * p2.vel
        
        # Verificar conservación
        assert_almost_equal(E_inicial, E_final, decimal=7)
        assert_allclose(P_inicial, P_final, atol=1e-7)

if __name__ == '__main__':
    unittest.main(verbosity=2)
