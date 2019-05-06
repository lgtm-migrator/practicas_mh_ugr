Core
====

.. automodule:: algorithms.core

En este modulo se encuentran las clases y funciones comunes
para todos los algoritmos. Principalmente existen dos componentes.
Una clase de la que heredan todos los algoritmos y que sirve para unificar
la interfaz de uso:


Algoritmo base
--------------

.. autoclass:: algorithms.core.AlgorithmBase
    :members:


Función fitness
---------------

Y lo segundo es la función fitness, la cuál también es común para todos
los algoritmos que exploren soluciones: Búsqueda local, Genéticos, etc.

.. autofunction:: algorithms.core.evaluate
