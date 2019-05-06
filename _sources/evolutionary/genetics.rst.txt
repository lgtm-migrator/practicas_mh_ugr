Genéticos y Meméticos
=====================

.. automodule:: algorithms.evolutionary

Dentro del módulo de algoritmos evolutivos encontramos dos clases
principales. La primera de ellas es la siguiente:

.. autoclass:: algorithms.evolutionary.EvolutionaryAlgorithm
  :members:

Esta clase realmente implementa los **algoritmos
genéticos**. Pero es lo suficientemente abstracta como para poder
implementar los meméticos a partir de ella. Por ese motivo se llama
*EvolutionaryAlgorithm*.

La clase de los algoritmos genéticos es bastante corta ya que únicamente
tiene que sobreescribir la llamada al método **run** para añadir la
estrategia memética.

.. autoclass:: algorithms.evolutionary.MemeticAlgorithm
  :members:

Finalmente, existe dos funciones auxiliares:

.. autofunction:: algorithms.evolutionary.genetic_algorithm.create_toolbox

Esta función crea un objeto deap.Toolbox que almacena todos los operadores
que va a utilizar el algoritmo evolutivo. Esto permite desacoplar las estrategias
de los operadores. Y hace que crear un nuevo algoritmo sea tan sencillo como
crear un nuevo Toolbox.

Por último, como Deap implementa por defecto la optimización de funciones
multiobjetivo, es necesario envolver la función :func:`algorithms.core.evaluate`
en una tupla de un solo elemento. Indicando así que solamente hay un objetivo.

.. autofunction:: algorithms.evolutionary.genetic_algorithm.evaluate_individual

Realmente, por la naturaleza de nuestra función fitness, podríamos utilizar
estrategias multiobjetivo. Donde el un objetivo sea la precision y otro la
reducción. Pero para no modificar la función fitness de la que dependen otras
partes del código, se ha envuelto en dicha tupla.
