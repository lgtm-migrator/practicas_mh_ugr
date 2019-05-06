Búsqueda Local
==============

.. automodule:: algorithms.local_search

Para el algoritmo de búsqueda local se han implementado dos
elementos principales. El primero de ellos es el algoritmo en sí,
que es el siguiente:

.. autofunction:: algorithms.local_search.local_search

Y el segundo es una clase que envuelve el algoritmo para darle
una sintaxis similar a los objetos de Sklearn. Esto permite la integración
automática con el resto del código. LA clase en la siguiente:

.. autoclass:: algorithms.local_search.LocalSearch
    :members:
