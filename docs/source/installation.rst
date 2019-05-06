Instalación
============

Requisitos
------------

Es necesario tener instalado Python 3 con el módulo venv que suele venir
instalado por defecto. También se recomienda tener instalado OpenMP y configurado
correctamente para g++/gcc si se quiere aprovechar mejor el paralelismo en la
aplicación


Instalación
------------

En el directorio FUENTES se encuentra un Makefile que permite instalar
todas las dependencias con el siguiente comando ::

  make install

Una vez instalado todas las dependencias debemos activar el entorno virtual ::

  source ./env/bin/activate


Uso
---

Para ejecutar todos los algoritmos a la vez debemos ejecutar lo siguiente: ::

  make run_all

Si su ordenador es potente, puede ejecutar varios algoritmos simultáneamente
usando la opción **--jobs** de GNU Make. ::

  make --jobs=4 run_all


Para más información sobre como ejecutar cada algoritmo individualmente,
utilizar la página de ayuda de la siguiente manera ::

  python3 practica2.py -h
