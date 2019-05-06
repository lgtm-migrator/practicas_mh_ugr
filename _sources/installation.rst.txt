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

Podemos ejecutar todos los algoritmos de manera individual con el script
principal correspondiente. O bien, ejecutar todos los algoritmos para
todos los conjuntos de datos con el siguiente comando ::

  make run_all


Para más información sobre como ejecutar cada algoritmo individualmente,
utilizar la página de ayuda de la siguiente manera ::

  python3 practica2.py -h
