# TFG_SPAM
Trabajo Final de Grado: Detección y Filtrado de SPAM utilizando Machine Learning y Modelos Matemáticos de Regresión

# Recursos utilizados
-> Enron-Spam dataset: Recopilado por V. Metsis, I. Androutsopoulos y G. Paliouras. Es un conjunto de datos que contiene un total de 17.171 mensajes de correo electrónico spam y 16.545 mensajes de correo electrónico no spam ("ham") (33.716 mensajes de correo electrónico en total). El conjunto de datos y la documentación originales se pueden encontrar en "https://www2.aueb.gr/users/ion/data/enron-spam/readme.txt".

Sin embargo, los conjuntos de datos originales se registran de tal manera que cada correo electrónico se encuentra en un archivo txt separado, distribuido en varios directorios.


# Opcion de ejecucion nro 1
# Pasos para utilizar devContainer (resumen)
1. Instalar plugin en vscode llamado "remote Development" con todos sus "extension packs"
2. Habilitar del servicio de docker
3. Ir a la extension "Remote Development" indicado con el icono "><", en la esquina inferior izquierda de vscode
4. Escoger entre las posibles opciones del menú desplegagle las referidas a "container"
5. Seleccionar "Reopen in Container"
6. Dentro de vscode, seleccionar una terminal en bash
7. Comprobar mediante el comando "id" y "whoami" que nos encontramos bajo el "user" vscode
8. Comprobar mediante "cat /etc/release" que se encuentra la configuración segun los especificado en la configuracion de "devcontainer.json"
9. Podemos comprobar mediante "pwd" el directorio dónde nos encontramos

# Pasos para ejecutar la comparativa
1. En un terminal bash ejecutar "python --version" para validar que la version es mayor a 3.8
2. Instalar los packages mediante "pip install -r requirements.txt"
3. Para ejecutar la comparativa correspondiente al "test", a utilizar "python tfg_v1_simple_ROC_Curve.py"
4. Para ejecutar la comparativa correspondiente al dataset "enron", a utilizar "python tfg_v2_enron_ROC_Curve.py"

# Pasos para finalizar el devContainer
1. Ir a la extension "Remote Development" indicado con el icono "><", en la esquina inferior izquierda de vscode. 
2. Seleccionar "Close Remote Connection"
3. O simplemente cerrar vscode sin inconveniente.

# Opcion de ejecucion nro 2
# Pasos para utilizar venv (resumen)
1. Instalar y utilizar "venv": https://docs.python.org/es/3/tutorial/venv.html


# Links útiles sobre devContainer
-> devContainer mediante vscode: https://code.visualstudio.com/docs/devcontainers/containers
-> cómo crear un devContainer: https://code.visualstudio.com/docs/devcontainers/create-dev-container
-> tutorial sobre devContainer sobre vscode: https://code.visualstudio.com/docs/devcontainers/tutorial
-> devContainer video: https://www.youtube.com/watch?v=b1RavPr_878&ab_channel=VisualStudioCode
-> cómo usarlo: https://learn.microsoft.com/es-es/training/modules/use-docker-container-dev-env-vs-code/



