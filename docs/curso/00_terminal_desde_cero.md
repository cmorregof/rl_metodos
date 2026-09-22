# La terminal desde cero

**Métodos numéricos · guía para la primera sesión · léela antes de tocar nada**

## 1 · Qué es la terminal

Todo lo que haces con el ratón (abrir una carpeta, abrir un programa, crear un archivo) se
puede hacer **escribiendo órdenes**. El programa donde se escriben esas órdenes es la
**terminal**. Es una ventana negra (o azul) con un cursor parpadeando. Parece de hacker; no
lo es. Es solo otra forma de decirle al computador qué hacer, más precisa y más rápida.

Una orden se escribe, se pulsa **Enter**, y el computador responde con texto. Eso es todo.

¿Por qué la usamos en este curso? Porque los programas que vamos a correr no tienen botones.
Se ejecutan con una orden, escriben su resultado en la terminal y guardan archivos en una
carpeta. Así trabaja casi toda la matemática computacional.

## 2 · Abrir la terminal

**Windows (los PCs del laboratorio):** pulsa la tecla Windows, escribe `terminal` y abre
**Terminal** (o **Windows PowerShell** si no aparece). Maximiza la ventana: la vamos a
necesitar grande.

**Mac:** Cmd + espacio, escribe `terminal`, Enter.

Verás algo así:

```
PS C:\Users\estudiante>
```

Eso se llama el **prompt**. Te dice **en qué carpeta estás** (`C:\Users\estudiante`) y espera
una orden. En Mac se ve parecido pero con `/Users/nombre` y termina en `%` o `$`.

## 3 · Las cinco órdenes que necesitas

Escribe cada una y pulsa Enter. Fíjate en lo que responde.

| orden | qué hace | ejemplo |
|---|---|---|
| `pwd` | te dice en qué carpeta estás (*print working directory*) | `pwd` |
| `ls` | lista lo que hay en la carpeta actual (*list*) | `ls` |
| `cd nombre` | entra en la carpeta `nombre` (*change directory*) | `cd Documents` |
| `cd ..` | sube a la carpeta de arriba | `cd ..` |
| `mkdir nombre` | crea una carpeta | `mkdir metodos` |

Tres trucos que ahorran mucho sufrimiento:

* **Tab** completa nombres. Escribe `cd Doc` y pulsa Tab: se convierte en `cd Documents`.
* **Flecha arriba** repite la orden anterior. No vuelvas a escribir nada largo.
* Si escribes algo mal, la terminal te lo dice en rojo o con "no se reconoce". No pasa nada.
  Corrige y vuelve a intentar. **No puedes romper el computador con estas órdenes.**

Prueba ahora:

```
cd Documents
mkdir metodos
cd metodos
pwd
```

Deberías ver `C:\Users\<tu usuario>\Documents\metodos`. Esa es tu carpeta de trabajo del curso.

## 4 · Qué significa "correr código"

Un programa es un archivo de texto con instrucciones. **Python** es el programa que lee esas
instrucciones y las ejecuta. Cuando en la terminal escribes

```
python archivo.py
```

le estás diciendo: "Python, lee `archivo.py` y haz lo que dice". Python trabaja, escribe
resultados en la terminal, y cuando termina vuelve a aparecer el prompt.

Comprueba que Python está instalado:

```
python --version
```

Debe responder `Python 3.1x.x`. Si dice que no se reconoce, avisa al docente (en Windows a veces
la orden es `py` en vez de `python`; prueba `py --version`).

## 5 · Descargar el material del curso

El código está en GitHub, una página donde se comparten programas. No necesitas cuenta.

1. Abre en el navegador: `https://github.com/cmorregof/rl_metodos`
2. Botón verde **Code** → **Download ZIP**.
3. El archivo `rl_metodos-main.zip` queda en `Descargas` (`Downloads`). Descomprímelo
   (clic derecho → *Extraer todo*) y mueve la carpeta `rl_metodos-main` a `Documents\metodos`.

Vuelve a la terminal. Sigues dentro de `metodos` (míralo en el prompt, o comprueba con `pwd`).
Entra en la carpeta que acabas de mover:

```
cd rl_metodos-main
ls
```

Deberías ver las carpetas `01_biseccion`, `02_punto_fijo`, … Si `ls` no las muestra, no estás
en la carpeta correcta: usa `pwd` para ver dónde estás y `cd` para moverte.

**Si cerraste la terminal o estás en otra carpeta**, escribe la ruta completa desde tu carpeta
de usuario. En Windows las carpetas se separan con `\`; en Mac y Linux con `/`, y una `\` en Mac
significa otra cosa (se "come" la letra siguiente):

```
cd Documents\metodos\rl_metodos-main     (Windows)
cd Documents/metodos/rl_metodos-main     (Mac)
```

## 6 · Instalar el proyecto de hoy

Cada carpeta es un proyecto independiente. Entra en el primero y crea un "entorno": una
carpeta privada donde Python instala lo que ese proyecto necesita sin tocar nada más del
computador.

```
cd 01_biseccion
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
```

(Mac: `.venv/bin/python -m pip install -e ".[dev]"`.)

La segunda orden tarda unos segundos y no dice nada. La tercera escribe muchas líneas;
lo importante es que la última diga `Successfully installed ...`. Si aparece un error rojo,
levanta la mano.

Desde ahora, "el Python del proyecto" es `.venv\Scripts\python`. Cada orden del laboratorio
empieza así.

## 7 · Correr el agente

```
.venv\Scripts\python -m bisectrl --seed 1
```

Se abre una pantalla llena de paneles que se mueven. Es un programa aprendiendo en vivo.
Dura uno o dos minutos. Si la ventana es pequeña se verá recortado: maximízala y vuelve a
correr (flecha arriba, Enter).

Cuando termina, imprime un informe y dice dónde lo guardó: en la carpeta `runs`, en una
subcarpeta con la fecha y hora. Ábrela desde el explorador de archivos y abre `informe.md`
con el Bloc de notas (o con VS Code si está instalado). Es texto plano: se lee igual.

## 8 · Si algo se atasca

* El programa no termina y quieres pararlo: **Ctrl + C**.
* La terminal "se quedó rara": ciérrala, abre otra y vuelve a la carpeta con `cd`.
* "No se reconoce el comando": revisa mayúsculas y que estés en la carpeta correcta (`pwd`).
* Windows PowerShell dice algo de "ejecución de scripts deshabilitada": por eso usamos
  `.venv\Scripts\python` directamente y nunca `activate`. Si te pasa, es que escribiste
  `activate`; no hace falta.

## 9 · Lo que acabas de aprender

Sabes abrir una terminal, moverte por carpetas, comprobar que Python existe, crear un
entorno para un proyecto y ejecutar un programa que escribe archivos. Eso es el 90 % de lo
que hace un matemático computacional al empezar cualquier cosa. El otro 10 % es leer lo que
el programa dice cuando falla, y eso se aprende fallando.
