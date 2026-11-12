# RIA P1 — Seguimiento de un vehículo en la ciudad (RoboboSim)

Un Robobo seguidor aprende por refuerzo (acciones continuas, Stable-Baselines3)
a circular detrás del guía (pelota verde) usando solo los blobs de su cámara,
leídos por ROS 2.

## Estructura

Los módulos compartidos con la P2 están en la raíz; lo exclusivo de la P1
(scripts de calibración y entrenamiento, modelos, registros, documentación)
está en `P1/`.

```
p1_robobo_env.py     Entorno Gymnasium (observación, acción, recompensa, reinicio)
p1_evaluar.py        Evaluador del anexo B (verdad del terreno solo aquí; escribe los CSV)
p1_graficas_eval.py  Métricas, tablas y figuras del anexo B desde los CSV de evaluación
config/p1_config.json  Configuración activa, la que usa la P2 (con POSX_SIGNO, CAMBIOS 35)

P1/
  p1_calibrar.py     Caracterización del sensor (apartado 5.1)
  p1_entrenar.py     Entrenamiento SB3, línea base aleatoria y curvas de aprendizaje
  config/            p1_config.json y calibracion.json con los que se entregó la P1
  modelos/           Políticas entrenadas: p1_modelo_<algoritmo>_s<semilla>.zip
  docs/              CAMBIOS.txt (registro de cambios de P1 y P2), memoria.txt, enunciado
  p1_logs/           CSV de Monitor, registros por paso y de evaluación
  p1_graficas/       Figuras y tablas de la P1
```

Las rutas son relativas al directorio de trabajo. Los comandos de la P1 se
lanzan **desde `P1/`** con la raíz en `PYTHONPATH` (así usan `P1/config`,
`P1/modelos` y `P1/p1_logs`); los de la P2, desde la raíz.

## Requisitos

- RoboboSim con el escenario de la ciudad y el guía circulando.
- ROS 2 con `robobo_ros2`, `stable-baselines3`, `gymnasium`, `matplotlib`, `robobosim`.
- Nodo puente del seguidor, solo con el módulo necesario:

```bash
ros2 run robobo_ros2 robobo_container --ros-args \
    -p ip:=host.docker.internal -p modules:="['blob']"
```

## Reproducción

```bash
cd P1
export PYTHONPATH=..

# 1. Calibración del sensor y volcado a config/p1_config.json
python3 p1_calibrar.py --distancia 0.375
python3 p1_entrenar.py --importar-calibracion

# 2. Comprobar el entorno y la configuración activa
python3 p1_entrenar.py --comprobar
python3 p1_entrenar.py --normalizacion

# 3. Línea base aleatoria
python3 p1_entrenar.py --aleatorio --episodios 8

# 4. Entrenamiento (3 semillas) -> modelos/p1_modelo_sac_s{0,1,2}.zip
python3 p1_entrenar.py --entrenar --algoritmo sac --pasos 20000 --semilla 0

# 5. Curvas de aprendizaje (media + banda de las semillas)
python3 p1_entrenar.py --graficas --grupos sac_s --ventana 10

# 6. Evaluación anexo B -> p1_logs/eval_<etiqueta>_<C>.csv
python3 ../p1_evaluar.py --algoritmo sac --semilla 0
python3 ../p1_evaluar.py --algoritmo aleatorio

# 7. Métricas, tablas y figuras del anexo B a partir de los CSV de evaluación
python3 ../p1_graficas_eval.py                    # todo p1_logs salvo 'aleatorio'
python3 ../p1_graficas_eval.py --etiquetas sac_s0 # sólo una política
python3 ../p1_graficas_eval.py --final C1=sac_s1 C2=sac_s2 C3=sac_s1
python3 ../p1_graficas_eval.py --solo-tablas      # sin matplotlib
```

Con la versión actual del contenedor la x del blob sale reflejada (CAMBIOS
35): para volver a ejecutar hoy las políticas de la P1 hay que usar
`--config ../config/p1_config.json` (POSX_SIGNO = -1), que es lo que hace
`p2_evaluar.py --sb3` en la comparación con la P2.

---

# Práctica 2 — Neuroevolución con NEAT

Mismo entorno, calibración y evaluador que la P1; el controlador es una red
evolucionada con NEAT-Python 2.0. Los ficheros de la P1 no cambian.

```
p2_neat.py            Evolución, reanudación, referencia del reto y reevaluación
p2_controladores.py   Controladores obs -> acción (NEAT, SB3, reto, aleatorio)
p2_evaluar.py         Anexo B con C1-C4 para cualquier controlador (CSV en p2_logs/)
p2_graficas.py        Curvas de evolución, especies, complejidad, causas de fin,
                      topología de la red y tablas del anexo B con paradas (p2_graficas/)
config/p2_neat.ini    Parámetros de NEAT
config/p2_config.json Calidad (A/B), duración del episodio, reto, simulador
p2_evol/<nombre>/     (se genera) configuración copiada, evolucion.csv, especies.csv,
                      episodios.csv, chk/p2-N, genomas/, mejor_genoma.pkl
```

Requisitos extra en la imagen nueva (se pierden al recrear el contenedor):

```bash
pip install --user robobosim                # no viene en la imagen
pip install --user --upgrade matplotlib     # el de Ubuntu no funciona con NumPy 2
```

RoboboSim con el multiplicador x1 y las físicas simplificadas, igual en la evolución, la evaluación y la comparación con la P1.

```bash
# Comprobaciones (indicaciones técnicas, apartado 10)
python3 p2_neat.py --comprobar --guia C3
python3 p2_neat.py --reto --guia C3 --episodios 3        # referencia -> p2_evol/referencias.csv

# Evolución incremental de un controlador único (una semilla por PC)
python3 p2_neat.py --evolucionar --nombre s0 --semilla 0 --generaciones 15 --guia C3
#   ... poner el guía en C4 en RoboboSim ...
python3 p2_neat.py --reanudar --nombre s0 --generaciones 15 --guia C4 --duracion 40
python3 p2_neat.py --reevaluar --nombre s0 --candidatos 5 --episodios 3 --guia C4

# Anexo B (C1-C4) y comparación con la P1
python3 p2_evaluar.py --neat p2_evol/s0/mejor_genoma.pkl --etiqueta neat_s0
python3 p2_evaluar.py --sb3 sac --modelo P1/modelos/p1_modelo_sac_s0.zip --configs C1 C2 C3
python3 p2_evaluar.py --reto
python3 p2_evaluar.py --aleatorio

# Figuras y tablas (sin ROS ni simulador)
python3 p2_graficas.py --evol s0 s1
python3 p2_graficas.py --red p2_evol/s0/mejor_genoma.pkl
python3 p2_graficas.py --anexo-b
```

Si la evolución se para sin mensaje (puente caído), se relanza el puente y se
reanuda: `--reanudar` usa el último `chk/p2-N`.
