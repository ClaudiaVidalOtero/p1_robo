# RIA P1 — Seguimiento de un vehículo en la ciudad (RoboboSim)

Un Robobo seguidor aprende por refuerzo (acciones continuas, Stable-Baselines3)
a circular detrás del guía (pelota verde) usando solo los blobs de su cámara,
leídos por ROS 2.

## Estructura

```
p1_robobo_env.py     Entorno Gymnasium (observación, acción, recompensa, reinicio)
p1_calibrar.py       Caracterización del sensor (apartado 5.1)
p1_entrenar.py       Entrenamiento SB3, línea base aleatoria y curvas de aprendizaje
p1_evaluar.py        Evaluador del anexo B (verdad del terreno solo aquí)

config/
  p1_config.json                    Configuración activa (constantes calibradas + recompensa)
  p1_config_ppo_s0.json             Configuración con la que se entrenó ppo_s0
  calibracion.json                  Medidas crudas de p1_calibrar.py
  calibracion_invalida_2026-10-14.json  Calibración descartada (ver docs/CAMBIOS.txt)
modelos/             Políticas entrenadas: p1_modelo_<algoritmo>_s<semilla>.zip
docs/
  CAMBIOS.txt        Registro de cambios (qué / dónde / por qué / efecto)
  memoria.txt        Notas de trabajo para la memoria
  RIA_P1_enunciado.pdf

p1_logs/             (se genera) CSV de Monitor, registros por paso y de evaluación
p1_checkpoints/      (se genera) checkpoints durante el entrenamiento
```

Todos los scripts se lanzan **desde esta carpeta** (las rutas son relativas).

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
# 1. Calibración del sensor y volcado a config/p1_config.json
python3 p1_calibrar.py --distancia 0.375
python3 -c "from p1_robobo_env import importar_calibracion as i; i()"

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
python3 p1_evaluar.py --algoritmo sac --semilla 0
python3 p1_evaluar.py --algoritmo aleatorio
python3 p1_evaluar.py --algoritmo sac --semilla 0 --resumen   # tablas desde los CSV
```

Para evaluar un modelo entrenado con otra configuración, pasar la suya:

```bash
python3 p1_evaluar.py --algoritmo ppo --modelo modelos/p1_modelo_ppo_s0.zip \
    --config config/p1_config_ppo_s0.json
```
