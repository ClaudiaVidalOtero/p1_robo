# RIA P1 — Seguimiento de un vehículo en la ciudad (RoboboSim)

Asignatura Robótica Inteligente Aplicada (Grado IA, UDC, 2026/27). Práctica 1, por parejas.
Un Robobo **seguidor** aprende por refuerzo (acciones continuas) a circular detrás de un Robobo **guía** (pelota verde en la parte trasera) por el anillo exterior del mundo **"Follow City"** de RoboboSim. Se entrena en simulación y la política se ejecuta en el robot real (lab 0.0 FIC) **sin reentrenar**.

Fuente de verdad: `docs/RIA_P1_enunciado.pdf` (versión simplificada, sustituye a la anterior).

**Entrega: domingo 18/10/2026, campus online (.zip).** Defensa oral: hay que conocer y saber modificar el código sin IA.

Avances pedidos por el profesor para la semana siguiente:
- **Obligatorio:** política entrenada en Follow City con el guía a **velocidad variable** (= C3).
- **Opcional:** mejorarla para que el seguidor no se salga del carril.
- Si el simulador va lento: activar "Físicas Simplificadas" (mismo ajuste en entrenamiento y evaluación; declararlo).

## Objetivo de la tarea

Mantener la separación en la banda objetivo, sin chocar, sin perder al guía de vista y sin salirse del carril (este último **con tolerancia**).

- El carril **no es inferible** de la observación → según el enunciado (ap. 4) **no puede ser término de la recompensa**; solo condición de fin y métrica. En el anexo B **salirse del carril no es fallo**: se juzga en los vídeos. Se valoran estrategias para que el seguidor vaya en el carril derecho.

## Stack

- **Gymnasium**: estructura del entorno
- **Stable-Baselines3**: acciones continuas. **Obligatorio probar al menos SAC y PPO en cada configuración del anexo B**; justificar la elección (cada paso cuesta tiempo real)
- **ROS 2 + `robobo_ros2`**: única vía a sensores y actuadores del seguidor
- Módulo `sim` del nodo puente: solo en simulación, **no existe en el robot real**

## Restricciones críticas (no romper)

- **Observación = solo sensores del seguidor vía ROS 2.** Prohibido meter pose del guía/seguidor, geometría del carril, separación real o error lateral. Regla: lo que la política necesita para decidir debe existir también en el robot real.
- Información privilegiada del sim **sí** vale para: preparar el inicio del episodio, decidir la terminación y registrar métricas.
- **Recompensa:** se recomienda usar solo magnitudes inferibles de la observación. Para cada término hay que decir en la memoria de qué magnitud depende y cómo la infiere el robot; si se usa algo no observable, justificarlo.
- **Decisión del equipo:** observación y recompensa salen solo de los sensores del seguidor (blobs; IR opcional, el enunciado valora usarlos cuando el guía está muy cerca).
- **`step()` debe devolver una observación posterior a aplicar la acción.** Garantizarlo explícitamente (esperar al primer mensaje de blobs con timestamp posterior al envío de la acción) y explicarlo en la memoria.
- Espacio de acción **continuo**. Interfaz de actuación libre (cmd_vel / ruedas): declarar cuál y por qué.
- Parámetros dependientes de calibración en **archivo de configuración**, no en código.
- Mismo multiplicador de velocidad del simulador (y mismas físicas) en entrenamiento y evaluación (declararlo).
- El entorno debe poder correr con el protocolo del anexo B y **sin simulador** (reinicio manual) para la prueba real.
- Nodo puente solo con los módulos necesarios; el de cámara no hace falta para blobs.

## Decisiones de diseño actuales

- Observación: `[posx, size, posy]` del blob (normalizados; definir qué se devuelve cuando la pelota no se ve). Rango del área del blob: 1–1000.
- Acción: velocidad lineal y angular (`cmd_vel`; se detiene si no recibe órdenes en 0,5 s de tiempo real).
- Recompensa: cada término separado, con su coeficiente numérico y la magnitud observable de la que depende.
- Detección: `smartphone/set_active_color_blobs` (green). **Un solo blob verde** por imagen; la ciudad real tiene otros elementos verdes.
- Periodo de control: decisión de diseño, declararlo. Blobs a 5 Hz en tiempo real.
- Reinicio por conexión directa a RoboboSim (ver docs/CAMBIOS.txt nº 6 y 20): el puente se lanza **solo con `['blob']`**, porque la conexión del módulo `sim` del puente se caía.

## Ficheros

```
README.md                 # Estructura e instrucciones de reproducción
p1_robobo_env.py          # Entorno Gymnasium + cargar_config()/importar_calibracion()
p1_calibrar.py            # Caracterización del sensor
p1_entrenar.py            # Entrenamiento / evaluación / gráficas SB3
p1_evaluar.py             # Evaluador del anexo B (verdad del terreno por conexión directa a RoboboSim; escribe los CSV)
p1_graficas_eval.py       # Métricas, tablas y figuras del anexo B desde los CSV (sin ROS ni simulador)
config/p1_config.json     # Configuración calibrada (RUTA_CONFIG)
config/calibracion.json   # Medidas crudas de p1_calibrar.py (RUTA_CALIBRACION)
modelos/                  # p1_modelo_<alg>_s<semilla>.zip (RUTA_MODELOS)
docs/CAMBIOS.txt          # Registro de todos los cambios (qué/dónde/por qué/efecto)
docs/memoria.txt          # Notas para la memoria
p1_logs/, p1_checkpoints/ # Generados al ejecutar (CSV, Monitor, TensorBoard, checkpoints)
```

Los scripts se ejecutan desde la raíz del proyecto (rutas relativas).

## Entregables (.zip)

1. **Memoria PDF**: máx. 6 páginas + anexos, Calibri 11 pt, interlineado sencillo, justificado, figuras/tablas numeradas con pie, coeficientes de la recompensa explícitos. Estructura:
   1. Observación, acción, recompensa, terminación/truncamiento, reinicio de episodios (+ decisiones de diseño relevantes)
   2. Pruebas para **cada una de las 3 configuraciones** del anexo B, con curvas de aprendizaje (distintos algoritmos, parámetros y varias ejecuciones antes de elegir la mejor política por configuración)
   3. Tablas de métricas del anexo B: línea base aleatoria (opcional pero se valora) y solución final de cada configuración
   4. Conclusiones: problemas y mejoras
   - Transferencia real: repetir caracterización en el robot real; qué se degrada y por qué (se evalúa el análisis, no el éxito)
2. **Declaración de uso de IA**: 1 página máx.; contribución propia vs. de la IA
3. **Código fuente comentado** con instrucciones para reproducir la política final de cada configuración
4. **Evidencias**: modelos + configuración, registro por episodio de cada entrenamiento y por paso de cada evaluación (CSV) + el código que los lee, **vídeos de la política final en las 3 configuraciones**

**Calificación:** proporcional a qué configuración se resuelva (C1 < C2 < C3 en dificultad); entrenar una política para cada una.

## Curvas de aprendizaje

- Recompensa y duración por episodio (mín.); recomendado fracción por causa de fin (colisión, pérdida, límite de tiempo)
- Eje X: pasos de `step()` (no episodios ni tiempo); eje Y con magnitud y unidad
- Media móvil (ventana en episodios) + serie cruda tenue
- Solución final con **≥3 semillas**: media + banda (indicar min/max o desv. típica)
- Línea horizontal: recompensa media de la política aleatoria en el entorno de entrenamiento
- Datos de Monitor/CSV, figuras con Matplotlib/Seaborn propias (no capturas de TensorBoard)
- Pie: configuración, nº semillas, ventana, pasos totales; comentar cada figura en el texto

## Protocolo de evaluación (anexo B) — sin modificaciones

- **3 ejecuciones de 120 s** (una por configuración), política **determinista**; reinicio de la simulación, ambos robots donde los coloca el escenario, seguidor parado (se admiten acciones de arranque predeterminadas)

| Config | Modo | Vel. mín | Vel. máx |
|---|---|---|---|
| C1 Lenta constante | Crucero | 10 | 10 |
| C2 Contraste | Crucero | 10 | 40 |
| C3 Fluctuante | Variable | 10 | 40 |

- Referencia: **pelota verde** = pose del guía desplazada **0,08 m hacia atrás** (en RoboboSim 4.3.0 se puede usar directamente la pose del guía)
- Separación = distancia centro del seguidor ↔ pelota. **Banda objetivo 0,20–0,45 m**
- Desviación = ángulo entre el rumbo del seguidor y la dirección a la pelota (equivale a la x del blob; compararlas comprueba la calibración)
- Fallos (cada paso): **colisión** separación < 0,12 m; **guía perdido** separación > 0,90 m **o** pelota sin detectar > 3 s seguidos. Otros valores se pueden usar si se justifican. Otra causa de terminación del entorno propio se registra y no cuenta como completado. **Salirse del carril no es fallo.**
- Registro por paso: instante, posición del seguidor y de la pelota (o guía), separación, desviación lateral, acción
- 4 métricas, en conjunto y por configuración: completados (nº y %), desenlaces por causa, separación (% pasos en banda + media), desviación (media de |ángulo| en grados + % pasos < 15°)
- La pose del guía requiere un segundo contenedor con el índice del otro robot y solo el módulo `sim` (o conexión directa a RoboboSim). Solo en el evaluador, nunca en la observación.

## Comandos

```bash
# Nodo puente (seguidor): solo blobs (el reinicio va por conexión directa)
ros2 run robobo_ros2 robobo_container --ros-args \
    -p ip:=host.docker.internal -p modules:="['blob']"

python3 p1_entrenar.py --comprobar
python3 p1_entrenar.py --aleatorio --episodios 8
python3 p1_entrenar.py --entrenar --algoritmo {ppo,sac} --pasos 20000 --semilla N
python3 p1_entrenar.py --evaluar  --algoritmo {ppo,sac} --semilla N
python3 p1_entrenar.py --graficas --ventana 10 --ref <recompensa_aleatoria>
python3 p1_evaluar.py --algoritmo {ppo,sac,aleatorio} --semilla N [--configs C1 C2 C3]
python3 p1_graficas_eval.py [--etiquetas ...] [--solo-tablas]   # métricas, tablas y figuras del anexo B
```

## Interfaz ROS 2 (namespace `/robobo/robot_0`)

| Recurso | Tipo | Notas |
|---|---|---|
| `smartphone/color_blobs` | BlobArray, 5 Hz | color, x, y, size; se publica también vacía |
| `smartphone/set_active_color_blobs` | servicio | red, green, blue, custom |
| `base/cmd_vel` | Twist | linear.x m/s, angular.z rad/s; se detiene a los 0,5 s sin órdenes |
| `base/move_wheels(_time)` | servicio | velocidad de ruedas en unidades del robot |
| `base/move_tilt` / `base/move_pan` | servicio | angle, speed |
| `base/wheel/{left,right}/speed` | Int32 | velocidad medida |
| `sim/robot_location`, `sim/pose` | — | mm y grados; plano (x, z); solo sim |
| `sim/set_robot_location`, `sim/reset_simulation` | servicio | colocar robot / reiniciar escena |

## Notas

- Los errores típicos de asistentes (p. ej. observación obsoleta en `step()`) no generan mensajes y solo se ven porque la política no aprende: verificar con pruebas.
- Los resultados exigen ejecuciones reales. Un resultado negativo bien diagnosticado, con evidencia, es válido.
- Todo cambio de código/config se anota en `docs/CAMBIOS.txt`.
