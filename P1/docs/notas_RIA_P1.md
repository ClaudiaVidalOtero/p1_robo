# RIA P1 — Seguimiento de un vehículo en Follow City

**Notas verificadas para redactar la memoria (curso 2026/27)**

> **Estado:** borrador de trabajo actualizado el 17/10/2026.
> Separa lo que ya está respaldado por los ficheros actuales de lo que todavía falta por decidir o probar. No hay que presentar como resultado final una ejecución incompleta, una configuración del guía que no quedó registrada o una cifra histórica que no coincida con los CSV vigentes.

**Formato de la memoria final:** máximo 6 páginas más anexos, Calibri 11, interlineado sencillo, texto justificado y pies numerados en figuras y tablas.

**Estructura solicitada:**

1. Entorno: observación, acción, recompensa, fin y reinicio.
2. Pruebas de C1, C2 y C3: algoritmos, parámetros, semillas y curvas.
3. Tablas del anexo B: línea base y solución final por configuración.
4. Conclusiones, problemas y mejoras.

Además, hay que incluir la caracterización del sensor y el análisis de la transferencia al robot real.

---

## 1. Entorno y decisiones de diseño

### 1.1. Observación

La política recibe únicamente los blobs verdes del sensor del seguidor, que ROS 2 publica en `/robobo/robot_0/smartphone/color_blobs`. El callback se queda con el blob verde de mayor área. El detector entrega como máximo uno, ya que, tanto en el simulador como en el robot real lo puede confundir con otros objetos verdes.

Los valores de la configuración activa (`config/p1_config.json`) son:

```
observación = [posx, size, posy]      (Box float32 de dimensión 3)
posx = clip((x_blob - 50) / 50, -1, 1)    # centro de imagen = 0
size = clip(size_blob / 330, 0, 1)
posy = clip(y_blob / 100, 0, 1)
```

La posición horizontal y vertical del detector se normaliza a una escala de 0 a 100, y `size` se divide por `SIZE_MAX = 330`. Si el blob no aparece, o es menor que `SIZE_MIN_VISTA = 12,9`, se considera no visible y la observación es `[0, 0, 0]`. 

Usamos `posy`  para diferenciar una pelota cortada por abajo de una pelota lejana con un `size` parecido. La observación no incluye poses, separación real, geometría del carril ni error lateral.


### 1.2. Acción y periodo de control

La acción es continua, `Box([-1,-1], [1,1])`, con dos componentes normalizados:

```
v = 0 + (a0 + 1) / 2 * (0,25 - 0) m/s      →  0 a 0,25 m/s
w = a1 * 0,5 rad/s                          →  -0,5 a +0,5 rad/s
```

El puente de `cmd_vel` deja de actuar si pasan 0,5 s de tiempo real sin recibir órdenes.

El coste real por paso que muestran los monitores es mayor que 0,2 s: SAC tarda aproximadamente 0,436–0,438 s/paso contando comunicación y reinicios. Este coste de ejecución, igual que la respuesta del robot a `cmd_vel`, no debe confundirse con el multiplicador de velocidad de RoboboSim. No utilizamos ningún factor de aceleración en el simulador.

### 1.3. Sincronización de la observación en `step()`

`step()` publica la acción, espera `PASO_S`, vuelve a publicar, vacía los mensajes que ya estaban en cola y espera un mensaje nuevo de blobs antes de construir y devolver la observación. El callback cuenta cada mensaje recibido, incluidas las listas vacías, y `_esperar_lectura_nueva()` exige que ese contador aumente. De este modo se evita devolver sin más el blob guardado antes de la acción.

Una precisión importante al describir la implementación: el código actual **no** compara una marca temporal del sensor con el instante de publicación. Lo que valida es que llegue un callback nuevo después de vaciar la cola. Por tanto, no hay que afirmar que se comprueba un timestamp posterior, salvo que antes se verifique que el tipo `BlobArray` lo proporciona y se añada esa comprobación.

El timeout de espera de lectura es de 2,0 s. Si el flujo se interrumpe, `step()` lanza `RuntimeError`.

### 1.4. Recompensa

Los coeficientes salen de `config/p1_config.json`. La recompensa se construye con datos del blob y de la acción, no con la verdad del terreno. Sea `p = posx`, `s = size`, `e(s)` la excursión normalizada fuera de la banda de size y `a1` el giro normalizado.

**Blob no visible**

- `r = -1,0` por paso (`R_PERDIDO_PASO`).

**Blob visible**

| Término | Valor |
|---|---|
| Vista | +0,5 |
| Centrado | −0,5 · \|p\| |
| Separación | +1,0 si 99,8 ≤ s ≤ 207 y no está cortado; en otro caso −0,8 · e(s), con e(s) en [0, 1] |
| Giro | −0,3 · \|a1\| |
| Cambio de giro | −0,3 · \|a1 − a1_anterior\| |

**Términos terminales**

| Término | Valor |
|---|---|
| Choque | −20,0 |
| Guía perdido | −5,0 |

Cada término depende de una magnitud observable: la visibilidad y el `size` del blob (vista, separación y pérdida), `posx` normalizada (centrado) y la acción actual y la anterior (giro y cambio de giro). `posy` sirve para comprobar si el blob está cortado cuando `Y_CHOQUE` está calibrado, pero en la configuración activa `Y_CHOQUE = null`, así que esa prueba está desactivada. El IR también está desactivado.

No hay término de carril: como se explica en el apartado 4, el carril no puede inferirse de la observación. El término de centrado ayuda a mantener al guía dentro del campo visual, pero no mide ni garantiza que el seguidor permanezca en el carril.

Con los coeficientes actuales, el peor paso visible antes de una terminación sigue siendo mejor que perder el blob (−0,8 > −1,0). La intención es que no compense abandonar al guía. Tampoco hay que interpretar `SIZE_OBJ = 139,1` como una recompensa de objetivo: el código da una recompensa constante en toda la banda de size.

**Justificación de los límites de la banda**

- El protocolo mide la distancia del centro del seguidor a la pelota, con una banda de 0,20–0,45 m.
- La calibración, en cambio, es de cámara a pelota. La diferencia cámara-centro que se usa en la configuración es `CAMARA_OFFSET_M = 0,09 m`; la evaluación que motivó el cambio dio medianas de 0,088 / 0,087 / 0,090 m (es una comprobación preliminar, no una calibración universal).
- 0,45 m desde el centro equivale aproximadamente a 0,36 m desde la cámara y a un size de 99,8. El extremo cercano de 0,20 m equivale a 0,11 m desde la cámara, pero por debajo de unos 0,25 m la pelota se corta y `size` deja de ser fiable, de modo que el size alto de la recompensa se limita a 207.
- En consecuencia, la banda de recompensa alcanzable con `size` corresponde a unos 0,34–0,45 m desde el centro, y no a todo el rango 0,20–0,45 m. La zona de 0,20–0,34 m del protocolo no se puede distinguir solo con `size`. Este compromiso hay que explicarlo en la memoria, no ocultarlo.

### 1.5. Caracterización del sensor

`p1_calibrar.py` hace una caracterización manual: se coloca la pelota a una distancia medida, se toman 20 lecturas nominales en reposo a 5 Hz y se registran `size` y `posx`, con la cámara inclinada 90°. El fichero `config/calibracion.json` contiene mediciones a 0,20, 0,25, 0,30, 0,40, 0,50, 0,60, 0,75 y 1,00 m. Estas son las medianas del size de las lecturas centradas con size positivo:

| Distancia (m) | 0,20 | 0,25 | 0,30 | 0,40 | 0,50 | 0,60 | 0,75 | 1,00 |
|---|---|---|---|---|---|---|---|---|
| Mediana de size | 84 | 181 | 182 | 82 | 47 | 34 | 23 | 15 |

El JSON conserva entre 10 y 11 registros de etapa por distancia, de los cuales 3 son centrados y positivos. Cada medida de size en reposo resume hasta 20 lecturas nominales, pero no guarda las 20 muestras individuales. Hay dispersión y una meseta (no monotonicidad) alrededor de 0,25–0,40 m, por lo que `size` es una estimación ruidosa y no una medida métrica precisa.

`importar_calibracion()` ajusta `size ≈ K/d²` usando distancias ≥ 0,25 m y deriva automáticamente los límites de `config/p1_config.json`. La configuración activa es: `SIZE_MAX = 330`, `SIZE_MIN_VISTA = 12,9`, `SIZE_BANDA_MIN = 99,8`, `SIZE_BANDA_MAX = 207` y `SIZE_CHOQUE = 323,4`, con tilt de 90°.

> **Pendiente:** todavía no se ha caracterizado el sensor en el robot real. Hay que repetir allí esta medición antes de hacer afirmaciones cuantitativas sobre la transferencia.

### 1.6. Terminación, truncamiento y reinicio

**Entrenamiento (configuración actual)**

- **Guía perdido:** `PERDIDO_MAX = 12` lecturas o pasos consecutivos sin blob, es decir, 2,4 s nominales con `PASO_S = 0,2 s`. Si no se trata de una desaparición por cercanía, el episodio termina como guía perdido y resta 5 puntos terminales.
- **Choque:** `Y_CHOQUE = null` desactiva la detección por borde inferior. Quedan `SIZE_CHOQUE = 323,4` como umbral de respaldo y la regla de desaparición: si el blob deja de verse tras `size ≥ 144` y `|posx| < 0,6`, se interpreta que el seguidor está demasiado cerca. `USAR_IR_CHOQUE = false`. Resta 20 puntos.
- **Truncamiento:** `PASOS_MAX = 300`, equivalente a 60 s nominales de control. No son 60 s de reloj, porque el coste real por paso incluye esperas de ROS y reinicios.

No hay que confundir estos umbrales propios del entrenamiento con los de evaluación. Durante `p1_evaluar.py` las terminaciones propias se anotan, pero el episodio solo lo corta el protocolo del anexo B.

**Reinicio en simulación**

Con `RESET_DIRECTO = true`, la escena se reinicia mediante una conexión directa a RoboboSim, no a través del módulo `sim` del puente. El puente del seguidor se lanza únicamente con `modules:="['blob']"`; el módulo de cámara no hace falta. Tras el reset se esperan `RESET_ESPERA_S = 1,5 s`, se fija el tilt y se activa el detector verde.

Con `POS_INICIO = null` no se teletransporta al seguidor, sino que se usa la colocación inicial de la escena. Hay un intervalo mínimo de 4 s entre resets. El guía sigue circulando durante los pasos y no espera al seguidor; el reinicio de entrenamiento reinicia la escena completa.

**Sin simulador**

`RoboboSeguidorEnv(usar_simulador=False)` pide a la persona que coloque al seguidor detrás del guía y pulse Enter. La política no necesita el módulo `sim` en el robot real.

---

## 2. Entrenamiento, experimentos y curvas

### 2.1. Algoritmos e hiperparámetros

Se usa Stable-Baselines3 con `MlpPolicy` y acciones continuas. Estos son los parámetros actuales en `p1_entrenar.py`:

| Parámetro | SAC | PPO |
|---|---|---|
| learning_rate | 3e-4 | 3e-4 |
| buffer_size | 50 000 | — |
| learning_starts | 500 | — |
| batch_size | 256 | 64 |
| tau | 0,005 | — |
| gamma | 0,99 | 0,99 |
| train_freq | 1 | — |
| gradient_steps | 1 | — |
| n_steps | — | 512 |
| n_epochs | — | 10 |
| gae_lambda | — | 0,95 |
| clip_range | — | 0,2 |
| ent_coef | — | 0 |

**Justificación.** SAC es off-policy y reutiliza las transiciones guardadas en un buffer, de modo que cuando las interacciones son costosas aprovecha mejor cada paso. PPO es on-policy y descarta cada lote tras actualizar; se prueba como comparación obligatoria, aunque sus actualizaciones necesitan pasos nuevos. PPO usa lotes de 512 pasos, lo que da unas 40 actualizaciones en 20 000 pasos. Este razonamiento no sustituye a comparar los resultados.

### 2.2. Entrenamientos y artefactos comprobados en el proyecto

Resumen leído de `p1_logs/monitor_*.csv`. El tiempo `t` puede incluir esperas y reinicios, y los pasos sumados pueden superar ligeramente los 20 000 al cerrar el último episodio.

| Ejecución | Episodios | Pasos (monitor) | Recompensa media | Longitud |
|---|---|---|---|---|
| SAC semilla 0 | 155 | 19 887 | 85,457 | 8–300 |
| SAC semilla 1 | 148 | 19 896 | 101,368 | 8–300 |
| SAC semilla 2 | 156 | 19 823 | 80,862 | 6–300 |
| PPO semilla 0 (actual) | 819 | 20 433 | −16,536 | 5–112 |

Tiempos acumulados de los monitores y coste medio observado:

| Ejecución | Tiempo acumulado | Pasos | Coste medio |
|---|---|---|---|
| SAC s0 | 8 704,84 s | 19 887 | 0,438 s/paso |
| SAC s1 | 8 673,37 s | 19 896 | 0,436 s/paso |
| SAC s2 | 8 687,48 s | 19 823 | 0,438 s/paso |
| PPO s0 actual | 12 549,49 s | 20 433 | 0,614 s/paso |

**Modelos presentes en `modelos/`:** `p1_modelo_sac_s0.zip`, `p1_modelo_sac_s1.zip`, `p1_modelo_sac_s2.zip` y `p1_modelo_ppo_s0.zip`.

Actualmente no aparece `config/p1_config_ppo_s0.json`, aunque el registro de cambios histórico lo menciona.

**Limitaciones de trazabilidad**

- La configuración del guía se cambia a mano en RoboboSim y no queda escrita en los CSV de Monitor. Por eso, «Variable 10–40» para SAC/PPO s0 es la configuración prevista según las notas del equipo, no un hecho que se pueda comprobar con el monitor. No se debe presentar como metadato probado sin recuperar evidencia del lanzamiento.
- Tampoco consta el multiplicador de velocidad ni de físicas de cada entrenamiento.
- La línea base de recompensa aleatoria guardada en `pasos_aleatorio_s0.csv` (media histórica −15,7) corresponde a una banda anterior, así que no sirve como referencia para las curvas actuales. Hay que repetirla con `config/p1_config.json`.

### 2.3. Curvas de aprendizaje que exige el enunciado

Para cada entrenamiento final:

- Eje X: llamadas acumuladas a `step()`. Eje Y con magnitud y unidad.
- Serie cruda tenue y media móvil de 10 episodios.
- Para al menos tres semillas, media y banda mín–máx.
- Como mínimo, la recompensa acumulada por episodio (adimensional) y la duración (en pasos; si se expresa en segundos, indicar que cada paso nominal son 0,2 s). Añadir la fracción de episodios según la causa de fin (choque, pérdida, truncamiento).
- Línea horizontal con la recompensa media aleatoria, recalculada con la recompensa actual.
- Pie de figura con la configuración del guía, el algoritmo y sus parámetros, las semillas, la ventana y los pasos totales; la figura se comenta en el texto.
- No usar capturas de TensorBoard.

`p1_entrenar.py --graficas` agrupa los monitores, usa `--ventana 10`, dibuja la media y la banda mín–máx, y admite `--ref` para la línea aleatoria. También imprime un texto base para el pie. Ya existen `p1_logs/fig_sac_curvas.png` y `p1_logs/fig_sac_ppo.png`, pero hay que rehacerlas o revisarlas con la referencia de recompensa correcta y con una anotación de configuración verificable.

**Lectura actual de los monitores**

- SAC s0, s1 y s2 tienen, respectivamente, 46/155, 49/148 y 44/156 episodios de 300 pasos, y las tres recompensas medias globales son positivas.
- PPO s0 actual no pasa de 112 pasos por episodio y su recompensa media es negativa.
- Estas cifras globales son descriptivas y no sustituyen a las curvas por semilla.

### 2.4. Matriz de experimentación requerida y estado

El anexo B define tres configuraciones del guía:

| Config | Modo | Descripción |
|---|---|---|
| C1 | Crucero | 10–10 |
| C2 | Crucero | 10–40 |
| C3 | Variable | 10–40 |

Para cada una hay que probar al menos SAC y PPO, comparar parámetros y ejecuciones, y entrenar una política final. La selección definitiva debe tener al menos tres semillas.

**Estado comprobado:** existen tres semillas SAC entrenadas y una PPO s0 actual, pero sus monitores no registran C1, C2 ni C3. Los ficheros no muestran un barrido completo de SAC y PPO para cada configuración de entrenamiento, así que no se puede afirmar que el requisito experimental esté cumplido. Falta completar la matriz y registrar, junto a cada ejecución, el modo y las velocidades, la semilla, los pasos, los parámetros y la física o multiplicador.

---

## 3. Evaluación según el anexo B

### 3.1. Protocolo común (no modificar)

La referencia es la pelota verde real. En RoboboSim 4.3.0 se puede usar la pose del guía; en el escenario actual, el evaluador identifica el objeto guía `PATHFOLLOWROBOBO` y mide directamente `CYLINDERMIDBALL`, que es la pelota. El código de evaluación indica que está a unos 0,118 m del centro del guía, frente a los 0,08 m de la aproximación del enunciado. El seguidor es el robot 0.

La conexión directa a RoboboSim se usa solo en `p1_evaluar.py`, para el reinicio y la verdad del terreno, y no pasa por la observación ni por la recompensa. Así, el puente del seguidor sigue lanzándose únicamente con `modules:="['blob']"`.

En cada configuración: reinicio, seguidor parado, política aprendida en modo determinista y hasta 120 s de duración. El escenario coloca a ambos robots; no se cambia el protocolo ni se cuentan las salidas del carril como fallo.

| Config | Modo del guía | Vel. mínima | Vel. máxima |
|---|---|---|---|
| C1 | Crucero | 10 | 10 |
| C2 | Crucero | 10 | 40 |
| C3 | Variable | 10 | 40 |

**Métricas por paso**

- **Separación:** distancia entre el centro del seguidor y el centro de la pelota. Banda: 0,20–0,45 m.
- **Desviación:** ángulo entre el rumbo del seguidor y la dirección a la pelota.
- **Fallos:** colisión si la separación es < 0,12 m; guía perdido si la separación es > 0,90 m o si el blob falta durante más de 3 s consecutivos. Salirse del carril no es fallo.

Cada CSV de evaluación registra el instante, las posiciones del seguidor y de la pelota (y del guía), la separación, la desviación, la visibilidad del blob, la acción y el desenlace. El evaluador calcula las métricas con la verdad del terreno, mientras que la política solo recibe los blobs. Las poses llegan a unos 5 Hz (se han observado huecos de hasta 0,27 s), lo que puede provocar un desfase; hay que declararlo como limitación de medición.

**Las cuatro métricas requeridas, por configuración y en conjunto**

1. **Completados:** número y porcentaje.
2. **Desenlaces por causa:** completado, colisión, perdido por distancia, perdido por falta de vista; distinguir los CSV interrumpidos o sin terminar.
3. **Separación:** porcentaje de pasos dentro de la banda y separación media en metros.
4. **Desviación:** media de |ángulo| en grados y porcentaje de pasos con |desviación| < 15°.

El porcentaje en banda y las medias de un episodio fallido usan solo los pasos observados hasta el fallo. Por eso hay que leerlos siempre junto al desenlace y a los completados: un porcentaje alto en un episodio que dura pocos segundos no es un éxito.

### 3.2. Resultados disponibles en los CSV actuales (provisionales)

Estas cifras se recalcularon el 17/10/2026 con `python p1_graficas_eval.py --solo-tablas` sobre `p1_logs`. No sustituyen a la selección y evaluación final de cada política. Las etiquetas SAC s0/s1/s2 y PPO s0 corresponden a los ficheros actuales.

El CSV vigente contradice el resumen histórico que había en estas notas (por ejemplo, no respalda SAC s0 con 3/3 completados). Para la memoria hay que usar el CSV que se entregue y repetir las ejecuciones que se elijan como finales.

**Formato de cada celda:** completados/episodios; % en banda; separación media (m); media de |desv.| (°); % con |desv.| < 15°. Los desenlaces y la duración se anotan más abajo.

| Etiqueta | C1 | C2 | C3 |
|---|---|---|---|
| PPO s0 actual | 0/1; 42,9; 0,337; 10,6; 57,1 | 0/1; 75,5; 0,426; 11,7; 76,1 | 0/1; 91,3; 0,400; 24,1; 52,2 |
| SAC s0 actual | 0/1; 30,8; 0,289; 19,0; 46,2 | 0/1; 32,4; 0,523; 29,3; 52,9 | 0/1; 51,6; 0,482; 32,5; 51,6 |
| SAC s1 actual | 1/1; 99,3; 0,353; 3,6; 98,0 | 0/1; 68,1; 0,441; 14,7; 86,1 | 0/1; 78,1; 0,406; 12,2; 88,6 |
| SAC s2 actual | 0/1; 97,8; 0,341; 9,9; 87,2 | 1/1; 64,1; 0,436; 5,4; 99,7 | **PARCIAL:** 0/1; 70,7; 0,417; 13,7; 58,5 |
| Aleatoria | no registrado | no registrado | 0/1; 75,0; 0,413; 26,4; 40,0 |

**Totales calculados por el script** (no son una selección final):

| Etiqueta | Completados | % banda | Separación | \|Desv.\| | % < 15° |
|---|---|---|---|---|---|
| PPO s0 actual | 0/3 | 76,2 | 0,420 m | 13,2° | 72,5 |
| SAC s0 actual | 0/3 | 39,7 | 0,468 m | 28,8° | 51,3 |
| SAC s1 actual | 1/3 | 89,7 | 0,378 m | 7,2° | 94,0 |
| SAC s2 actual | 1/3 en los datos incluidos | 76,2 | 0,402 m | 7,6° | 92,1 |

En SAC s2, C3 está incompleto, así que ese total no debe usarse como evaluación cerrada.

**Desenlaces y tiempos exactos de los registros vigentes**

- **Aleatoria:** C3 perdido por vista a los 7,9 s (una sola ejecución, solo C3).
- **PPO s0:** C1 colisión a los 2,9 s; C2 perdido por vista a los 63,4 s; C3 perdido por vista a los 9,1 s.
- **SAC s0:** C1 perdido por vista a los 5,4 s; C2 perdido por distancia a los 13,4 s; C3 perdido por distancia a los 12,6 s.
- **SAC s1:** C1 completado a los 120,4 s; C2 perdido por distancia a los 28,7 s; C3 perdido por vista a los 45,4 s.
- **SAC s2:** C1 perdido por vista a los 71,8 s; C2 completado a los 120,2 s; C3 con CSV abierto y sin desenlace a los 16,5 s (incompleto).

La aleatoria solo tiene C3. Las evaluaciones de PPO s0 y SAC s0/s1/s2 están registradas, pero no cubren las tres configuraciones, y la de SAC s2 en C3 está interrumpida.

**Velocidad media real del guía** (calculada a partir de esos mismos CSV):

| Etiqueta | C1 | C2 | C3 |
|---|---|---|---|
| PPO s0 | 0,033 m/s | 0,137 m/s | 0,110 m/s |
| SAC s0 | 0,040 m/s | 0,162 m/s | 0,142 m/s |
| SAC s1 | 0,050 m/s | 0,131 m/s | 0,128 m/s |
| SAC s2 | 0,049 m/s | 0,151 m/s | 0,140 m/s (C3 parcial) |
| Aleatoria | — | — | 0,096 m/s (episodio corto) |

Estas cifras ayudan a comprobar la configuración del guía, pero una velocidad media no demuestra por sí sola que el modo y los límites fueran los correctos.

El evaluador también mide la correlación entre la desviación del terreno y el `posx` del blob. Los CSV actuales dan *r* entre −0,735 y −0,927 según la etiqueta. El signo negativo se debe al convenio de ejes; lo que interesa es |*r*| y su interpretación calibrada.

### 3.3. Evaluaciones finales pendientes

- Elegir una política final reproducible para cada configuración C1/C2/C3. No presentar un modelo como final solo porque tenga 3 semillas.
- Ejecutar el protocolo completo con la política determinista seleccionada: reinicio por configuración, velocidad y modo del guía verificados, mismo multiplicador y físicas que en el entrenamiento, máximo 120 s y registro en CSV.
- Repetir la línea base aleatoria necesaria para las curvas con la recompensa y la configuración actuales. La referencia antigua de −15,7 no sirve.
- Generar tablas y figuras a partir de los CSV finales y explicar cada resultado.
- Grabar los vídeos finales de C1, C2 y C3. El carril se valora en ellos, no como término de recompensa ni como fallo del protocolo.

**Comandos de postproceso**

```bash
python3 p1_graficas_eval.py

# Inspección sin ROS ni simulador
python3 p1_graficas_eval.py --solo-tablas
```

---

## 4. Conclusiones, riesgos y transferencia

### Hallazgos que se pueden sostener con los registros

- La cámara por sí sola deja una ambigüedad cerca del guía: la pelota puede salir del encuadre al acercarse y su `size` deja de crecer. Perderla por un lateral puede producir una observación parecida, y la política sin blob puede ponerse a girar buscando una pelota que en realidad está demasiado cerca.
- La recompensa basada en `size` no cubre todo el extremo cercano de la banda del anexo: el offset cámara-centro de 0,09 m y el clipping limitan la banda de entrenamiento observable.
- PPO s0 actual pierde la pelota en C2 y C3 y colisiona en C1, según los CSV actuales. SAC s0 actual también falla pronto, así que no se puede afirmar que sus ejecuciones actuales completen C1, C2 o C3.
- El PPO exploratorio con la banda antigua completó las tres ejecuciones, pero en C2 solo mantuvo un 34,2 % en banda, con una separación media de 0,544 m. Completar no implica mantener la separación deseada.
- Salirse del carril no lo mide la política ni es un fallo del anexo B. Hay que revisar en vídeo si seguir centrado en la pelota deja al seguidor en el carril derecho.

### Mejoras razonables (aún no implementadas ni entrenadas)

- Caracterizar y añadir el IR frontal como observación, para distinguir la proximidad peligrosa cuando el blob desaparece. Después habría que reentrenar y medirlo como una variante; ese resultado no puede atribuirse a los modelos actuales.
- Registrar automáticamente el modo, las velocidades, el multiplicador o las físicas y la semilla junto al Monitor. Ahora el guía se configura a mano y no existe un metadato persistente que pruebe las condiciones del entrenamiento.
- Repetir la caracterización en el robot real y actualizar el fichero de configuración, en lugar de fijar a mano los valores calibrados en el código.

### Transferencia al robot real (pendiente de ejecución)

Antes de ejecutar la política final hay que repetir `p1_calibrar.py` en el robot del laboratorio. Después hay que analizar las diferencias de `size`, `posx` y `posy` según la cámara, la iluminación, la presencia de otros elementos verdes, la latencia de ROS y de la red, la frecuencia efectiva y la dinámica lineal y angular.

El factor de respuesta medido en simulación (15,8 m en 120 s con una `v` ordenada media de 0,070 m/s, es decir, una velocidad media real de unos 0,131 m/s y una razón de aprox. 1,9) **no** es el multiplicador de velocidad del simulador y no debe usarse como si lo fuera. La política no se reentrena durante la transferencia.

---

## 5. Evidencias y estado de la entrega

**Código y configuración:** `p1_robobo_env.py`, `p1_calibrar.py`, `p1_entrenar.py`, `p1_evaluar.py`, `p1_graficas_eval.py`, `config/p1_config.json` y `config/calibracion.json`. La configuración activa incluye calibración, pero la configuración histórica de PPO que mencionan el README y CAMBIOS no está en `config/`.

**Modelos en `modelos/`:** `p1_modelo_sac_s0.zip`, `p1_modelo_sac_s1.zip`, `p1_modelo_sac_s2.zip` y `p1_modelo_ppo_s0.zip`. Hay que asociar a cada modelo su configuración exacta y sus parámetros para poder reproducirlo. El archivo que no se localiza no debe declararse como entregado.

**Registros**

- `p1_logs/monitor_*.csv`: uno por episodio de entrenamiento.
- `p1_logs/eval_*.csv`: uno por paso de evaluación.
- `p1_entrenar.py --graficas` y `p1_graficas_eval.py` regeneran figuras y tablas.
- Conviene conservar los CSV fuente que correspondan a los resultados que finalmente se incluyan, y no mezclar ejecuciones incompletas con las finales.

**Pendientes de entrega**

- [ ] Modelos y configuración reproducibles de la política final de cada C1/C2/C3.
- [ ] Registros finales de entrenamiento y evaluación, con curvas y tablas.
- [ ] Vídeos de la política final en C1, C2 y C3.
- [ ] Caracterización y prueba o análisis de transferencia en el robot real.
- [ ] Memoria PDF final (límite de 6 páginas más anexos, con el formato del enunciado).
- [ ] Declaración de uso de IA, aparte y de máximo una página, indicando qué hizo el equipo y qué asistencia se utilizó.

**Comandos de reproducción desde la raíz del proyecto**

```bash
ros2 run robobo_ros2 robobo_container --ros-args \
    -p ip:=host.docker.internal -p modules:="['blob']"
python3 p1_entrenar.py --comprobar
python3 p1_entrenar.py --aleatorio --episodios 8
python3 p1_entrenar.py --entrenar --algoritmo sac --pasos 20000 --semilla 0
python3 p1_entrenar.py --graficas --grupos sac_s --ventana 10
python3 p1_evaluar.py --algoritmo sac --semilla 0
python3 p1_graficas_eval.py
```
