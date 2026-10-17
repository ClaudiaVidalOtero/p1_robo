#!/usr/bin/env python3
"""
Práctica 1 — Entorno Gymnasium: el Robobo SEGUIDOR sigue al vehículo guía.

El guía es otro Robobo con una pelota verde en la parte trasera. El seguidor
sólo puede percibirlo a través del detector de blobs de color del smartphone,
que entrega, como mucho, UN blob verde por imagen. Toda la percepción y toda
la actuación del seguidor pasan por ROS 2 (nodo puente robobo_ros2).

El guía no reacciona al seguidor: si el seguidor se para, el guía continúa.
El guía lo conduce el simulador (o, en el robot real, el profesorado). Por eso
RESET sólo reinicia la escena y recoloca al seguidor; el guía vuelve a arrancar
desde donde lo deje el escenario y sigue circulando mientras el seguidor
aprende.

----------------------------------------------------------------------
REGLA DE ORO (no romper)
----------------------------------------------------------------------
El espacio de observación SÓLO puede contener información de los sensores del
seguidor recibida por ROS 2. En esta práctica eso es, exclusivamente, el blob
verde: su posición horizontal en la imagen y su tamaño. La pose del guía o del
seguidor, la separación real, el error lateral o la geometría del carril son
información privilegiada del simulador: se pueden usar para preparar el inicio
del episodio, para la recompensa, para terminar el episodio y para registrar
métricas, pero NUNCA para construir la observación. Lo que la política
necesita para decidir debe existir también en el robot real.

----------------------------------------------------------------------
OBSERVACIÓN (Box float32, dimensión 2 + 1 con OBS_Y + 1 con OBS_IR)
----------------------------------------------------------------------
    [0] posx = (x_blob - POSX_CENTRO) / POSX_CENTRO   en [-1, 1]
                (0 = pelota centrada; signo = a la derecha/izquierda)
    [1] size = size_blob / SIZE_MAX                    en [0, 1]

    Pelota no visible (o demasiado pequeña, por debajo de SIZE_MIN_VISTA):
    se devuelve [0.0, 0.0]. No se confunde con "pelota centrada y grande"
    porque en ese caso size > 0; el cero en size es la señal de "no la veo".

    Con OBS_Y = true (por defecto) se añade la altura de la pelota:
    [2] posy = y_blob / IMG_ALTO                       en [0, 1]
    Rompe la ambigüedad de size: a 0,20 m la pelota sale cortada por abajo y
    su size (~81) es casi igual al de 0,40 m (~88), pero su y es mucho mayor
    (está en el borde inferior de la imagen). Es un sensor del seguidor que
    también existe en el robot real.

    Variante opcional (OBS_IR = true, apartado 8): se añade un componente más
    ir = IR frontal máximo / IR_SATURACION   en [0, 1]
    Imprescindible para saber que el guía está pegado cuando el blob ya no se
    ve: por debajo de ~0,25 m el size deja de crecer y la pelota sale de cuadro.

----------------------------------------------------------------------
ACCIÓN (Box float32, dimensión 2, continua)
----------------------------------------------------------------------
    [0] velocidad lineal   v = V_MIN + (a0 + 1)/2 * (V_MAX - V_MIN)   m/s
    [1] velocidad angular  w = a1 * W_MAX                              rad/s

    Se actúa con el tópico base/cmd_vel (Twist). Elección justificada en la
    memoria: es la interfaz natural para un robot diferencial y el propio
    puente resuelve la cinemática. OJO: cmd_vel se detiene si no recibe
    órdenes en 0,5 s de tiempo REAL, así que el periodo de control debe ser
    menor que ese valor (aquí PASO_S = 0,2 s, igual que el periodo del blob).

----------------------------------------------------------------------
RECOMPENSA (sólo posx y size del blob; cada término con su coeficiente)
----------------------------------------------------------------------
    vista       +R_VISTA  si el blob es visible (ver siempre > perderlo)
    centrado    -W_CENT   * |posx|                    (mantener el guía al frente)
    separacion  +W_BANDA  si size en banda objetivo
                -K_FUERA  * e            si fuera de banda (e en [0, 1])
    giro        -K_GIRO   * |w| / W_MAX               (suavidad)
    d_giro      -K_DGIRO  * |w - w_prev| / W_MAX      (evita oscilar el volante)
    pérdida     -R_PERDIDO_PASO  cada paso sin ver el guía
    terminal    -R_CHOQUE  al chocar  y  -R_PERDIDO_FIN  al perder el guía

    Con los coeficientes por defecto el paso visible vale siempre MÁS que el
    perdido (peor visible: +0,5 -0,8 -0,5 = -0,8 > -1,0 = perdido). Así la
    política no tiene incentivo a "rendirse" perdiendo de vista al guía.

Los cuatro objetivos del enunciado se recogen así:
  1. separación en banda  -> término separacion (size ~ 1/separación²)
  2. no chocar            -> pelota cortada por abajo (y >= Y_CHOQUE), o que
                             desaparece tras verse cerca y centrada; término
                             terminal R_CHOQUE (SIZE_CHOQUE queda como respaldo,
                             pero la calibración muestra que no se alcanza)
  3. no perder de vista   -> término pérdida y centrado
  4. no salirse del carril -> término centrado: seguir al guía centrado es lo
     único que arrastra al seguidor por su mismo carril (no hay sensor de
     carril; se documenta esta limitación).

----------------------------------------------------------------------
REQUISITOS
----------------------------------------------------------------------
1. RoboboSim en ejecución con el escenario de la ciudad y el guía circulando.
2. Nodo puente en otra terminal:
       ros2 run robobo_ros2 robobo_container --ros-args \
            -p ip:=host.docker.internal -p modules:="['blob']"
   (el módulo de cámara no hace falta para blobs; se cargan sólo los necesarios.
   Con RESET_DIRECTO=false el reinicio va por el puente y hace falta además
   el módulo 'sim': modules:="['blob','sim']")
3. Configuración calibrada en config/p1_config.json. Se regenera con
       python3 -c "from p1_robobo_env import importar_calibracion as i; i()"
   a partir de config/calibracion.json (medidas de p1_calibrar.py).

Sin módulo sim (robot real, o pruebas sin simulador) el entorno funciona con
usar_simulador=False: entonces el reposicionamiento entre episodios lo hace
una persona (pulsa Enter).

----------------------------------------------------------------------
ESTE ARCHIVO
----------------------------------------------------------------------
Contiene `RoboboSeguidorEnv` y utilidades de configuración/calibración. El
entrenamiento y la evaluación están en p1_entrenar.py. Las constantes que
dependen de la calibración se leen de config/p1_config.json, no están en el código.
"""

import contextlib
import json
import math
import os
import statistics
import time
from typing import ClassVar

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from gymnasium import spaces
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


try:
    from robobo_ros2_interfaces.msg import BlobArray
    from robobo_ros2_interfaces.srv import MoveTilt, SetActiveColorBlobs
    HAY_INTERFACES_BLOB = True
except ImportError:
    BlobArray = SetActiveColorBlobs = MoveTilt = None
    HAY_INTERFACES_BLOB = False


try:
    from robobo_ros2_interfaces.msg import RobotLocation
    from robobo_ros2_interfaces.srv import ResetSimulation
    HAY_INTERFACES_SIM = True
except ImportError:
    RobotLocation = ResetSimulation = None
    HAY_INTERFACES_SIM = False

try:
    from robobo_ros2_interfaces.srv import SetRobotLocation
except ImportError:
    SetRobotLocation = None

AVISO_SIN_BLOB = (
    'No se encuentran las interfaces de blobs '
    '(robobo_ros2_interfaces.msg.BlobArray).\n'
    'Cargar el espacio de trabajo del puente:\n'
    '      source /opt/ros/robobo/setup.bash')
AVISO_SIN_SIM = (
    'No se encuentran las interfaces del módulo sim '
    '(robobo_ros2_interfaces.msg.RobotLocation).\n'
    'Cargar el espacio de trabajo del puente o crear el entorno con\n'
    'usar_simulador=False (--sin-simulador) y reponer el robot a mano.')



NS_BASE = '/robobo/robot_0/base'
NS_SIM = '/robobo/robot_0/sim'
NS_SMARTPHONE = '/robobo/robot_0/smartphone'

COLOR_BLOB = 'green'
ESPERA_MAX_S = 3.0

# Pulsos de giro del barrido de posx de p1_calibrar.py.
W_GIRO, GIRO_S = 0.3, 0.3

RUTA_CONFIG = os.path.join('config', 'p1_config.json')
RUTA_CALIBRACION = os.path.join('config', 'calibracion.json')
RUTA_MODELOS = 'modelos'


# =====================================================================
# Configuración 
# =====================================================================

# Valores de reserva. Sirven para arrancar sin p1_config.json y como base
# sobre la que importar_calibracion() escribe las constantes medidas.
CONFIG_POR_DEFECTO = {
    'NS': '/robobo/robot_0',
    'COLOR_BLOB': COLOR_BLOB,

    'IMG_ANCHO': 100.0,        # rango de x del blob (px); centro en la mitad
    'POSX_CENTRO': 50.0,

    'SIZE_MAX': 330.0,         # normalización del tamaño del blob

    'BANDA_PROTOCOLO_M': [0.20, 0.45],  # Banda objetivo del anexo B: separación CENTRO del seguidor - PELOTA.
    'CAMARA_OFFSET_M': 0.09,   # Offset del centro del seguidor a la cámara (0,09 m).
    'SIZE_BANDA_MIN': 100.0,   # size a 0,45 m del centro (0,36 m de la cámara)
    'SIZE_BANDA_MAX': 207.0,   # size a 0,25 m de la cámara: por debajo la
                               # pelota sale cortada y el size deja de crecer
    'SIZE_OBJ': 144.0,         # size a 0,30 m de la cámara (0,39 m del centro)
    'SIZE_CHOQUE': 323.4,      # size equivalente a ~0,20 m (colisión)
    'SIZE_MIN_VISTA': 12.9,    # por debajo se considera "no fiable" (~1,0 m)
    'SIZE_CERCA': 144.0,       # ~0,30 m: si la pelota desaparece tras verse a
                               # menos de esto, se interpreta como "se acercó
                               # de más" (la pelota sale del encuadre por abajo)
    'POSX_CERCA_MAX': 0.6,     # ...pero sólo si estaba centrada (|posx| menor
                               # que esto); si estaba en un borde, ha salido
                               # por el lateral al girar y no es un choque

    'V_MIN': 0.0,              # m/s (sin marcha atrás en la versión base)
    'V_MAX': 0.15,             # m/s (debe SUPERAR la velocidad del guía)
    'W_MAX': 0.5,              # rad/s (limitado: 0,2 s * 0,5 = ~6°/paso)

    'PASO_S': 0.2,             # periodo de control = periodo del blob (5 Hz)
    'TIMEOUT_LECTURA_S': 2.0,  # espera máxima por un blob nuevo
    'PERDIDO_MAX': 12,         # pasos consecutivos sin ver el guía -> perdido
    'PASOS_MAX': 300,          # 60 s / 0,2 s (protocolo del anexo B)

    # RoboboSim corta la conexión (Broken pipe) si recibe reinicios de escena
    # encadenados. Se deja un intervalo mínimo entre reinicios y una espera
    # tras cada uno para que la escena se estabilice.
    'RESET_ESPERA_S': 1.5,         # espera tras sim/reset_simulation
    'RESET_INTERVALO_MIN_S': 4.0,  # tiempo mínimo entre dos reinicios
    # La conexión permanente del módulo 'sim' del puente se cae al cabo de
    # unos episodios (RoboboSim corta el socket) y la librería no reconecta.
    # Con RESET_DIRECTO el entorno abre una conexión nueva y corta con
    # RoboboSim para cada reinicio (sólo gestión del episodio: sensores y
    # actuadores siguen yendo únicamente por ROS 2). El puente se lanza
    # entonces sólo con modules:="['blob']".
    'RESET_DIRECTO': True,
    'SIM_IP': 'host.docker.internal',

    # La calibración se hizo con la cámara horizontal (tilt 90°). Hay que
    # fijarlo al inicio del entrenamiento o la relación size<->distancia no
    # vale y el seguidor se acerca de más hasta que la pelota sale del encuadre.
    'TILT': 90.0,              # grados
    'TILT_VEL': 30.0,          # velocidad del servo al colocar el tilt

    'USAR_IR_CHOQUE': False,   # (arriesgado fuera del carril: paredes cercanas)
    'IR_SATURACION': 1000.0,
    'IR_CHOQUE': 0.8,
    # Añade el IR frontal máximo como 3er componente de la observación. Sirve
    # para saber que el guía está ENCIMA cuando el blob ya no se ve (por debajo
    # de ~0,25 m el size deja de crecer). Por defecto off: la observación base
    # es [posx, size] y así se entrega; activarlo es una variante del apartado 8.
    'OBS_IR': False,

    # Altura de la pelota (blob.y). Con la cámara a 90° la pelota sale
    # cortada por abajo a ~0,20 m y su size cae a ~81 (lo mismo que a 0,40 m):
    # sólo la y distingue "pegado" de "lejos".
    'IMG_ALTO': 100.0,         # rango de y del blob (px)
    'OBS_Y': True,             # añade posy a la observación
    'Y_CHOQUE': None,          # CAL: y a partir de la cual la pelota está
                               # cortada (guía encima) -> choque. None = sin
                               # calibrar (no se aplica). Lo calcula
                               # importar_calibracion() con medidas de y.

    'POS_INICIO': None,        # lista de [x_mm, z_mm, yaw_deg] o None

    'W_CENT': 0.5,
    'W_BANDA': 1.0,
    'K_FUERA': 0.8,
    'K_GIRO': 0.3,
    'K_DGIRO': 0.3,            # penaliza el CAMBIO de giro (evita las "eses")
    'R_VISTA': 0.5,            # ver el blob siempre suma: NUNCA compensa perderlo
    'R_PERDIDO_PASO': 1.0,
    'R_PERDIDO_FIN': 5.0,
    'R_CHOQUE': 20.0,          # > 12*R_PERDIDO_PASO + R_PERDIDO_FIN (=17):
                               # chocar debe salir PEOR que perder al guía
}


def cargar_config(ruta=RUTA_CONFIG):
    """Lee p1_config.json y lo completa con los valores de reserva.

    Se usa una copia de CONFIG_POR_DEFECTO para no mutar la constante global.
    Si el archivo no existe, se devuelven sólo los valores por defecto.
    """
    c = dict(CONFIG_POR_DEFECTO)
    if ruta and os.path.exists(ruta):
        with open(ruta) as f:
            c.update(json.load(f))
    return c


def importar_calibracion(json_cal=RUTA_CALIBRACION, ruta_config=RUTA_CONFIG,
                         guardar=True):
    """Deriva las constantes del entorno de calibracion.json (apartado 5.1).

    calibracion.json lo produce p1_calibrar.py: una lista de medidas
    {distancia, lecturas:[{etapa, size, posx}], ...}. De ahí se obtiene:

      - K de la ley de área  size ≈ K / distancia²  (mediana de d²·size para
        d >= 0,25 m; por debajo de 0,25 m la pelota sale del encuadre y el
        size deja de crecer, así que esos puntos no se usan para el ajuste),
      - SIZE_MAX y los tamaños en las separaciones clave (0,25 / 0,375 / 0,50
        m) y a la distancia de colisión (0,20 m) del anexo B,
      - IMG_ANCHO y POSX_CENTRO a partir de las x observadas.

    Escribe el resultado en p1_config.json, conservando el resto de claves.
    Devuelve el diccionario de configuración actualizado.
    """
    with open(json_cal) as f:
        datos = json.load(f)
    if isinstance(datos, dict):
        datos = [datos]

    puntos, xs, ys = [], [], {}
    for e in datos:
        d = float(e['distancia'])
        lec = e.get('lecturas', [])
        centros = [r['size'] for r in lec
                   if r.get('etapa') in ('centro', 'reposo')
                   and r.get('size', 0) > 0]
        # y de la pelota centrada (sólo en medidas nuevas, que la guardan)
        y_d = [r['y'] for r in lec
               if r.get('etapa') in ('centro', 'reposo')
               and r.get('size', 0) > 0 and r.get('y') is not None]
        if y_d:
            ys.setdefault(d, []).extend(y_d)
        todos = [r['size'] for r in lec if r.get('size', 0) > 0]
        tam = (statistics.median(centros) if centros
               else (statistics.median(todos) if todos else 0.0))
        puntos.append((d, tam))
        xs.extend(r['posx'] for r in lec
                  if r.get('size', 0) > 0 and r.get('posx') is not None)
    puntos.sort()

    ks = [d * d * s for d, s in puntos if d >= 0.25 and s > 0]
    K = statistics.median(ks) if ks else 0.0

    def size(d):
        return K / (d * d) if d > 0 else 0.0

    c = cargar_config(ruta_config)
    medida_max = max((s for _, s in puntos), default=0.0)
    c['SIZE_MAX'] = float(int(math.ceil(max(medida_max, size(0.20)) / 10.0) * 10))
    # Banda del protocolo (centro-pelota) pasada a distancia de la cámara.
    # El borde cercano no puede bajar de 0,25 m de la cámara: más cerca la
    # pelota sale cortada y el size ya no crece (ver calibración).
    sep_min, sep_max = c['BANDA_PROTOCOLO_M']
    cal_lejos = sep_max - c['CAMARA_OFFSET_M']
    cal_cerca = max(sep_min - c['CAMARA_OFFSET_M'], 0.25)
    c['SIZE_BANDA_MIN'] = round(size(cal_lejos), 1)
    c['SIZE_BANDA_MAX'] = round(size(cal_cerca), 1)
    c['SIZE_OBJ'] = round(size((cal_lejos + cal_cerca) / 2.0), 1)
    c['SIZE_CHOQUE'] = round(min(size(0.20), c['SIZE_MAX']), 1)
    c['SIZE_MIN_VISTA'] = round(size(1.0), 1)
    if xs:
        c['IMG_ANCHO'] = float(int(math.ceil(max(xs) / 10.0) * 10))
        c['POSX_CENTRO'] = c['IMG_ANCHO'] / 2.0

    # Y_CHOQUE: punto medio entre la y a 0,20 m (pelota cortada, choque) y la
    # y a 0,25 m (borde cercano de la banda). Hacen falta medidas con y a
    # ambas distancias; si no, se deja como estaba.
    y_med = {d: statistics.median(v) for d, v in ys.items()}
    y20 = next((v for d, v in y_med.items() if abs(d - 0.20) < 1e-6), None)
    y25 = next((v for d, v in y_med.items() if abs(d - 0.25) < 1e-6), None)
    for d in sorted(y_med):
        print(f'  y mediana a {d:.3f} m = {y_med[d]:.1f}')
    if y20 is not None and y25 is not None and y20 > y25:
        c['Y_CHOQUE'] = round((y20 + y25) / 2.0, 1)
        print(f'  Y_CHOQUE = {c["Y_CHOQUE"]:.1f}  (entre y(0,25)={y25:.1f} '
              f'e y(0,20)={y20:.1f})')
    else:
        print('  AVISO: faltan medidas de y a 0,20 y 0,25 m (o no separan): '
              f'Y_CHOQUE se queda en {c.get("Y_CHOQUE")}.')

    print(f'Ajuste size = K / d²   con K = {K:.3f}')
    print(f'  size(0,20 m) = {size(0.20):7.1f}   size(0,25 m) = {size(0.25):7.1f}'
          )
    print(f'  size(0,375 m)= {size(0.375):7.1f}   size(0,50 m) = {size(0.50):7.1f}'
          )
    print('  IMG_ANCHO = {:.0f}   POSX_CENTRO = {:.0f}'
          .format(c['IMG_ANCHO'], c['POSX_CENTRO']))
    print('  SIZE_MAX = {:.0f}   banda = [{:.1f}, {:.1f}]   choque = {:.1f}'
          .format(c['SIZE_MAX'], c['SIZE_BANDA_MIN'],
                  c['SIZE_BANDA_MAX'], c['SIZE_CHOQUE']))

    if guardar:
        limite = dict(c)
        with open(ruta_config, 'w') as f:
            json.dump(limite, f, indent=2, ensure_ascii=False)
        print(f'Configuración escrita en {ruta_config}')
    return c


def dif_ang(a, b):
    """Diferencia angular a-b en (-180, 180]. La usa p1_calibrar.py."""
    return (a - b + 180.0) % 360.0 - 180.0


# =====================================================================
# El entorno
# =====================================================================

class RoboboSeguidorEnv(gym.Env):
    """El Robobo seguidor aprende a circular detrás del guía (pelota verde).

    Parámetros
        pasos_max       longitud máxima del episodio (truncado). 300 = 60 s.
        usar_simulador  True en simulación (módulo sim); False en el robot real.
        verbose         imprime una línea por paso.
        ruta_config     archivo de configuración calibrada.

    La observación es la de la cabecera (2 a 4 componentes). La acción es continua
    (dimensión 2). La recompensa y las condiciones de terminación se explican
    arriba y en _recompensa().
    """

    metadata: ClassVar[dict] = {'render_modes': []}

    def __init__(self, pasos_max=None, usar_simulador=True, verbose=False,
                 ruta_config=RUTA_CONFIG):
        super().__init__()

        if not HAY_INTERFACES_BLOB:
            raise RuntimeError(AVISO_SIN_BLOB)

        self.c = cargar_config(ruta_config)
        self.pasos_max = int(pasos_max if pasos_max is not None
                             else self.c['PASOS_MAX'])
        self.usar_simulador = usar_simulador
        self.verbose = verbose

        # -------------------------------------------------- ROS 2
        # rclpy.init() sólo puede llamarse una vez por proceso: la guarda
        # permite crear el entorno varias veces (entrenar y luego evaluar).
        self._rclpy_propio = not rclpy.ok()
        if self._rclpy_propio:
            rclpy.init()
        self.nodo = Node('p1_seguidor_env')
        ns = self.c['NS']

        # Publicador y suscripciones creados UNA sola vez. Crearlos dentro de
        # step() perdería las primeras órdenes (el descubrimiento de ROS 2
        # tarda cientos de milisegundos).
        self.pub_vel = self.nodo.create_publisher(Twist, ns + '/base/cmd_vel', 10)

        self._blob = None      # último blob verde: (x, y, size) o None
        self._n_msgs = 0       # número de mensajes de blobs recibidos
        self.nodo.create_subscription(
            BlobArray, ns + '/smartphone/color_blobs', self._cb_blobs, 1)

        # Infrarrojos frontales: sólo para una terminación por colisión
        # opcional (USAR_IR_CHOQUE). NO entran en la observación.
        self._ir = None
        self.nodo.create_subscription(
            Int32MultiArray, ns + '/base/ir', self._cb_ir, 1)

        self.cli_blobs = self.nodo.create_client(
            SetActiveColorBlobs, ns + '/smartphone/set_active_color_blobs')

        # Inclinación de la cámara (tilt). Se fija a 90° al inicio: la
        # calibración size<->distancia se hizo con la cámara horizontal y sin
        # esto el seguidor se acerca de más y pierde la pelota por abajo.
        self.cli_tilt = None
        if MoveTilt is not None:
            self.cli_tilt = self.nodo.create_client(
                MoveTilt, ns + '/base/move_tilt')

        # -------------------------------------------------- módulo sim
        # Canal privilegiado (sólo simulación): reinicio de la escena,
        # colocación del seguidor y pose real para registrar métricas. Nada
        # de esto entra en la observación.
        self.cli_reset = None
        self.cli_colocar = None
        self._t_ultimo_reset = None   # instante del último reset_simulation
        self._pose = None
        if usar_simulador and self.c['RESET_DIRECTO']:
            # El reinicio va por conexión propia (_reset_directo); el puente
            # no necesita el módulo sim.
            pass
        elif usar_simulador:
            if not HAY_INTERFACES_SIM:
                raise RuntimeError(AVISO_SIN_SIM)
            self.cli_reset = self.nodo.create_client(
                ResetSimulation, ns + '/sim/reset_simulation')
            if not self.cli_reset.wait_for_service(timeout_sec=5.0):
                raise RuntimeError(
                    f'No responde {ns}/sim/reset_simulation.\n'
                    "¿robobo_container lanzado con el módulo 'sim'? "
                    'Alternativa: --sin-simulador.')
            self.nodo.create_subscription(
                RobotLocation, ns + '/sim/robot_location', self._cb_pose, 1)
            if SetRobotLocation is not None:
                self.cli_colocar = self.nodo.create_client(
                    SetRobotLocation, ns + '/sim/set_robot_location')

        # -------------------------------------------------- espacios
        # [posx, size] + [posy] si OBS_Y + [ir] si OBS_IR
        bajo, alto = [-1.0, 0.0], [1.0, 1.0]
        if self.c.get('OBS_Y'):
            bajo.append(0.0)
            alto.append(1.0)
        if self.c.get('OBS_IR'):
            bajo.append(0.0)
            alto.append(1.0)
        self.observation_space = spaces.Box(
            low=np.array(bajo, dtype=np.float32),
            high=np.array(alto, dtype=np.float32),
            dtype=np.float32)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # -------------------------------------------------- estado
        self._pasos = 0
        self._perdidos = 0
        self._ultimo_size = 0.0
        self._ultimo_posx = 0.0
        self._a_prev = np.zeros(2, dtype=np.float32)
        self._v = 0.0
        self._w = 0.0

        self._colocar_tilt()    # 90°: igual que en la calibración
        self._activar_verde()
        # Esperar el primer mensaje confirma que el puente está en marcha
        # antes de que el algoritmo empiece a entrenar.
        self._esperar_lectura_nueva()

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    def _cb_blobs(self, msg):
        """Guarda el blob verde de mayor tamaño (o None).

        El detector publica también listas vacías, así que cada mensaje cuenta
        como lectura nueva. Es un solo blob verde por imagen, pero en la ciudad
        hay otros elementos verdes: se toma el mayor y se filtra por tamaño
        mínimo en _observacion().
        """
        mejor = None
        for b in msg.blobs:
            if b.color == self.c['COLOR_BLOB'] and (mejor is None or b.size > mejor[2]):
                mejor = (float(b.x), float(b.y), float(b.size))
        self._blob = mejor
        self._n_msgs += 1

    def _cb_ir(self, msg):
        self._ir = list(msg.data)

    def _spin(self, timeout=0.0):
        rclpy.spin_once(self.nodo, timeout_sec=timeout)

    def _vaciar_cola(self):
        """Descarta lo recibido mientras el robot se movía.

        Lo que hay en la cola se capturó antes de que la acción acabara de
        surtir efecto, así que no sirve como observación del paso.
        """
        for _ in range(100):
            antes = self._n_msgs
            self._spin(0.0)
            if self._n_msgs == antes:
                return

    def _esperar_lectura_nueva(self, inicial=False):
        """Bloquea hasta recibir un mensaje POSTERIOR a esta llamada.

        Éste es el punto delicado de acoplar Gymnasium y ROS 2. Gymnasium
        exige que step() devuelva la observación RESULTANTE de la acción; ROS 2
        entrega los mensajes cuando el ejecutor los procesa. Si se devuelve la
        última lectura guardada sin más, se devuelve una observación ANTERIOR a
        la acción y el entrenamiento no aprende (sin ningún mensaje de error).
        Aquí se espera un contador de mensajes estrictamente mayor.
        """
        objetivo = self._n_msgs + 1
        t0 = time.time()
        while self._n_msgs < objetivo:
            self._spin(0.05)
            if time.time() - t0 > self.c['TIMEOUT_LECTURA_S']:
                raise RuntimeError(
                    'No llegan mensajes de {}/smartphone/color_blobs.\n'
                    "¿robobo_container con modules:=\"['blob']\"?"
                    .format(self.c['NS']) if inicial else
                    'Se ha interrumpido el flujo de blobs durante el episodio.')

    def _publicar_velocidad(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub_vel.publish(m)

    def _detener(self):
        """Detiene las ruedas y espera a que la orden surta efecto."""
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _llamar(self, cliente, peticion, espera=10.0):
        if cliente is None:
            return None
        if not cliente.wait_for_service(timeout_sec=espera):
            raise RuntimeError('Servicio no disponible: ' + cliente.srv_name)
        futuro = cliente.call_async(peticion)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=espera)
        return futuro.result()

    def _activar_verde(self):
        """Activa la detección del color verde (única vía de percepción)."""
        req = SetActiveColorBlobs.Request()
        req.red, req.green, req.blue, req.custom = False, True, False, False
        self._llamar(self.cli_blobs, req)

    def _colocar_tilt(self):
        """Fija la inclinación de la cámara (servicio base/move_tilt).

        Debe coincidir con el tilt usado en la calibración (90° por defecto).
        Si no se ajusta, la relación size<->distancia medida no se cumple en el
        entrenamiento y el seguidor se acerca de más hasta perder la pelota.
        """
        if self.cli_tilt is None:
            return
        req = MoveTilt.Request()
        try:
            req.angle = float(self.c['TILT'])
            req.speed = float(self.c['TILT_VEL'])
        except AttributeError:
            return
        try:
            self._llamar(self.cli_tilt, req)
            time.sleep(0.5)   # dar tiempo a que el servo llegue a la posición
        except RuntimeError as exc:
            print(f'Aviso: no se pudo fijar el tilt: {exc}')

    # -----------------------------------------------------------------
    # Canal privilegiado del simulador (sólo métricas y preparación)
    # -----------------------------------------------------------------

    def _cb_pose(self, msg):
        """Pose real del seguidor (mm y grados). Sólo para registrar."""
        self._pose = (msg.position.x, msg.position.z, msg.rotation.y)

    def _reiniciar_escena(self):
        # Respetar un intervalo mínimo desde el reinicio anterior: si los
        # episodios acaban en 1-2 pasos, los reinicios llegan encadenados y
        # RoboboSim acaba cerrando la conexión.
        if self._t_ultimo_reset is not None:
            falta = (self.c['RESET_INTERVALO_MIN_S']
                     - (time.time() - self._t_ultimo_reset))
            if falta > 0:
                time.sleep(falta)
        if self.c['RESET_DIRECTO']:
            self._reset_directo()
            self._t_ultimo_reset = time.time()
            time.sleep(self.c['RESET_ESPERA_S'])
            return
        futuro = self.cli_reset.call_async(ResetSimulation.Request())
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=5.0)
        self._t_ultimo_reset = time.time()
        if futuro.result() is None or not futuro.result().success:
            raise RuntimeError(
                'El reinicio de la escena ha fallado. Si el puente muestra '
                '"Broken pipe", RoboboSim ha cerrado la conexión: relanza '
                'robobo_container y sube RESET_INTERVALO_MIN_S en la config.')
        time.sleep(self.c['RESET_ESPERA_S'])

    def _reset_directo(self, intentos=3):
        """Reinicia la escena con una conexión nueva y corta a RoboboSim.

        La conexión permanente del puente se cae tras unos episodios y su
        librería no reconecta (la siguiente llamada hace sys.exit). Abrir una
        conexión por reinicio hace que un corte no afecte al siguiente.
        """
        import websocket
        from robobosim.RoboboSim import RoboboSim
        from robobosim.utils.ConnectionState import ConnectionState

        ultimo_error = None
        for _ in range(intentos):
            sim = RoboboSim(self.c['SIM_IP'])
            try:
                sim.connect()
                if sim.rem.connectionState != ConnectionState.CONNECTED:
                    raise ConnectionError('RoboboSim no acepta la conexión')
                time.sleep(0.3)   # que procese la contraseña antes del reset
                sim.resetSimulation()
                time.sleep(0.2)   # que el mensaje salga antes de cerrar
                return
            # sendMessage() de robobosim hace sys.exit si no hay conexión.
            except (OSError, SystemExit, websocket.WebSocketException) as exc:
                ultimo_error = exc
                time.sleep(1.0)
            finally:
                with contextlib.suppress(OSError, websocket.WebSocketException):
                    sim.disconnect()
        raise RuntimeError(
            f'No se pudo reiniciar RoboboSim en {self.c["SIM_IP"]}:50505 '
            f'tras {intentos} intentos ({ultimo_error}). ¿Está abierto?')

    def _colocar_seguidor(self):
        """Coloca el seguidor en una pose de arranque (opcional).

        Aleatorizar el punto de partida evita que la política memorice una
        secuencia fija. Las poses están en milímetros y grados (plano x, z).
        Es preparación del episodio, permitida con el módulo sim.
        """
        if self.cli_colocar is None or not self.c.get('POS_INICIO'):
            return
        posiciones = self.c['POS_INICIO']
        idx = int(self.np_random.integers(len(posiciones)))
        x, z, yaw = posiciones[idx]
        req = SetRobotLocation.Request()
        try:
            req.position.x, req.position.y, req.position.z = float(x), 0.0, float(z)
            req.rotation.x, req.rotation.y, req.rotation.z = 0.0, float(yaw), 0.0
        except AttributeError:
            return
        try:
            self._llamar(self.cli_colocar, req)
        except RuntimeError:
            pass

    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------

    def _size_visible(self):
        """Tamaño del blob si es fiable, o None si no se ve / es ruido."""
        if self._blob is None:
            return None
        size = self._blob[2]
        if size < self.c['SIZE_MIN_VISTA']:
            return None
        return size

    def _pelota_cortada(self):
        """True si la pelota se ve cortada por el borde inferior.

        Indica que el guía está encima (~0,20 m): el size ya no sirve porque
        cae al cortarse la pelota. Sólo se aplica si Y_CHOQUE está calibrado.
        """
        y_choque = self.c.get('Y_CHOQUE')
        return bool(y_choque is not None and self._size_visible() is not None
                    and self._blob[1] >= y_choque)

    def _ir_frontal(self):
        """IR frontal máximo normalizado en [0, 1] (0 si no hay lectura)."""
        if not self._ir:
            return 0.0
        ir_max = max(max(v_, 0) for v_ in self._ir[:5])
        return float(np.clip(ir_max / self.c['IR_SATURACION'], 0.0, 1.0))

    def _observacion(self):
        size = self._size_visible()
        if size is None:
            # No se ve la pelota. Con OBS_IR el IR frontal sí se incluye: es la
            # única señal de que el guía puede estar pegado (blob fuera de cuadro).
            base = [0.0, 0.0] + ([0.0] if self.c.get('OBS_Y') else [])
        else:
            x = self._blob[0]
            posx = float(np.clip((x - self.c['POSX_CENTRO'])
                                 / self.c['POSX_CENTRO'], -1.0, 1.0))
            base = [posx, float(np.clip(size / self.c['SIZE_MAX'], 0.0, 1.0))]
            if self.c.get('OBS_Y'):
                base.append(float(np.clip(self._blob[1] / self.c['IMG_ALTO'],
                                          0.0, 1.0)))
        if self.c.get('OBS_IR'):
            base.append(self._ir_frontal())
        return np.array(base, dtype=np.float32)

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

    def _excursion_banda(self, size):
        """Distancia normalizada fuera de la banda objetivo, en [0, 1].

        0 dentro de la banda; 1 en el borde extremo de lo medible. Da pendiente
        para que la política sepa hacia dónde corregir.
        """
        lo, hi = self.c['SIZE_BANDA_MIN'], self.c['SIZE_BANDA_MAX']
        if size < lo:
            e = (lo - size) / lo
        elif size > hi:
            e = (size - hi) / max(self.c['SIZE_MAX'] - hi, 1e-6)
        else:
            return 0.0
        return float(np.clip(e, 0.0, 1.0))

    def _recompensa(self, obs, a, a_prev):
        """Recompensa del paso, sólo con posx y size del blob.

        Devuelve (r, términos) para poder registrar cada término por separado.
        a_prev es la acción del paso anterior: su cambio en el giro se penaliza
        (K_DGIRO) para romper las oscilaciones de volante ("eses").
        """
        c = self.c
        posx = float(obs[0])
        size = self._size_visible()

        if size is None:
            # No se ve el guía: no se puede medir nada, sólo se penaliza.
            # OJO: la clave se llama 'r_perdido' y no 'perdido' para no pisar
            # el booleano 'perdido' del info (son cosas distintas).
            return -c['R_PERDIDO_PASO'], {'r_perdido': -c['R_PERDIDO_PASO']}

        # Pelota cortada = pegado al guía: su size cae dentro de la banda,
        # pero NO es una separación correcta.
        en_banda = (c['SIZE_BANDA_MIN'] <= size <= c['SIZE_BANDA_MAX']
                    and not self._pelota_cortada())
        terminos = {
            'vista': c['R_VISTA'],
            'centrado': -c['W_CENT'] * abs(posx),
            'separacion': (c['W_BANDA'] if en_banda
                           else -c['K_FUERA'] * self._excursion_banda(size)),
            'giro': -c['K_GIRO'] * abs(float(a[1])),
            'd_giro': -c['K_DGIRO'] * abs(float(a[1]) - float(a_prev[1])),
        }
        return float(sum(terminos.values())), terminos

    # -----------------------------------------------------------------
    # Interfaz de Gymnasium
    # -----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)   # fija self.np_random

        self._detener()

        if self.usar_simulador:
            # La escena vuelve a su estado inicial (el guía rearranca y
            # circula); después se puede recolocar el seguidor.
            self._reiniciar_escena()   # incluye la espera RESET_ESPERA_S
            self._colocar_seguidor()
            self._colocar_tilt()    # el reinicio puede devolver el tilt a 0
            self._activar_verde()   # el reinicio puede desactivar los blobs
            time.sleep(0.3)
        else:
            # Sin simulador (robot real): el reinicio lo hace una persona.
            input('Coloca el seguidor detrás del guía y pulsa Enter... ')

        self._pasos = 0
        self._perdidos = 0
        self._ultimo_size = 0.0
        self._ultimo_posx = 0.0
        self._a_prev = np.zeros(2, dtype=np.float32)
        self._v = self._w = 0.0

        # Descartar lo acumulado durante el reposicionamiento y esperar una
        # lectura tomada ya en la posición nueva.
        self._vaciar_cola()
        self._esperar_lectura_nueva()
        return self._observacion(), {}

    def step(self, accion):
        c = self.c
        a = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)

        # 1. Reescalar la acción normalizada a unidades del SI y publicarla.
        v = c['V_MIN'] + (float(a[0]) + 1.0) / 2.0 * (c['V_MAX'] - c['V_MIN'])
        w = float(a[1]) * c['W_MAX']
        self._v, self._w = v, w
        self._publicar_velocidad(v, w)

        # 2. Dejar que la acción surta efecto, sin procesar mensajes: los que
        #    lleguen ahora se descartan en el paso siguiente.
        time.sleep(c['PASO_S'])
        # Refresco por si PASO_S fuese mayor que el timeout de cmd_vel (0,5 s).
        self._publicar_velocidad(v, w)

        # 3. Vaciar la cola y  4. esperar una lectura POSTERIOR a la acción.
        self._vaciar_cola()
        self._esperar_lectura_nueva()

        # 5. Evaluar el resultado.
        self._pasos += 1
        obs = self._observacion()
        r, terminos = self._recompensa(obs, a, self._a_prev)
        self._a_prev = a

        size = self._size_visible()
        visible = size is not None
        self._perdidos = self._perdidos + 1 if not visible else 0

        # La pelota sale del encuadre por abajo cuando el guía está ENCIMA: si
        # desaparece justo después de haberse visto cerca, no es que se haya
        # perdido de vista, es que el seguidor se ha acercado demasiado. Se
        # cuenta como colisión para que la política aprenda a no embestir
        # (el size deja de crecer por debajo de ~0,25 m, así que no se puede
        # detectar el exceso de cercanía sólo con su valor actual).
        # Sólo si estaba centrada: cerca de un borde de la imagen lo probable
        # es que haya salido por el LATERAL al girar, y eso no es un choque
        # (cuenta para "guía perdido" si no vuelve a verse).
        demasiado_cerca = bool(not visible and
                               self._ultimo_size >= c['SIZE_CERCA'] and
                               abs(self._ultimo_posx) < c['POSX_CERCA_MAX'])

        # Colisión: pelota cortada por abajo (y), guía demasiado cerca (size
        # grande; respaldo, en la práctica no se alcanza), pérdida por
        # cercanía o, opcionalmente, proximidad frontal de los infrarrojos.
        cortada = self._pelota_cortada()
        choque = bool(cortada
                      or (size is not None and size >= c['SIZE_CHOQUE'])
                      or demasiado_cerca)
        if c['USAR_IR_CHOQUE'] and self._ir is not None:
            ir_max = max(max(v_, 0) for v_ in self._ir[:5]) / c['IR_SATURACION']
            choque = choque or ir_max >= c['IR_CHOQUE']

        perdido = bool((not demasiado_cerca)
                       and self._perdidos >= c['PERDIDO_MAX'])

        if visible:
            self._ultimo_size = size
            self._ultimo_posx = float(obs[0])
        if choque:
            r -= c['R_CHOQUE']
        if perdido:
            r -= c['R_PERDIDO_FIN']

        terminated = bool(choque or perdido)
        truncated = bool((not terminated) and self._pasos >= self.pasos_max)
        if terminated or truncated:
            self._detener()
        if terminated and self._pasos <= 2:
            # Un episodio que muere nada más empezar suele indicar un problema
            # de arranque (guía no visible o pegado tras el reinicio), no de
            # la política. Se avisa siempre para poder diagnosticarlo.
            print(f'AVISO: episodio terminado en el paso {self._pasos} '
                  f'({"choque" if choque else "guía perdido"}, '
                  f'size {float(size) if size is not None else 0.0:.1f}).')

        info = {
            'choque': choque,
            'perdido': perdido,
            'demasiado_cerca': demasiado_cerca,
            'visible': visible,
            'en_banda': bool(size is not None and not cortada
                             and c['SIZE_BANDA_MIN'] <= size <= c['SIZE_BANDA_MAX']),
            'cortada': cortada,
            'y_raw': float(self._blob[1]) if visible else 0.0,
            'posx': float(obs[0]),
            'size': float(obs[1]),
            'size_raw': float(size) if size is not None else 0.0,
            'v': v,
            'w': w,
            'pose_sim': self._pose,
            **terminos,
        }
        if self.verbose:
            print('paso {:3d}  posx {:+.2f}  size {:.2f} (raw {:6.1f})  '
                  'y {:5.1f}  v {:+.2f}  w {:+.2f}  r {:+.2f}{}'.format(
                      self._pasos, obs[0], obs[1],
                      float(size) if size is not None else 0.0,
                      info['y_raw'], v, w, r,
                      '  CHOQUE (encima)' if demasiado_cerca else
                      ('  CHOQUE (cortada)' if cortada else
                       '  CHOQUE' if choque else
                       ('  GU\u00cdA PERDIDO' if perdido else ''))))

        return obs, float(r), terminated, truncated, info

    def close(self):
        # Cierre best-effort: un fallo al parar no debe impedir liberar el nodo.
        with contextlib.suppress(Exception):
            self._detener()
        with contextlib.suppress(Exception):
            self.nodo.destroy_node()
        # No se llama a rclpy.shutdown(): otros entornos del mismo proceso
        # podrían seguir usándolo.
