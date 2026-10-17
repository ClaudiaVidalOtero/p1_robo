#!/usr/bin/env python3
"""
Práctica 1 — Entorno Gymnasium: el Robobo SEGUIDOR sigue al vehículo guía.

El guía lleva una pelota verde en la parte trasera y no reacciona al
seguidor. El seguidor sólo lo percibe con el detector de blobs de color del
smartphone (un blob verde por imagen). Toda la percepción y la actuación del
seguidor pasan por ROS 2 (nodo puente robobo_ros2).

OBSERVACIÓN (Box float32; sólo sensores del seguidor)
    [0] posx = (x_blob - POSX_CENTRO) / POSX_CENTRO   en [-1, 1]
    [1] size = size_blob / SIZE_MAX                    en [0, 1]
    [2] posy = y_blob / IMG_ALTO                       en [0, 1]   (OBS_Y)
    [3] ir   = IR frontal máximo / IR_SATURACION       en [0, 1]   (OBS_IR)
    Pelota no visible (o size < SIZE_MIN_VISTA): posx, size y posy valen 0.
    posy distingue "pegado" (pelota cortada por abajo) de "lejos", que dan un
    size parecido.

ACCIÓN (Box float32, continua)
    [0] v = V_MIN + (a0 + 1)/2 * (V_MAX - V_MIN)   m/s
    [1] w = a1 * W_MAX                              rad/s
    Se publica en base/cmd_vel, que se detiene si no recibe órdenes en 0,5 s:
    el periodo de control PASO_S (0,2 s) es menor.

RECOMPENSA (por paso, a partir de posx, size e y del blob)
    vista       +R_VISTA                       si el blob es visible
    centrado    -W_CENT  * |posx|
    separacion  +W_BANDA si size en banda;  -K_FUERA * excursión si fuera
    giro        -K_GIRO  * |a1|
    d_giro      -K_DGIRO * |a1 - a1_anterior|
    perdido     -R_PERDIDO_PASO                si el blob no es visible
    terminal    -R_CHOQUE al chocar; -R_PERDIDO_FIN al perder al guía

TERMINACIÓN: choque (pelota cortada, size >= SIZE_CHOQUE o pelota que
desaparece tras verse cerca y centrada) o guía perdido (PERDIDO_MAX pasos
sin verlo). TRUNCAMIENTO: pasos_max.

REINICIO: con simulador, reinicio de la escena por conexión directa con
RoboboSim (el puente sólo necesita el módulo 'blob'); sin simulador (robot
real), una persona recoloca el seguidor y pulsa Enter.

Las constantes que dependen de la calibración se leen de config/p1_config.json.
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

AVISO_SIN_BLOB = (
    'No se encuentran las interfaces de blobs '
    '(robobo_ros2_interfaces.msg.BlobArray).\n'
    'Cargar el espacio de trabajo del puente:\n'
    '      source /opt/ros/robobo/setup.bash')

COLOR_BLOB = 'green'

RUTA_CONFIG = os.path.join('config', 'p1_config.json')
RUTA_CALIBRACION = os.path.join('config', 'calibracion.json')
RUTA_MODELOS = 'modelos'


# =====================================================================
# Configuración
# =====================================================================

# Valores de reserva: se usan si falta una clave en p1_config.json.
CONFIG_POR_DEFECTO = {
    'NS': '/robobo/robot_0',
    'COLOR_BLOB': COLOR_BLOB,

    'IMG_ANCHO': 100.0,        # rango de x del blob (px)
    'POSX_CENTRO': 50.0,
    'IMG_ALTO': 100.0,         # rango de y del blob (px)
    'SIZE_MAX': 330.0,         # normalización del size

    'BANDA_PROTOCOLO_M': [0.20, 0.45],  # banda del anexo B (centro-pelota)
    'CAMARA_OFFSET_M': 0.09,   # distancia del centro del robot a la cámara
    'SIZE_BANDA_MIN': 100.0,   # size en el borde lejano de la banda
    'SIZE_BANDA_MAX': 207.0,   # size en el borde cercano (0,25 m de la cámara)
    'SIZE_OBJ': 144.0,         # size en el centro de la banda
    'SIZE_CHOQUE': 323.4,      # size a ~0,20 m
    'SIZE_MIN_VISTA': 12.9,    # size a ~1,0 m; por debajo, "no visible"
    'SIZE_CERCA': 144.0,       # si la pelota desaparece tras verse con este
    'POSX_CERCA_MAX': 0.6,     # size y |posx| menor que esto, es un choque
    'Y_CHOQUE': None,          # y a partir de la cual la pelota sale cortada

    'V_MIN': 0.0,              # m/s
    'V_MAX': 0.15,             # m/s (mayor que la velocidad máxima del guía)
    'W_MAX': 0.5,              # rad/s

    'PASO_S': 0.2,             # periodo de control = periodo del blob (5 Hz)
    'TIMEOUT_LECTURA_S': 2.0,  # espera máxima por un blob nuevo
    'PERDIDO_MAX': 12,         # pasos seguidos sin ver el guía -> perdido
    'PASOS_MAX': 300,          # 300 pasos * 0,2 s = 60 s por episodio

    'RESET_ESPERA_S': 1.5,         # espera tras reiniciar la escena
    'RESET_INTERVALO_MIN_S': 4.0,  # RoboboSim corta la conexión si recibe
                                   # reinicios encadenados
    'SIM_IP': 'host.docker.internal',

    'TILT': 90.0,              # inclinación de la cámara usada al calibrar
    'TILT_VEL': 30.0,

    'OBS_Y': True,             # añade posy a la observación
    'OBS_IR': False,           # añade el IR frontal a la observación
    'USAR_IR_CHOQUE': False,   # el IR frontal también detecta choque
    'IR_SATURACION': 1000.0,
    'IR_CHOQUE': 0.8,

    'W_CENT': 0.5,
    'W_BANDA': 1.0,
    'K_FUERA': 0.8,
    'K_GIRO': 0.3,
    'K_DGIRO': 0.3,
    'R_VISTA': 0.5,
    'R_PERDIDO_PASO': 1.0,
    'R_PERDIDO_FIN': 5.0,
    'R_CHOQUE': 20.0,          # mayor que perder al guía (12*1 + 5 = 17)
}


def cargar_config(ruta=RUTA_CONFIG):
    """Lee p1_config.json y completa las claves que falten con los valores de reserva."""
    c = dict(CONFIG_POR_DEFECTO)
    if ruta and os.path.exists(ruta):
        with open(ruta, encoding='utf-8') as f:
            c.update(json.load(f))
    return c


def importar_calibracion(json_cal=RUTA_CALIBRACION, ruta_config=RUTA_CONFIG,
                         guardar=True):
    """Deriva las constantes del entorno a partir de calibracion.json.

    Ajusta la ley size ≈ K / d² con las medidas de d >= 0,25 m (más cerca la
    pelota sale cortada y el size deja de crecer) y calcula los size de la
    banda, de choque y de visibilidad mínima, el centro de la imagen y
    Y_CHOQUE. Escribe el resultado en p1_config.json y lo devuelve.
    """
    with open(json_cal, encoding='utf-8') as f:
        datos = json.load(f)
    if isinstance(datos, dict):
        datos = [datos]

    puntos, xs, ys = [], [], {}
    for e in datos:
        d = float(e['distancia'])
        lec = e.get('lecturas', [])
        centrada = [r for r in lec
                    if r.get('etapa') in ('centro', 'reposo') and r.get('size', 0) > 0]
        ys_d = [r['y'] for r in centrada if r.get('y') is not None]
        if ys_d:
            ys.setdefault(d, []).extend(ys_d)
        todos = [r['size'] for r in lec if r.get('size', 0) > 0]
        if centrada:
            tam = statistics.median(r['size'] for r in centrada)
        else:
            tam = statistics.median(todos) if todos else 0.0
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
    # Banda del protocolo (centro-pelota) pasada a distancia desde la cámara;
    # el borde cercano no baja de 0,25 m porque ahí el size deja de crecer.
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

    # Y_CHOQUE: punto medio entre la y a 0,20 m (pelota cortada) y a 0,25 m.
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
    print(f'  size(0,20 m) = {size(0.20):7.1f}   size(0,25 m) = {size(0.25):7.1f}')
    print(f'  size(0,375 m)= {size(0.375):7.1f}   size(0,50 m) = {size(0.50):7.1f}')
    print(f'  IMG_ANCHO = {c["IMG_ANCHO"]:.0f}   POSX_CENTRO = {c["POSX_CENTRO"]:.0f}')
    print(f'  SIZE_MAX = {c["SIZE_MAX"]:.0f}   banda = [{c["SIZE_BANDA_MIN"]:.1f}, '
          f'{c["SIZE_BANDA_MAX"]:.1f}]   choque = {c["SIZE_CHOQUE"]:.1f}')

    if guardar:
        with open(ruta_config, 'w', encoding='utf-8') as f:
            json.dump(c, f, indent=2, ensure_ascii=False)
        print(f'Configuración escrita en {ruta_config}')
    return c


def dif_ang(a, b):
    """Diferencia angular a - b en (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


# =====================================================================
# El entorno
# =====================================================================

class RoboboSeguidorEnv(gym.Env):
    """Entorno del Robobo seguidor.

    pasos_max       longitud máxima del episodio (truncado).
    usar_simulador  True: reinicio de la escena de RoboboSim; False: manual.
    verbose         imprime una línea por paso.
    ruta_config     archivo de configuración calibrada.
    """

    metadata: ClassVar[dict] = {'render_modes': []}

    def __init__(self, pasos_max=None, usar_simulador=True, verbose=False,
                 ruta_config=RUTA_CONFIG):
        super().__init__()
        if not HAY_INTERFACES_BLOB:
            raise RuntimeError(AVISO_SIN_BLOB)

        self.c = cargar_config(ruta_config)
        self.pasos_max = int(pasos_max if pasos_max is not None else self.c['PASOS_MAX'])
        self.usar_simulador = usar_simulador
        self.verbose = verbose

        # rclpy.init() sólo una vez por proceso (puede haber varios entornos).
        if not rclpy.ok():
            rclpy.init()
        self.nodo = Node('p1_seguidor_env')
        ns = self.c['NS']

        # Publicador y suscripciones se crean una sola vez: el descubrimiento
        # de ROS 2 tarda y se perderían las primeras órdenes.
        self.pub_vel = self.nodo.create_publisher(Twist, ns + '/base/cmd_vel', 10)

        self._blob = None      # último blob verde: (x, y, size) o None
        self._n_msgs = 0       # mensajes de blobs recibidos
        self.nodo.create_subscription(
            BlobArray, ns + '/smartphone/color_blobs', self._cb_blobs, 1)

        self._ir = None
        self.nodo.create_subscription(Int32MultiArray, ns + '/base/ir', self._cb_ir, 1)

        self.cli_blobs = self.nodo.create_client(
            SetActiveColorBlobs, ns + '/smartphone/set_active_color_blobs')
        self.cli_tilt = self.nodo.create_client(MoveTilt, ns + '/base/move_tilt')

        self._t_ultimo_reset = None

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
            high=np.array(alto, dtype=np.float32), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self._pasos = 0
        self._perdidos = 0
        self._ultimo_size = 0.0
        self._ultimo_posx = 0.0
        self._a_prev = np.zeros(2, dtype=np.float32)

        self._colocar_tilt()
        self._activar_verde()
        self._esperar_lectura_nueva(inicial=True)   # comprueba que el puente publica

    # -----------------------------------------------------------------
    # Comunicación con ROS 2
    # -----------------------------------------------------------------

    def _cb_blobs(self, msg):
        """Guarda el blob verde de mayor tamaño (o None si no hay)."""
        mejor = None
        for b in msg.blobs:
            if b.color == self.c['COLOR_BLOB'] and (mejor is None or b.size > mejor[2]):
                mejor = (float(b.x), float(b.y), float(b.size))
        self._blob = mejor
        self._n_msgs += 1      # las listas vacías también cuentan como lectura

    def _cb_ir(self, msg):
        self._ir = list(msg.data)

    def _spin(self, timeout=0.0):
        rclpy.spin_once(self.nodo, timeout_sec=timeout)

    def _vaciar_cola(self):
        """Procesa y descarta los mensajes recibidos mientras el robot se movía."""
        for _ in range(100):
            antes = self._n_msgs
            self._spin(0.0)
            if self._n_msgs == antes:
                return

    def _esperar_lectura_nueva(self, inicial=False):
        """Bloquea hasta recibir un mensaje de blobs POSTERIOR a esta llamada.

        Garantiza que step() devuelve la observación resultante de la acción y
        no una lectura anterior guardada.
        """
        objetivo = self._n_msgs + 1
        t0 = time.time()
        while self._n_msgs < objetivo:
            self._spin(0.05)
            if time.time() - t0 > self.c['TIMEOUT_LECTURA_S']:
                if inicial:
                    raise RuntimeError(
                        f'No llegan mensajes de {self.c["NS"]}/smartphone/color_blobs. '
                        "Lanza robobo_container con modules:=\"['blob']\".")
                raise RuntimeError('Se ha interrumpido el flujo de blobs durante el episodio.')

    def _publicar_velocidad(self, v, w):
        m = Twist()
        m.linear.x = float(v)
        m.angular.z = float(w)
        self.pub_vel.publish(m)

    def _detener(self):
        for _ in range(3):
            self._publicar_velocidad(0.0, 0.0)
            time.sleep(0.05)

    def _llamar(self, cliente, peticion, espera=10.0):
        if not cliente.wait_for_service(timeout_sec=espera):
            raise RuntimeError('Servicio no disponible: ' + cliente.srv_name)
        futuro = cliente.call_async(peticion)
        rclpy.spin_until_future_complete(self.nodo, futuro, timeout_sec=espera)
        return futuro.result()

    def _activar_verde(self):
        req = SetActiveColorBlobs.Request()
        req.red, req.green, req.blue, req.custom = False, True, False, False
        self._llamar(self.cli_blobs, req)

    def _colocar_tilt(self):
        """Fija el tilt de la calibración; con otro, la relación size-distancia no vale."""
        req = MoveTilt.Request()
        req.angle = float(self.c['TILT'])
        req.speed = float(self.c['TILT_VEL'])
        try:
            self._llamar(self.cli_tilt, req)
            time.sleep(0.5)    # tiempo para que el servo llegue
        except RuntimeError as exc:
            print(f'Aviso: no se pudo fijar el tilt: {exc}')

    # -----------------------------------------------------------------
    # Reinicio de la escena (sólo simulación)
    # -----------------------------------------------------------------

    def _reiniciar_escena(self):
        # Intervalo mínimo entre reinicios: si llegan seguidos, RoboboSim
        # cierra la conexión.
        if self._t_ultimo_reset is not None:
            falta = self.c['RESET_INTERVALO_MIN_S'] - (time.time() - self._t_ultimo_reset)
            if falta > 0:
                time.sleep(falta)
        self._reset_directo()
        self._t_ultimo_reset = time.time()
        time.sleep(self.c['RESET_ESPERA_S'])

    def _reset_directo(self, intentos=3):
        """Reinicia la escena con una conexión nueva y corta a RoboboSim.

        La conexión permanente del módulo sim del puente se cae tras unos
        episodios y no reconecta; una conexión por reinicio evita el problema.
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
                time.sleep(0.3)   # que procese la autenticación
                sim.resetSimulation()
                time.sleep(0.2)   # que el mensaje salga antes de cerrar
                return
            # robobosim hace sys.exit si se envía sin conexión.
            except (OSError, SystemExit, websocket.WebSocketException) as exc:
                ultimo_error = exc
                time.sleep(1.0)
            finally:
                with contextlib.suppress(OSError, websocket.WebSocketException):
                    sim.disconnect()
        raise RuntimeError(
            f'No se pudo reiniciar RoboboSim en {self.c["SIM_IP"]}:50505 '
            f'tras {intentos} intentos ({ultimo_error}).')

    # -----------------------------------------------------------------
    # Observación
    # -----------------------------------------------------------------

    def _size_visible(self):
        """Size del blob, o None si no se ve o es demasiado pequeño."""
        if self._blob is None or self._blob[2] < self.c['SIZE_MIN_VISTA']:
            return None
        return self._blob[2]

    def _pelota_cortada(self):
        """True si la pelota sale cortada por abajo (guía encima). Requiere Y_CHOQUE."""
        y_choque = self.c.get('Y_CHOQUE')
        return bool(y_choque is not None and self._size_visible() is not None
                    and self._blob[1] >= y_choque)

    def _ir_frontal(self):
        """IR frontal máximo normalizado en [0, 1]."""
        if not self._ir:
            return 0.0
        ir_max = max(max(v, 0) for v in self._ir[:5])
        return float(np.clip(ir_max / self.c['IR_SATURACION'], 0.0, 1.0))

    def _observacion(self):
        size = self._size_visible()
        if size is None:
            base = [0.0, 0.0] + ([0.0] if self.c.get('OBS_Y') else [])
        else:
            posx = (self._blob[0] - self.c['POSX_CENTRO']) / self.c['POSX_CENTRO']
            base = [float(np.clip(posx, -1.0, 1.0)),
                    float(np.clip(size / self.c['SIZE_MAX'], 0.0, 1.0))]
            if self.c.get('OBS_Y'):
                base.append(float(np.clip(self._blob[1] / self.c['IMG_ALTO'], 0.0, 1.0)))
        if self.c.get('OBS_IR'):
            base.append(self._ir_frontal())
        return np.array(base, dtype=np.float32)

    # -----------------------------------------------------------------
    # Recompensa
    # -----------------------------------------------------------------

    def _excursion_banda(self, size):
        """Distancia normalizada fuera de la banda de size, en [0, 1] (0 dentro)."""
        lo, hi = self.c['SIZE_BANDA_MIN'], self.c['SIZE_BANDA_MAX']
        if size < lo:
            e = (lo - size) / lo
        elif size > hi:
            e = (size - hi) / max(self.c['SIZE_MAX'] - hi, 1e-6)
        else:
            return 0.0
        return float(np.clip(e, 0.0, 1.0))

    def _recompensa(self, obs, a, a_prev):
        """Recompensa del paso y diccionario con cada término por separado."""
        c = self.c
        size = self._size_visible()
        if size is None:
            return -c['R_PERDIDO_PASO'], {'r_perdido': -c['R_PERDIDO_PASO']}

        # Con la pelota cortada el size cae dentro de la banda, pero está pegado.
        en_banda = (c['SIZE_BANDA_MIN'] <= size <= c['SIZE_BANDA_MAX']
                    and not self._pelota_cortada())
        terminos = {
            'vista': c['R_VISTA'],
            'centrado': -c['W_CENT'] * abs(float(obs[0])),
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
        super().reset(seed=seed, options=options)
        self._detener()

        if self.usar_simulador:
            self._reiniciar_escena()
            self._colocar_tilt()    # el reinicio puede cambiar el tilt
            self._activar_verde()   # y desactivar los blobs
            time.sleep(0.3)
        else:
            input('Coloca el seguidor detrás del guía y pulsa Enter... ')

        self._pasos = 0
        self._perdidos = 0
        self._ultimo_size = 0.0
        self._ultimo_posx = 0.0
        self._a_prev = np.zeros(2, dtype=np.float32)

        # Observación tomada ya en la posición nueva.
        self._vaciar_cola()
        self._esperar_lectura_nueva()
        return self._observacion(), {}

    def step(self, accion):
        c = self.c
        a = np.clip(np.asarray(accion, dtype=np.float32), -1.0, 1.0)

        # 1. Acción normalizada -> m/s y rad/s.
        v = c['V_MIN'] + (float(a[0]) + 1.0) / 2.0 * (c['V_MAX'] - c['V_MIN'])
        w = float(a[1]) * c['W_MAX']
        self._publicar_velocidad(v, w)

        # 2. Esperar a que la acción surta efecto y repetir la orden por el
        #    temporizador de 0,5 s de cmd_vel.
        time.sleep(c['PASO_S'])
        self._publicar_velocidad(v, w)

        # 3. Descartar lo anterior y esperar una lectura posterior a la acción.
        self._vaciar_cola()
        self._esperar_lectura_nueva()

        self._pasos += 1
        obs = self._observacion()
        r, terminos = self._recompensa(obs, a, self._a_prev)
        self._a_prev = a

        size = self._size_visible()
        visible = size is not None
        self._perdidos = 0 if visible else self._perdidos + 1

        # Si la pelota desaparece justo después de verse cerca y centrada, ha
        # salido por abajo: el seguidor se ha echado encima del guía.
        demasiado_cerca = bool(not visible
                               and self._ultimo_size >= c['SIZE_CERCA']
                               and abs(self._ultimo_posx) < c['POSX_CERCA_MAX'])
        cortada = self._pelota_cortada()
        choque = bool(cortada
                      or (visible and size >= c['SIZE_CHOQUE'])
                      or demasiado_cerca)
        if c['USAR_IR_CHOQUE'] and self._ir is not None:
            choque = choque or self._ir_frontal() >= c['IR_CHOQUE']
        perdido = bool(not demasiado_cerca and self._perdidos >= c['PERDIDO_MAX'])

        if visible:
            self._ultimo_size = size
            self._ultimo_posx = float(obs[0])
        if choque:
            r -= c['R_CHOQUE']
        if perdido:
            r -= c['R_PERDIDO_FIN']

        terminated = bool(choque or perdido)
        truncated = bool(not terminated and self._pasos >= self.pasos_max)
        if terminated or truncated:
            self._detener()
        if terminated and self._pasos <= 2:
            # Un fin inmediato suele ser un problema del arranque, no de la política.
            print(f'AVISO: episodio terminado en el paso {self._pasos} '
                  f'({"choque" if choque else "guía perdido"}, '
                  f'size {size if visible else 0.0:.1f}).')

        info = {
            'choque': choque,
            'perdido': perdido,
            'demasiado_cerca': demasiado_cerca,
            'visible': visible,
            'en_banda': bool(visible and not cortada
                             and c['SIZE_BANDA_MIN'] <= size <= c['SIZE_BANDA_MAX']),
            'cortada': cortada,
            'y_raw': float(self._blob[1]) if visible else 0.0,
            'posx': float(obs[0]),
            'size': float(obs[1]),
            'size_raw': float(size) if visible else 0.0,
            'v': v,
            'w': w,
            **terminos,
        }
        if self.verbose:
            fin = ('  CHOQUE (encima)' if demasiado_cerca else
                   '  CHOQUE (cortada)' if cortada else
                   '  CHOQUE' if choque else
                   '  GUÍA PERDIDO' if perdido else '')
            print(f'paso {self._pasos:3d}  posx {obs[0]:+.2f}  size {obs[1]:.2f} '
                  f'(raw {info["size_raw"]:6.1f})  y {info["y_raw"]:5.1f}  '
                  f'v {v:+.2f}  w {w:+.2f}  r {r:+.2f}{fin}')

        return obs, float(r), terminated, truncated, info

    def close(self):
        with contextlib.suppress(Exception):
            self._detener()
        with contextlib.suppress(Exception):
            self.nodo.destroy_node()
        # Sin rclpy.shutdown(): otros entornos del proceso pueden seguir usándolo.
