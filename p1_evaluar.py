#!/usr/bin/env python3
"""
Práctica 1 — Evaluador del protocolo común (anexo B).

Ejecuta una política (aprendida o aleatoria) en modo determinista sobre el
entorno RoboboSeguidorEnv y la mide con la VERDAD DEL TERRENO del simulador,
con los criterios de fallo del protocolo e independientemente de la
recompensa y de las terminaciones propias del entorno.

Este script SÓLO ejecuta episodios: escribe un registro por paso y por
configuración (p1_logs/eval_<etiqueta>_<C>.csv). Las métricas, las tablas y
las figuras del anexo B se generan aparte con p1_graficas_eval.py, que es
quien define PROTOCOLO, CONFIGURACIONES y COLUMNAS (esquema del CSV).

----------------------------------------------------------------------
PROTOCOLO (anexo B, sin modificaciones)
----------------------------------------------------------------------
  Configuraciones del guía (se ajustan A MANO en RoboboSim; el script espera
  a que se pulse Enter antes de cada una):

      C1  Lenta constante   Crucero   10 - 10
      C2  Contraste         Crucero   10 - 40
      C3  Fluctuante        Variable  10 - 40

  Cada episodio: reinicio de la simulación, seguidor parado, 120 s.
  Fallos (se comprueban en cada paso):
      colisión      separación < 0,12 m
      guía perdido  separación > 0,90 m, o pelota sin detectar > 3 s seguidos
  Salirse del carril NO es fallo (se juzga en los vídeos).

  Separación = distancia entre el centro del seguidor y la pelota verde.
  Banda objetivo 0,20 - 0,45 m.
  Desviación = ángulo entre el rumbo del seguidor y la dirección a la pelota.

----------------------------------------------------------------------
VERDAD DEL TERRENO (sólo aquí; NUNCA en la observación ni en la política)
----------------------------------------------------------------------
  Se lee por una conexión directa con RoboboSim (librería robobosim, la misma
  que usa el entorno para reiniciar la escena), no por el módulo sim del
  puente: así el puente sigue lanzándose sólo con modules:="['blob']".

  En el escenario "Follow City" (comprobado con la propia librería):
      - el seguidor es el robot 0 (getRobotLocation(0)),
      - el guía NO es un robot sino el objeto 'PATHFOLLOWROBOBO',
      - la pelota verde es el objeto 'CYLINDERMIDBALL'.
  Por eso la separación se mide a la posición REAL de la pelota en lugar de
  desplazar la pose del guía 0,08 m hacia atrás: es la magnitud que define el
  anexo ("referencia de todas las medidas") medida sin aproximación. En la
  escena la pelota está a ~0,118 m del centro del guía, no a 0,08 m.

  Coordenadas en mm, suelo en el plano (x, z), rumbo = rotation.y en grados.
  Dirección de avance de un rumbo ry: (sin ry, cos ry) en (x, z) (comprobado:
  el desplazamiento del guía coincide con su rotation.y).

----------------------------------------------------------------------
USO
----------------------------------------------------------------------
  Puente (en otra terminal):
      ros2 run robobo_ros2 robobo_container --ros-args \\
          -p ip:=host.docker.internal -p modules:="['blob']"

  Política aprendida, las tres configuraciones seguidas:
      python3 p1_evaluar.py --algoritmo ppo --semilla 0
  Sólo una configuración:
      python3 p1_evaluar.py --algoritmo ppo --semilla 0 --configs C3
  Línea base aleatoria:
      python3 p1_evaluar.py --algoritmo aleatorio

  Registro por paso: p1_logs/eval_<etiqueta>_<C>.csv (uno por configuración).
  Tablas y figuras (sin simulador, a partir de esos CSV):
      python3 p1_graficas_eval.py
"""

import argparse
import contextlib
import csv
import math
import os
import time

import numpy as np

from p1_graficas_eval import (
    COLUMNAS,
    CONFIGURACIONES,
    PROTOCOLO,
    RUTA_LOGS,
    leer_csv,
    velocidad_guia,
)
from p1_robobo_env import (
    RUTA_CONFIG,
    RUTA_MODELOS,
    RoboboSeguidorEnv,
    cargar_config,
    dif_ang,
)

# Identificadores en la escena "Follow City" de RoboboSim.
ID_SEGUIDOR = 0
OBJ_GUIA = 'PATHFOLLOWROBOBO'
OBJ_PELOTA = 'CYLINDERMIDBALL'


# =====================================================================
# Verdad del terreno
# =====================================================================

class VerdadTerreno:
    """Pose del seguidor, del guía y de la pelota leídas de RoboboSim.
    """

    def __init__(self, ip, espera_max_s=5.0, caducidad_s=2.0):
        self.ip = ip
        self.espera_max_s = espera_max_s
        self.caducidad_s = caducidad_s
        self.sim = None
        self._pelota_prev = None
        self._t_pelota = time.time()
        self._conectar()

    def _conectar(self):
        from robobosim.RoboboSim import RoboboSim
        self.cerrar()
        self.sim = RoboboSim(self.ip)
        self.sim.connect()

    def _conectado(self):
        from robobosim.utils.ConnectionState import ConnectionState
        return (self.sim is not None
                and self.sim.rem.connectionState == ConnectionState.CONNECTED)

    def _crudo(self):
        rem = self.sim.rem
        return (rem.getRobotLocation(ID_SEGUIDOR),
                rem.getObjectLocation(OBJ_GUIA),
                rem.getObjectLocation(OBJ_PELOTA))

    def leer(self, nueva=False):
        """Devuelve (seguidor, guia, pelota) como tuplas (x, z, rumbo).
        """
        if not nueva and self._conectado():
            ahora = self._crudo()
            if ahora[2] is not self._pelota_prev:
                self._pelota_prev, self._t_pelota = ahora[2], time.time()
            if (all(d is not None for d in ahora)
                    and time.time() - self._t_pelota < self.caducidad_s):
                return tuple(_pose(d) for d in ahora)
        for intento in range(2):
            if not self._conectado():
                self._conectar()
            antes = self._crudo()
            t0 = time.time()
            while time.time() - t0 < self.espera_max_s:
                ahora = self._crudo()
                completos = all(d is not None for d in ahora)
                # La pelota se renueva siempre (~22 Hz). Del seguidor se
                # espera una pose nueva, salvo que tarde más de 1 s.
                renovados = (ahora[2] is not antes[2]
                             and (ahora[0] is not antes[0]
                                  or time.time() - t0 > 1.0))
                if completos and renovados:
                    self._pelota_prev, self._t_pelota = ahora[2], time.time()
                    return tuple(_pose(d) for d in ahora)
                time.sleep(0.01)
            if intento == 0:
                print('Aviso: no llegan posiciones de RoboboSim; reconectando.')
                self._conectar()
        faltan = [n for n, d in zip(('seguidor', OBJ_GUIA, OBJ_PELOTA), self._crudo())
                  if d is None]
        raise RuntimeError(
            f'No se reciben posiciones de RoboboSim en {self.ip}:50505 '
            f'(faltan: {faltan or "ninguna, pero no se renuevan"}). '
            '¿Está abierto el escenario Follow City?')

    def cerrar(self):
        if self.sim is not None:
            with contextlib.suppress(Exception):
                self.sim.disconnect()
        self.sim = None


def _pose(d):
    return (d['position']['x'], d['position']['z'], d['rotation']['y'])


def medir(seguidor, pelota):
    """Separación (m) y desviación (°) del seguidor respecto a la pelota.

    Desviación con el convenio de rotation.y de RoboboSim. Comparada con la
    posx del blob sale con signo CONTRARIO (correlación medida -0,91).
    """
    dx = pelota[0] - seguidor[0]
    dz = pelota[1] - seguidor[1]
    separacion = math.hypot(dx, dz) / 1000.0
    direccion = math.degrees(math.atan2(dx, dz))
    return separacion, dif_ang(direccion, seguidor[2])


# =====================================================================
# Episodios
# =====================================================================

def politica_de(args, env):
    """Devuelve (función obs -> acción, etiqueta para los archivos)."""
    if args.algoritmo == 'aleatorio':
        return (lambda obs: env.action_space.sample()), 'aleatorio'

    from stable_baselines3 import PPO, SAC
    ruta = args.modelo or os.path.join(RUTA_MODELOS, f'p1_modelo_{args.algoritmo}_s{args.semilla}')
    ruta = ruta.removesuffix('.zip')
    if not os.path.exists(ruta + '.zip'):
        raise SystemExit(f'No existe {ruta}.zip')
    clase = PPO if args.algoritmo == 'ppo' else SAC
    modelo = clase.load(ruta)
    dim_modelo = modelo.observation_space.shape
    if dim_modelo != env.observation_space.shape:
        raise SystemExit(
            f'El modelo espera observaciones {dim_modelo} y el entorno da '
            f'{env.observation_space.shape}. Ajusta OBS_Y / OBS_IR en la '
            'configuración a los que se usaron al entrenarlo.')
    etiqueta = args.etiqueta or os.path.basename(ruta).replace('p1_modelo_', '')
    # Determinista: la acción media de la política, sin exploración.
    return (lambda obs: modelo.predict(obs, deterministic=True)[0]), etiqueta


def episodio(env, verdad, politica, config, num, escritor, args):
    """Ejecuta un episodio del protocolo y devuelve su desenlace."""
    p = PROTOCOLO
    obs, _ = env.reset()           # reinicio de la escena, seguidor parado
    t_reset = time.time()
    verdad.leer(nueva=True)        # posiciones ya posteriores al reinicio

    t0 = time.time()
    t_visto = t0                   # instante de la última detección de la pelota
    paso, desenlace = 0, None
    print(f'\n{config} episodio {num}: (reinicio en {t0 - t_reset:.1f} s)')

    while desenlace is None:
        accion = np.asarray(politica(obs), dtype=np.float32)
        # Las terminaciones propias del entorno se REGISTRAN pero no cortan
        # el episodio: el protocolo decide el fin con sus propios criterios.
        obs, _, term_env, _, info = env.step(accion)
        seguidor, guia, pelota = verdad.leer()
        ahora = time.time()
        paso += 1
        t = ahora - t0

        separacion, desviacion = medir(seguidor, pelota)
        visible = bool(info.get('visible'))
        if visible:
            t_visto = ahora
        t_sin_ver = ahora - t_visto

        if separacion < p['COLISION_M']:
            desenlace = 'colision'
        elif separacion > p['PERDIDO_M']:
            desenlace = 'perdido_distancia'
        elif t_sin_ver > p['SIN_VER_S']:
            desenlace = 'perdido_sin_ver'
        elif t >= p['DURACION_S']:
            desenlace = 'completado'

        en_banda = p['BANDA_MIN_M'] <= separacion <= p['BANDA_MAX_M']
        escritor.writerow({
            'config': config, 'episodio': num, 'paso': paso, 't': f'{t:.3f}',
            'seg_x_mm': f'{seguidor[0]:.1f}', 'seg_z_mm': f'{seguidor[1]:.1f}',
            'seg_rumbo_deg': f'{seguidor[2]:.1f}',
            'pelota_x_mm': f'{pelota[0]:.1f}', 'pelota_z_mm': f'{pelota[1]:.1f}',
            'guia_x_mm': f'{guia[0]:.1f}', 'guia_z_mm': f'{guia[1]:.1f}',
            'separacion_m': f'{separacion:.4f}', 'en_banda': int(en_banda),
            'desviacion_deg': f'{desviacion:.2f}',
            'visible': int(visible), 'blob_posx': f'{info.get("posx", 0.0):.4f}',
            'blob_size': f'{info.get("size_raw", 0.0):.1f}',
            't_sin_ver': f'{t_sin_ver:.2f}',
            'a0': f'{float(accion[0]):.4f}', 'a1': f'{float(accion[1]):.4f}',
            'v': f'{info.get("v", 0.0):.4f}', 'w': f'{info.get("w", 0.0):.4f}',
            'entorno_choque': int(bool(term_env and info.get('choque'))),
            'entorno_perdido': int(bool(term_env and info.get('perdido'))),
            'desenlace': desenlace or '',
        })
        if args.detalle:
            print(f'  t {t:6.2f}  sep {separacion:.3f} m  desv {desviacion:+6.1f}°  '
                  f'blob {"sí" if visible else "no"} posx {info.get("posx", 0.0):+.2f}  '
                  f'v {info.get("v", 0.0):.2f} w {info.get("w", 0.0):+.2f}'
                  + (f'  -> {desenlace}' if desenlace else ''))

    env._detener()
    print(f'  {desenlace}  a los {t:.1f} s ({paso} pasos), separación final '
          f'{separacion:.3f} m')
    return desenlace


def main():
    a = argparse.ArgumentParser(
        description='Evaluador del protocolo común (anexo B), con verdad del terreno.')
    a.add_argument('--algoritmo', choices=['ppo', 'sac', 'aleatorio'], required=True)
    a.add_argument('--semilla', type=int, default=0)
    a.add_argument('--modelo', default=None,
                   help='ruta del .zip (por defecto modelos/p1_modelo_<algoritmo>_s<semilla>)')
    a.add_argument('--etiqueta', default=None,
                   help='nombre para los archivos de registro (por defecto, el del modelo)')
    a.add_argument('--configs', nargs='+', choices=sorted(CONFIGURACIONES),
                   default=sorted(CONFIGURACIONES))
    a.add_argument('--episodios', type=int, default=1,
                   help='episodios por configuración (el protocolo pide 1 de 120 s)')
    a.add_argument('--config', default=RUTA_CONFIG, help='configuración del entorno')
    a.add_argument('--detalle', action='store_true', help='una línea por paso')
    args = a.parse_args()

    c = cargar_config(args.config)
    # pasos_max enorme: el fin del episodio lo decide el protocolo (120 s),
    # no el truncado del entorno de entrenamiento.
    env = RoboboSeguidorEnv(pasos_max=10**9, usar_simulador=True,
                            ruta_config=args.config)
    verdad = None
    try:
        politica, etiqueta = politica_de(args, env)
        verdad = VerdadTerreno(c['SIM_IP'])
        os.makedirs(RUTA_LOGS, exist_ok=True)
        print(f'Evaluando "{etiqueta}" ({"determinista" if args.algoritmo != "aleatorio" else "aleatoria"}). '
              'Recuerda: mismo multiplicador de velocidad y físicas que en el entrenamiento.')

        for config in args.configs:
            input(f'\nPon el guía en {config}: {CONFIGURACIONES[config]}\n'
                  'y pulsa Enter... ')
            ruta = os.path.join(RUTA_LOGS, f'eval_{etiqueta}_{config}.csv')
            with open(ruta, 'w', newline='') as f:
                escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
                escritor.writeheader()
                for num in range(1, args.episodios + 1):
                    episodio(env, verdad, politica, config, num, escritor, args)
                    f.flush()
            print(f'registro por paso: {ruta}')
            media, perfil = velocidad_guia(leer_csv(ruta))
            if media is not None:
                print(f'velocidad media real del guía en {config}: {media:.3f} m/s '
                      '(por 10 s: ' + ' '.join(f'{v:.2f}' for v in perfil) + ')')

        print(f'\nTablas y figuras del anexo B:\n'
              f'    python3 p1_graficas_eval.py --etiquetas {etiqueta}')
    finally:
        if verdad is not None:
            verdad.cerrar()
        env.close()


if __name__ == '__main__':
    main()
