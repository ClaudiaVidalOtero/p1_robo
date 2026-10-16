#!/usr/bin/env python3
"""
Práctica 1 — Evaluador del protocolo común (anexo B).

Ejecuta una política (aprendida o aleatoria) en modo determinista sobre el
entorno RoboboSeguidorEnv y la mide con la VERDAD DEL TERRENO del simulador,
con los criterios de fallo del protocolo e independientemente de la
recompensa y de las terminaciones propias del entorno.

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

  Métricas, en conjunto y por configuración:
      completados   número y % de episodios completados
      desenlaces    episodios terminados por cada causa
      separación    % de pasos en banda y separación media
      desviación    media de |desviación| (°) y % de pasos con |desv| < 15°

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
  Rehacer las tablas a partir de los CSV ya guardados (sin simulador):
      python3 p1_evaluar.py --algoritmo ppo --semilla 0 --resumen

  Registro por paso: p1_logs/eval_<etiqueta>_<C>.csv (uno por configuración).
  Las métricas se calculan SIEMPRE a partir de esos CSV, así que las tablas de
  la memoria se pueden rehacer con --resumen.
"""

import argparse
import contextlib
import csv
import glob
import math
import os
import time

import numpy as np

from p1_robobo_env import (
    RUTA_CONFIG,
    RUTA_MODELOS,
    RoboboSeguidorEnv,
    cargar_config,
    dif_ang,
)

RUTA_LOGS = 'p1_logs'

# Parámetros del protocolo (anexo B). Son comunes a todos los grupos y no
# dependen de la calibración: por eso están aquí y no en p1_config.json.
PROTOCOLO = {
    'DURACION_S': 120.0,
    'BANDA_MIN_M': 0.20,
    'BANDA_MAX_M': 0.45,
    'COLISION_M': 0.12,
    'PERDIDO_M': 0.90,
    'SIN_VER_S': 3.0,
    'DESV_UMBRAL_DEG': 15.0,
}

CONFIGURACIONES = {
    'C1': 'Lenta constante   modo Crucero   velocidad 10 - 10',
    'C2': 'Contraste         modo Crucero   velocidad 10 - 40',
    'C3': 'Fluctuante        modo Variable  velocidad 10 - 40',
}

# Identificadores en la escena "Follow City" de RoboboSim.
ID_SEGUIDOR = 0
OBJ_GUIA = 'PATHFOLLOWROBOBO'
OBJ_PELOTA = 'CYLINDERMIDBALL'

COLUMNAS = ['config', 'episodio', 'paso', 't',
            'seg_x_mm', 'seg_z_mm', 'seg_rumbo_deg',
            'pelota_x_mm', 'pelota_z_mm', 'guia_x_mm', 'guia_z_mm',
            'separacion_m', 'en_banda', 'desviacion_deg',
            'visible', 'blob_posx', 'blob_size', 't_sin_ver',
            'a0', 'a1', 'v', 'w',
            'entorno_choque', 'entorno_perdido', 'desenlace']


# =====================================================================
# Verdad del terreno
# =====================================================================

class VerdadTerreno:
    """Pose del seguidor, del guía y de la pelota leídas de RoboboSim.

    RoboboSim envía las posiciones por la conexión websocket de forma
    continua; la librería las guarda en un diccionario que se REEMPLAZA con
    cada mensaje. Comparar la identidad del diccionario permite saber si ha
    llegado una lectura nueva sin usar los callbacks de la librería (el de
    objetos no funciona en robobosim 1.5.1: se registra con otra clave).

    Sólo se leen datos, nunca se envía nada: así no puede saltar el sys.exit
    que hace la librería al enviar sin conexión.

    Frecuencias medidas: pelota y guía ~22 Hz; seguidor ~5 Hz (huecos de
    hasta 0,27 s). En cada paso se toma la ÚLTIMA lectura sin esperar: esperar
    una pose nueva del seguidor alargaría el paso hasta 0,27 s y cambiaría el
    periodo de control respecto al entrenamiento. El error que introduce es
    como mucho v * 0,27 s (~4 cm a 0,15 m/s); se declara en la memoria.
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

        nueva=False: la última lectura recibida (uso en cada paso). Si la
        pelota, que se renueva a ~22 Hz, lleva más de caducidad_s sin
        renovarse, la conexión se da por caída y se espera una lectura nueva.
        nueva=True: espera a que lleguen posiciones renovadas tras la llamada
        (uso tras reiniciar la escena). Si no llega nada, reconecta
        una vez.
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
                # espera una pose nueva, salvo que tarde más de 1 s (por si
                # RoboboSim no la envía mientras está parado).
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


def velocidad_guia(filas):
    """Velocidad media real del guía (m/s) y su perfil por tramos de 10 s.

    La configuración del guía se pone a mano en RoboboSim y nada la confirma.
    Medir su velocidad con la verdad del terreno detecta una configuración
    que no se ha aplicado (en la primera evaluación, C1, C2 y C3 dieron el
    mismo perfil: 0,136 m/s y 16,3 m en 120 s).
    """
    if len(filas) < 2:
        return None, []
    t = np.array([float(f['t']) for f in filas])
    x = np.array([float(f['guia_x_mm']) for f in filas])
    z = np.array([float(f['guia_z_mm']) for f in filas])
    d = np.hypot(np.diff(x), np.diff(z)) / 1000.0
    dt = np.diff(t)
    media = float(d.sum() / max(t[-1] - t[0], 1e-6))
    perfil = []
    for a in np.arange(0.0, t[-1], 10.0):
        m = (t[1:] >= a) & (t[1:] < a + 10.0)
        if m.any():
            perfil.append(float(d[m].sum() / dt[m].sum()))
    return media, perfil


def avisar_velocidad_guia(por_config_filas):
    """Imprime la velocidad del guía de cada configuración y avisa si dos
    configuraciones distintas tienen el mismo perfil (configuración no
    aplicada) o si C1 (velocidad constante) fluctúa."""
    print('\nVelocidad real del guía (verdad del terreno):')
    perfiles = {}
    for config, filas in sorted(por_config_filas.items()):
        for ep in sorted({f['episodio'] for f in filas}, key=int):
            media, perfil = velocidad_guia([f for f in filas if f['episodio'] == ep])
            if media is None:
                continue
            perfiles.setdefault(config, []).append((media, perfil))
            variacion = (max(perfil) - min(perfil)) if len(perfil) > 1 else 0.0
            print(f'  {config} ep {ep}: media {media:.3f} m/s   por 10 s: '
                  + ' '.join(f'{v:.2f}' for v in perfil)
                  + f'   (rango {variacion:.2f})')
    # Solo se compara C1 con las demás: C2 y C3 comparten rango (10-40) y su
    # media puede coincidir aunque la configuración esté bien aplicada.
    if 'C1' in perfiles:
        a = 'C1'
        for b in sorted(perfiles):
            if b == a:
                continue
            ma, mb = perfiles[a][0][0], perfiles[b][0][0]
            if abs(ma - mb) < 0.01:
                print(f'  AVISO: {a} y {b} tienen la misma velocidad media del guía '
                      f'({ma:.3f} vs {mb:.3f} m/s). ¿Se aplicó la configuración del guía?')
    if 'C1' in perfiles:
        perfil = perfiles['C1'][0][1]
        if len(perfil) > 1 and max(perfil) - min(perfil) > 0.03:
            print('  AVISO: en C1 (10-10, constante) la velocidad del guía fluctúa. '
                  '¿Se aplicó la configuración del guía?')


# =====================================================================
# Métricas (siempre a partir de los CSV por paso)
# =====================================================================

def leer_csv(ruta):
    with open(ruta, newline='') as f:
        return list(csv.DictReader(f))


def metricas(filas):
    """Las cuatro métricas del anexo B sobre un conjunto de filas por paso."""
    p = PROTOCOLO
    episodios = {}
    for fila in filas:
        episodios.setdefault((fila['config'], fila['episodio']), []).append(fila)
    desenlaces = {}
    for pasos in episodios.values():
        d = pasos[-1]['desenlace'] or 'sin_terminar'
        desenlaces[d] = desenlaces.get(d, 0) + 1
    n = len(episodios)
    sep = np.array([float(f['separacion_m']) for f in filas]) if filas else np.zeros(0)
    desv = np.abs(np.array([float(f['desviacion_deg']) for f in filas])) if filas else np.zeros(0)
    completados = desenlaces.get('completado', 0)
    return {
        'episodios': n,
        'completados': completados,
        'pct_completados': 100.0 * completados / n if n else 0.0,
        'desenlaces': desenlaces,
        'pasos': len(filas),
        'pct_en_banda': 100.0 * float(np.mean([f['en_banda'] == '1' for f in filas])) if filas else 0.0,
        'separacion_media_m': float(sep.mean()) if len(sep) else float('nan'),
        'desviacion_media_deg': float(desv.mean()) if len(desv) else float('nan'),
        'pct_desv_bajo_umbral': 100.0 * float(np.mean(desv < p['DESV_UMBRAL_DEG'])) if len(desv) else 0.0,
    }


def comprobar_calibracion(filas):
    """Correlación entre la desviación real y la posx del blob (anexo B.1).

    Deben tener el mismo signo y estar muy correlacionadas: si no, la x del
    blob no mide lo que se cree (calibración o convenio de signos).
    """
    pares = [(float(f['desviacion_deg']), float(f['blob_posx']))
             for f in filas if f['visible'] == '1']
    if len(pares) < 10:
        return None
    d, x = np.array(pares).T
    if d.std() == 0 or x.std() == 0:
        return None
    return float(np.corrcoef(d, x)[0, 1]), len(pares)


def imprimir_tabla(titulo, por_config, total):
    print('\n' + titulo)
    print('=' * len(titulo))
    cab = (f'{"":8s}{"episodios":>10s}{"complet.":>10s}{"colisión":>10s}'
           f'{"perd.dist":>10s}{"perd.vista":>11s}{"% banda":>9s}'
           f'{"sep. media":>11s}{"|desv| °":>10s}{"% <15°":>8s}')
    print(cab)
    for nombre, m in [*por_config.items(), ('TOTAL', total)]:
        d = m['desenlaces']
        print(f'{nombre:8s}{m["episodios"]:>10d}'
              f'{m["completados"]:>5d} ({m["pct_completados"]:3.0f}%)'
              f'{d.get("colision", 0):>9d}{d.get("perdido_distancia", 0):>10d}'
              f'{d.get("perdido_sin_ver", 0):>11d}{m["pct_en_banda"]:>8.1f}%'
              f'{m["separacion_media_m"]:>9.3f} m{m["desviacion_media_deg"]:>10.1f}'
              f'{m["pct_desv_bajo_umbral"]:>7.1f}%')
    print('\nCausas: colisión (< 0,12 m), guía perdido por distancia (> 0,90 m) o '
          'por pelota sin detectar > 3 s.')


def resumen(etiqueta):
    rutas = sorted(glob.glob(os.path.join(RUTA_LOGS, f'eval_{etiqueta}_C*.csv')))
    if not rutas:
        raise SystemExit(f'No hay registros {RUTA_LOGS}/eval_{etiqueta}_C*.csv')
    por_config, filas_config, todas = {}, {}, []
    for ruta in rutas:
        filas = leer_csv(ruta)
        if filas:
            por_config[filas[0]['config']] = metricas(filas)
            filas_config[filas[0]['config']] = filas
            todas.extend(filas)
    imprimir_tabla(f'ANEXO B — {etiqueta}  ({", ".join(rutas)})',
                   por_config, metricas(todas))
    avisar_velocidad_guia(filas_config)
    cal = comprobar_calibracion(todas)
    if cal is not None:
        # Primera evaluación (PPO s0, C3): r = -0,91. El signo es negativo
        # porque la x del blob crece hacia el lado contrario al de la
        # desviación; lo que comprueba la calibración es que |r| sea alto.
        print(f'Correlación desviación real <-> posx del blob: {cal[0]:+.3f} '
              f'({cal[1]} pasos con pelota visible). |r| debe ser alto '
              '(signo esperado: negativo).')


# =====================================================================
# Línea de órdenes
# =====================================================================

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
    a.add_argument('--resumen', action='store_true',
                   help='sólo rehace las tablas a partir de los CSV guardados')
    args = a.parse_args()

    if args.resumen:
        etiqueta = args.etiqueta or (
            'aleatorio' if args.algoritmo == 'aleatorio'
            else os.path.basename(args.modelo or os.path.join(RUTA_MODELOS, f'p1_modelo_{args.algoritmo}_s{args.semilla}'))
            .replace('.zip', '').replace('p1_modelo_', ''))
        resumen(etiqueta)
        return

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

        resumen(etiqueta)
    finally:
        if verdad is not None:
            verdad.cerrar()
        env.close()


if __name__ == '__main__':
    main()
