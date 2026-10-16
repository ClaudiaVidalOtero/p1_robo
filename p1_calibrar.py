#!/usr/bin/env python3
"""
Calibración MANUAL de la pelota verde (sólo size y posx).

Cada ejecución mide UNA distancia real, que das en el comando, y termina:

    python3 p1_calibrar.py --distancia 0.25

El robot NO avanza ni retrocede (tilt fijo, 90 por defecto):

  1. Pones la pelota a esa distancia y pulsas Enter ('v' = ver el size en
     vivo mientras la colocas).
  2. El script toma N lecturas en reposo, te dice el size (mediana, std,
     x, y) y lo guarda en config/calibracion.json ligado a esa distancia.
  3. Barrido de posx (ver abajo). Fin. Para otra distancia, vuelves a
     lanzar el comando.

Después de medir el size, SIN pedir nada más (la pelota ya está colocada),
el robot hace el barrido: gira sobre sí mismo a derecha e izquierda hasta
perder la pelota (o --max-pulsos), vuelve al centro y registra la x en
cada pulso. --sin-barrido lo omite; --solo-barrido se salta la medida de
size y hace sólo el barrido.

El JSON (--json, por defecto config/calibracion.json) es una LISTA de medidas.
Cada ejecución AÑADE una entrada al final, sin borrar las anteriores:

    [
      {
        "distancia": 0.2,
        "fecha": "2026-10-12 10:30:00",
        "lecturas": [
          {"etapa": "centro", "size": 55, "posx": 45}
        ],
        "media_de_blobsize": 57.5
      },
      {
        "distancia": 0.25,
        ...
      }
    ]

Si el fichero existía con el formato antiguo (un único diccionario), se
conserva como primera entrada de la lista.

    python3 -c "from p1_robobo_env import importar_calibracion as i; i()"

Ejemplos:
    python3 p1_calibrar.py --distancia 0.2          # size + barrido
    python3 p1_calibrar.py --distancia 0.375 --sin-barrido
    python3 p1_calibrar.py --distancia 0.375 --solo-barrido
    python3 p1_calibrar.py --distancia 0.5 --tilt 90 --n-medidas 30

Nodo puente necesario:
    ros2 run robobo_ros2 robobo_container --ros-args \
         -p ip:=host.docker.internal -p modules:="['blob','sim']"
(el módulo sim es opcional: si está, el yaw sale de la pose del propio
robot; si no, se integra la velocidad angular ordenada.)
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from p1_robobo_env import (
    COLOR_BLOB,
    ESPERA_MAX_S,
    GIRO_S,
    HAY_INTERFACES_BLOB,
    HAY_INTERFACES_SIM,
    NS_BASE,
    NS_SIM,
    NS_SMARTPHONE,
    RUTA_CALIBRACION,
    W_GIRO,
    BlobArray,
    MoveTilt,
    RobotLocation,
    SetActiveColorBlobs,
    dif_ang,
)

TILT_DEFECTO = 90.0
N_MEDIDAS = 20       # lecturas por distancia (a 5 Hz son ~4 s)
MAX_PULSOS = 12      # tope de pulsos de giro hacia cada lado
OBLIGATORIAS = ('0.2', '0.25', '0.5')


def leer_distancia(texto):
    """'0.25' o '0,25' -> 0.25 (argparse type)."""
    try:
        d = float(str(texto).strip().replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError(f'"{texto}" no es un número')
    if d <= 0:
        raise argparse.ArgumentTypeError('la distancia debe ser > 0')
    return d


def cargar_previas(json_path):
    """Devuelve la lista de medidas ya guardadas en json_path ([] si no hay).

    - Fichero inexistente -> [].
    - Formato antiguo (un dict suelto) -> [ese dict].
    - Fichero ilegible -> se aparta como <json>.corrupto para no perderlo.
    """
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path) as fh:
            datos = json.load(fh)
    except (OSError, ValueError):
        copia = json_path + '.corrupto'
        os.replace(json_path, copia)
        print(f'AVISO: {json_path} no se pudo leer; se movió a {copia} y se empieza '
              'uno nuevo.')
        return []
    if isinstance(datos, list):
        return datos
    if isinstance(datos, dict):
        return [datos]
    return []


def calibrar_manual(distancia, tilt=TILT_DEFECTO, n_medidas=N_MEDIDAS,
                    max_pulsos=MAX_PULSOS, tamanos=True, barrido=True,
                    json_path=RUTA_CALIBRACION):
    if not HAY_INTERFACES_BLOB:
        raise RuntimeError('No se encuentran las interfaces de blobs '
                           '(robobo_ros2_interfaces). Revisa los imports.')
    # JSON deliberadamente sencillo: una distancia y sus lecturas.
    resumen = {
        'distancia': float(distancia),
        'lecturas': [],
        'media_de_blobsize': 0.0,
    }

    if not rclpy.ok():
        rclpy.init()
    nodo = Node('p1_calibracion_manual')

    est = {'blob': None, 'n': 0, 'yaw': None}
    giro = [0.0]                     # yaw integrado si no hay simulador

    # ------------------------------------------------------------ ROS
    def cb_blobs(m):
        v = [b for b in m.blobs if b.color == COLOR_BLOB]
        est['blob'] = max(v, key=lambda b: b.size) if v else None
        est['n'] += 1

    nodo.create_subscription(
        BlobArray, NS_SMARTPHONE + '/color_blobs', cb_blobs, 1)

    if HAY_INTERFACES_SIM:
        nodo.create_subscription(
            RobotLocation, NS_SIM + '/robot_location',
            lambda m: est.__setitem__('yaw', m.rotation.y), 1)

    pub = nodo.create_publisher(Twist, NS_BASE + '/cmd_vel', 10)

    def servicio(tipo, nombre, **campos):
        """Llama al servicio y falla explícitamente si no responde."""
        cli = nodo.create_client(tipo, nombre)
        print(f'Esperando servicio {nombre}...', flush=True)
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                f'Servicio {nombre} no disponible. Comprueba que el nodo puente '
                'robobo_container está activo y que los nombres de servicio '
                'coinciden con tu versión de robobo_ros2.')
        req = tipo.Request()
        for k, v in campos.items():
            setattr(req, k, v)
        fut = cli.call_async(req)
        limite = time.monotonic() + 10.0
        while rclpy.ok() and not fut.done() and time.monotonic() < limite:
            rclpy.spin_once(nodo, timeout_sec=0.1)
        if not fut.done():
            raise RuntimeError(
                f'Timeout esperando respuesta de {nombre}.')
        exc = fut.exception()
        if exc is not None:
            raise RuntimeError(
                f'Error llamando a {nombre}: {exc}')
        respuesta = fut.result()
        print(f'Respuesta de {nombre}: {respuesta}', flush=True)
        return respuesta

    def lectura_nueva(n_msgs=2):
        objetivo = est['n'] + n_msgs
        t0 = time.time()
        while est['n'] < objetivo:
            rclpy.spin_once(nodo, timeout_sec=0.05)
            if time.time() - t0 > ESPERA_MAX_S:
                raise RuntimeError(
                    f'No llegan blobs de {NS_SMARTPHONE}/color_blobs. ¿robobo_container '
                    'con el módulo blob?')

    def mover(v, w, segundos):
        msg = Twist()
        msg.linear.x, msg.angular.z = float(v), float(w)
        t0 = time.time()
        while time.time() - t0 < segundos:
            pub.publish(msg)
            time.sleep(0.1)
        msg.linear.x = msg.angular.z = 0.0
        for _ in range(3):
            pub.publish(msg)
            time.sleep(0.05)

    def girar(sg):
        mover(0.0, sg * W_GIRO, GIRO_S)
        giro[0] += math.degrees(sg * W_GIRO * GIRO_S)

    def yaw():
        return est['yaw'] if est['yaw'] is not None else giro[0]

    def leer():
        """Un mensaje fresco -> dict con size, x, y, yaw."""
        lectura_nueva(2)             # el primero puede ser anterior a la espera
        b = est['blob']
        return {'size': b.size if b else 0,
                'x': b.x if b else None,
                'y': b.y if b else None,
                'yaw': yaw()}

    def fmt(v, spec, ancho):
        return (format(v, spec) if v is not None else '--').rjust(ancho)

    def guardar():
        visibles = [r['size'] for r in resumen['lecturas'] if r['size'] > 0]
        resumen['media_de_blobsize'] = (
            statistics.mean(visibles) if visibles else 0.0)
        resumen['fecha'] = time.strftime('%Y-%m-%d %H:%M:%S')

        historial = cargar_previas(json_path)
        historial.append(resumen)
        # Escritura atómica: si se corta a mitad, no se pierde lo anterior.
        tmp = json_path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(historial, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, json_path)
        print(f'Medida añadida a {json_path} (total: {len(historial)}).')

    # ------------------------------------------------------------ SIZE
    def medir_size(n):
        lectura_nueva(2)             # descarta lo que hubiera en cola
        m = []
        for _ in range(n):
            lectura_nueva(1)
            b = est['blob']
            m.append({'size': b.size if b else 0,
                      'x': b.x if b else None,
                      'y': b.y if b else None})
        vis = [d for d in m if d['size'] > 0]
        r = {'fecha': time.strftime('%Y-%m-%d %H:%M:%S'), 'tilt': tilt,
             'n_lecturas': n, 'visibles': len(vis)}
        if vis:
            sz = [d['size'] for d in vis]
            r.update({
                'size_mediana': statistics.median(sz),
                'size_media': statistics.mean(sz),
                'size_std': statistics.pstdev(sz),
                'size_min': min(sz), 'size_max': max(sz),
                'x_mediana': statistics.median(d['x'] for d in vis),
                'y_mediana': statistics.median(d['y'] for d in vis)})
        r['lecturas'] = m
        return r

    def ver_en_vivo():
        print('   (size en vivo, Ctrl+C para volver)')
        try:
            while True:
                d = leer()
                print('\r   size={}  x={}  y={}      '.format(
                    fmt(d['size'], '.6g', 8), fmt(d['x'], '.6g', 7),
                    fmt(d['y'], '.6g', 7)), end='', flush=True)
        except KeyboardInterrupt:
            print()

    def pedir_colocacion():
        while True:
            op = input(f'Coloca la pelota centrada a {distancia:.3f} m y pulsa Enter '
                       '(v = ver size en vivo): ')
            if op.strip().lower() == 'v':
                ver_en_vivo()
                continue
            return

    def etapa_tamanos():
        print(f'\n=== SIZE a {distancia:.3f} m (el robot no se mueve) ==='
              )
        pedir_colocacion()
        while True:
            r = medir_size(n_medidas)
            if r['visibles']:
                break
            input('   No se ve la pelota. Recoloca y pulsa Enter para '
                  'repetir... ')
        # Las lecturas en reposo se guardan (etapa 'reposo') con su y: la
        # altura de la pelota en la imagen distingue "pegado" (pelota cortada
        # por abajo) de "lejos", que dan un size parecido.
        resumen['lecturas'].extend(
            {'etapa': 'reposo', 'size': d['size'], 'posx': d['x'], 'y': d['y']}
            for d in r['lecturas'])
        print('   size = {:.6g} (mediana)   media {:.6g}   std {:.3g}   '
              'rango [{:.6g}, {:.6g}]\n'
              '   x = {:.1f}   y = {:.1f}   visible {}/{}'.format(
                  r['size_mediana'], r['size_media'], r['size_std'],
                  r['size_min'], r['size_max'], r['x_mediana'],
                  r['y_mediana'], r['visibles'], r['n_lecturas']))

        # Conserva el resumen de tamaño en consola; el JSON sencillo guarda
        # las lecturas de posx del barrido.

    # ------------------------------------------------------------ BARRIDO
    def etapa_barrido(ya_colocada):
        print(f'\n=== BARRIDO lateral de posx a {distancia:.3f} m (giro sobre el '
              'sitio) ===' )
        if not ya_colocada:
            pedir_colocacion()
        print('{:<8s}{:>10s}{:>10s}{:>10s}{:>9s}'.format(
            'etapa', 'size', 'x', 'y', 'yaw(°)'))
        lecturas = []

        def registrar(etapa, d):
            lecturas.append({
                'etapa': etapa,
                'size': d['size'],
                'posx': d['x'],
                'y': d['y'],
            })
            print('{:<8s}{}{}{}{}'.format(
                etapa, fmt(d['size'], '.6g', 10), fmt(d['x'], '.6g', 10),
                fmt(d['y'], '.6g', 10), fmt(d['yaw'], '.1f', 9)))

        c0 = leer()
        registrar('centro', c0)
        if c0['size'] == 0:
            print('No se detecta el blob en el centro; no se inicia el giro '
                  'para evitar barrer sin referencia. Comprueba que la pelota '
                  'está visible, que el color está activado y que el tilt '
                  'se ha colocado correctamente. Repite la ejecución.')
            return
        centros, lados, vis_all = [c0], {}, [c0]

        for nombre, sg in (('der', -1), ('izq', +1)):
            dados, ult = 0, None
            for i in range(1, max_pulsos + 1):
                girar(sg)
                dados += 1
                d = leer()
                registrar(f'{nombre}{i}', d)
                if d['size'] == 0:
                    break
                ult = d
                vis_all.append(d)
            for _ in range(dados):
                girar(-sg)
            c = leer()
            registrar('centro', c)
            centros.append(c)
            lados[nombre] = (ult, dados)

        vis_c = [c for c in centros if c['size'] > 0]
        xs = [d['x'] for d in vis_all]
        x_centro = statistics.mean(c['x'] for c in vis_c)
        b = {'distancia_m': float(distancia),
             'fecha': time.strftime('%Y-%m-%d %H:%M:%S'), 'tilt': tilt,
             'x_centro': x_centro,
             'x_min_visto': min(xs), 'x_max_visto': max(xs),
             'x_semiancho_sugerido': max(abs(min(xs) - x_centro),
                                         abs(max(xs) - x_centro)),
             'size_barrido_mediana': statistics.median(
                 d['size'] for d in vis_all)}
        if centros[-1]['size'] > 0:
            b['deriva_x_al_volver'] = centros[-1]['x'] - c0['x']
        for nombre, (ult, dados) in lados.items():
            dy = dif_ang(ult['yaw'], c0['yaw']) if ult else 0.0
            b['x_por_grado_' + nombre] = (
                (ult['x'] - c0['x']) / dy if ult and abs(dy) > 1e-6 else None)
            b['grados_hasta_perder_' + nombre] = abs(dy) if ult else None
            b['pulsos_' + nombre] = dados
        b['fuente_yaw'] = ('simulador' if est['yaw'] is not None
                           else 'integrado (w*t)')
        print('\n' + json.dumps(b, indent=2))
        resumen['lecturas'].extend(lecturas)
        guardar()

    # ------------------------------------------------------------ main
    try:
        # Coloca primero la cámara en el tilt solicitado (90° por defecto)
        # y espera a que termine antes de activar/buscar blobs.
        print(f'Solicitando tilt {tilt:.1f}° antes de activar/buscar blobs...'
              , flush=True)
        servicio(MoveTilt, NS_BASE + '/move_tilt',
                 angle=float(tilt), speed=30.0)
        time.sleep(2.0)
        servicio(SetActiveColorBlobs,
                 NS_SMARTPHONE + '/set_active_color_blobs',
                 red=False, green=True, blue=False, custom=False)
        time.sleep(1.0)
        print(f'Tilt solicitado: {tilt:.1f}°. '
              f'Distancia real: {distancia:.3f} m.', flush=True)

        if tamanos:
            etapa_tamanos()
        if barrido:
            etapa_barrido(ya_colocada=tamanos)   # guarda al terminar
        elif tamanos:
            guardar()   # --sin-barrido: antes no se guardaba nada
        print('\nListo. Siguiente: otra distancia, o importar con\n'
              '  python3 -c "from p1_robobo_env import importar_calibracion '
              'as i; i()"')
    except KeyboardInterrupt:
        print(f'\nInterrumpido (lo ya guardado en {json_path} se conserva).')
    finally:
        # Parada de seguridad best-effort: no debe ocultar el error original.
        with contextlib.suppress(Exception):
            mover(0.0, 0.0, 0.1)
        nodo.destroy_node()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--distancia', '-d', type=leer_distancia, required=True,
                    help='distancia real de la pelota al robot, en m')
    ap.add_argument('--tilt', type=float, default=TILT_DEFECTO)
    ap.add_argument('--n-medidas', type=int, default=N_MEDIDAS)
    ap.add_argument('--max-pulsos', type=int, default=MAX_PULSOS)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--sin-barrido', action='store_true',
                   help='medir sólo el size, sin barrido de posx')
    g.add_argument('--solo-barrido', action='store_true',
                   help='hacer sólo el barrido (sin medir size)')
    ap.add_argument('--json', default=RUTA_CALIBRACION)
    a = ap.parse_args()
    calibrar_manual(a.distancia, tilt=a.tilt, n_medidas=a.n_medidas,
                    max_pulsos=a.max_pulsos,
                    tamanos=not a.solo_barrido,
                    barrido=not a.sin_barrido,
                    json_path=a.json)
