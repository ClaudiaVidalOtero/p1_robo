#!/usr/bin/env python3
"""
Práctica 2 — Evaluador del anexo B (C1-C4) para cualquier controlador.

Es el evaluador de la P1 (p1_evaluar.episodio: mismos criterios de fin y
mismo registro por paso) ampliado con C4 (variable con paradas, 2
ejecuciones). El controlador es una función obs -> acción, así que se evalúan
igual la red de NEAT, la política de SB3 de la P1 (comparación del apartado
6.5), el controlador del reto y la línea base aleatoria.

La configuración del guía se pone a mano en RoboboSim; el script espera a que
se pulse Enter antes de cada una. Mismo multiplicador y físicas que en la
evolución (y que en la P1 para la comparación).

Escribe p2_logs/eval_<etiqueta>_<C>.csv con las columnas de la P1. Las
paradas del guía de C4 se calculan después a partir de su pose registrada.

ROBOT REAL (--real, apartado 6.6): sin simulador ni verdad del terreno. Antes
de cada episodio una persona coloca el seguidor detrás del guía y pulsa
Enter. El episodio acaba a los 120 s, con la pelota sin verse más de 3 s
(criterio del anexo B que sí se mide a bordo) o con Ctrl-C, tras el cual se
pregunta la causa (colisión, guía perdido u otra) vista por el operador. Se
registra por paso el blob, la acción y el choque/pérdida que estima el
entorno, en p2_logs/real_<etiqueta>_<C>.csv. Usar --config con la
calibración del robot real.

USO (puente lanzado con modules:="['blob']")
    python3 p2_evaluar.py --neat p2_evol/s0/mejor_genoma.pkl --etiqueta neat_s0
    python3 p2_evaluar.py --sb3 sac --modelo P1/modelos/p1_modelo_sac_s0.zip --configs C1 C2 C3
    python3 p2_evaluar.py --reto
    python3 p2_evaluar.py --aleatorio
    python3 p2_evaluar.py --real --neat p2_evol/s0/mejor_genoma.pkl --etiqueta neat_s0 \\
        --configs C1 --episodios 3 --config config/p1_config_real.json
"""

import argparse
import csv
import os
import time

import numpy as np

from p1_evaluar import VerdadTerreno, episodio
from p1_graficas_eval import COLUMNAS, PROTOCOLO
from p1_robobo_env import RUTA_CONFIG, RoboboSeguidorEnv, cargar_config
from p2_controladores import (
    RUTA_CONFIG_P2,
    cargar_config_p2,
    controlador_aleatorio,
    controlador_neat,
    controlador_reto,
    controlador_sb3,
)

RUTA_LOGS_P2 = 'p2_logs'

CONFIGURACIONES = {
    'C1': 'Lenta constante   modo Crucero              velocidad 10 - 10',
    'C2': 'Contraste         modo Crucero              velocidad 10 - 40',
    'C3': 'Fluctuante        modo Variable             velocidad 10 - 40',
    'C4': 'Con paradas       modo Variable con paradas velocidad 10 - 40',
}
EJECUCIONES = {'C1': 1, 'C2': 1, 'C3': 1, 'C4': 2}

# Registro por paso en el robot real: sólo lo que se mide a bordo.
COLUMNAS_REAL = ['config', 'episodio', 'paso', 't', 'visible', 'blob_posx',
                 'blob_size', 'blob_y', 't_sin_ver', 'en_banda_size',
                 'a0', 'a1', 'v', 'w', 'entorno_choque', 'entorno_perdido',
                 'desenlace']
CAUSAS_OPERADOR = {'c': 'colision', 'p': 'perdido_operador', 'o': 'otra'}


def construir_controlador(args, env, c_env):
    """Devuelve (función obs -> acción, etiqueta)."""
    if args.neat:
        ini = args.neat_ini or os.path.join(os.path.dirname(args.neat), 'p2_neat.ini')
        nombre = os.path.basename(os.path.dirname(args.neat)) or 'red'
        return controlador_neat(args.neat, ini), args.etiqueta or f'neat_{nombre}'
    if args.sb3:
        if not args.modelo:
            raise SystemExit('--sb3 necesita --modelo')
        nombre = os.path.basename(args.modelo).removesuffix('.zip').replace('p1_modelo_', '')
        return controlador_sb3(args.sb3, args.modelo), args.etiqueta or f'p1_{nombre}'
    if args.reto:
        r = cargar_config_p2(args.config_p2)['RETO']
        busqueda = r.get('BUSQUEDA') if args.busqueda else None
        return (controlador_reto(c_env, r['K_GIRO'], r['K_VEL'], busqueda),
                args.etiqueta or ('reto_busqueda' if busqueda else 'reto'))
    return controlador_aleatorio(env), args.etiqueta or 'aleatorio'


def episodio_real(env, politica, config, num, escritor, args):
    """Un episodio en el robot real con los criterios del anexo B medibles a bordo."""
    p = PROTOCOLO
    obs, _ = env.reset()           # pide recolocar el seguidor y pulsar Enter
    t0 = t_visto = time.time()
    paso, desenlace, t = 0, None, 0.0
    print(f'\n{config} episodio {num} (Ctrl-C si choca o se pierde a la vista del operador)')
    try:
        while desenlace is None:
            accion = np.asarray(politica(obs), dtype=np.float32)
            obs, _, term_env, _, info = env.step(accion)
            ahora = time.time()
            paso += 1
            t = ahora - t0
            visible = bool(info.get('visible'))
            if visible:
                t_visto = ahora
            if ahora - t_visto > p['SIN_VER_S']:
                desenlace = 'perdido_sin_ver'
            elif t >= p['DURACION_S']:
                desenlace = 'completado'
            escritor.writerow({
                'config': config, 'episodio': num, 'paso': paso, 't': f'{t:.3f}',
                'visible': int(visible), 'blob_posx': f'{info.get("posx", 0.0):.4f}',
                'blob_size': f'{info.get("size_raw", 0.0):.1f}',
                'blob_y': f'{info.get("y_raw", 0.0):.1f}',
                't_sin_ver': f'{ahora - t_visto:.2f}',
                'en_banda_size': int(bool(info.get('en_banda'))),
                'a0': f'{float(accion[0]):.4f}', 'a1': f'{float(accion[1]):.4f}',
                'v': f'{info.get("v", 0.0):.4f}', 'w': f'{info.get("w", 0.0):.4f}',
                'entorno_choque': int(bool(term_env and info.get('choque'))),
                'entorno_perdido': int(bool(term_env and info.get('perdido'))),
                'desenlace': desenlace or '',
            })
            if args.detalle:
                print(f'  t {t:6.2f}  blob {"sí" if visible else "no"} '
                      f'posx {info.get("posx", 0.0):+.2f} size {info.get("size_raw", 0.0):6.1f}  '
                      f'v {info.get("v", 0.0):.2f} w {info.get("w", 0.0):+.2f}'
                      + (f'  -> {desenlace}' if desenlace else ''))
    except KeyboardInterrupt:
        env._detener()
        r = ''
        while r not in CAUSAS_OPERADOR:
            r = input('\nCausa (c = colisión, p = guía perdido, o = otra): ').strip().lower()
        desenlace = CAUSAS_OPERADOR[r]
        # La última fila registrada lleva el desenlace del operador.
        escritor.writerow({'config': config, 'episodio': num, 'paso': paso + 1,
                           't': f'{time.time() - t0:.3f}', 'desenlace': desenlace})
    env._detener()
    print(f'  {desenlace} a los {t:.1f} s ({paso} pasos)')
    return desenlace


def main():
    a = argparse.ArgumentParser(description='Evaluador del anexo B (C1-C4), Práctica 2.')
    ctrl = a.add_mutually_exclusive_group(required=True)
    ctrl.add_argument('--neat', metavar='GENOMA.pkl', help='genoma de NEAT')
    ctrl.add_argument('--sb3', choices=['sac', 'ppo'], help='política de la P1')
    ctrl.add_argument('--reto', action='store_true', help='controlador programado a mano')
    ctrl.add_argument('--aleatorio', action='store_true')
    a.add_argument('--neat-ini', default=None,
                   help='configuración de NEAT (por defecto, la de la carpeta del genoma)')
    a.add_argument('--modelo', default=None, help='.zip de SB3')
    a.add_argument('--etiqueta', default=None)
    a.add_argument('--configs', nargs='+', choices=sorted(CONFIGURACIONES),
                   default=sorted(CONFIGURACIONES))
    a.add_argument('--config', default=RUTA_CONFIG, help='configuración del entorno (P1)')
    a.add_argument('--config-p2', default=RUTA_CONFIG_P2)
    a.add_argument('--detalle', action='store_true', help='una línea por paso')
    a.add_argument('--busqueda', action='store_true',
                   help='--reto: variante que, sin pelota, gira hacia el último posx visto')
    a.add_argument('--real', action='store_true',
                   help='robot real: sin simulador, reinicio manual (apartado 6.6)')
    a.add_argument('--episodios', type=int, default=None,
                   help='episodios por configuración (por defecto, los del anexo B)')
    args = a.parse_args()

    c_env = cargar_config(args.config)
    # Una red de NEAT se evalúa con la observación con la que evolucionó: el
    # bloque ENTORNO (OBS_MEMORIA) de la p2_config.json de su carpeta. La
    # política de la P1, el reto y el aleatorio usan la observación de la P1.
    ajustes = None
    if args.neat:
        ruta_p2 = os.path.join(os.path.dirname(args.neat), 'p2_config.json')
        ajustes = cargar_config_p2(ruta_p2 if os.path.exists(ruta_p2)
                                   else args.config_p2).get('ENTORNO')
    env = RoboboSeguidorEnv(pasos_max=10**9, usar_simulador=not args.real,
                            ruta_config=args.config, ajustes=ajustes)
    verdad = None
    try:
        politica, etiqueta = construir_controlador(args, env, c_env)
        if env.observation_space.shape[0] != 3 and args.sb3:
            print('Aviso: la observación no tiene 3 componentes; revisa OBS_Y / OBS_IR.')
        os.makedirs(RUTA_LOGS_P2, exist_ok=True)
        if args.real:
            print(f'Evaluando "{etiqueta}" en el ROBOT REAL con {args.config}.')
        else:
            verdad = VerdadTerreno(c_env['SIM_IP'])
            sim = cargar_config_p2(args.config_p2).get('SIM', {})
            print(f'Evaluando "{etiqueta}". Simulador: multiplicador x{sim.get("MULTIPLICADOR")}, '
                  f'físicas {sim.get("FISICAS")} (igual que en la evolución).')

        for config in args.configs:
            input(f'\nPon el guía en {config}: {CONFIGURACIONES[config]}\n'
                  'y pulsa Enter... ')
            prefijo = 'real' if args.real else 'eval'
            ruta = os.path.join(RUTA_LOGS_P2, f'{prefijo}_{etiqueta}_{config}.csv')
            n = args.episodios or EJECUCIONES[config]
            with open(ruta, 'w', newline='', encoding='utf-8') as f:
                escritor = csv.DictWriter(f, fieldnames=COLUMNAS_REAL if args.real else COLUMNAS)
                escritor.writeheader()
                for num in range(1, n + 1):
                    if args.real:
                        episodio_real(env, politica, config, num, escritor, args)
                    else:
                        episodio(env, verdad, politica, config, num, escritor, args)
                    f.flush()
            print(f'registro por paso: {ruta}')
    finally:
        if verdad is not None:
            verdad.cerrar()
        env.close()


if __name__ == '__main__':
    main()
