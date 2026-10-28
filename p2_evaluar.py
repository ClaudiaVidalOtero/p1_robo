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

USO (puente lanzado con modules:="['blob']")
    python3 p2_evaluar.py --neat p2_evol/s0/mejor_genoma.pkl --etiqueta neat_s0
    python3 p2_evaluar.py --sb3 sac --modelo modelos/p1_modelo_sac_s0.zip --configs C1 C2 C3
    python3 p2_evaluar.py --reto
    python3 p2_evaluar.py --aleatorio
"""

import argparse
import csv
import os

from p1_evaluar import VerdadTerreno, episodio
from p1_graficas_eval import COLUMNAS
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
        return controlador_reto(c_env, r['K_GIRO'], r['K_VEL']), args.etiqueta or 'reto'
    return controlador_aleatorio(env), args.etiqueta or 'aleatorio'


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
    args = a.parse_args()

    c_env = cargar_config(args.config)
    env = RoboboSeguidorEnv(pasos_max=10**9, usar_simulador=True, ruta_config=args.config)
    verdad = None
    try:
        politica, etiqueta = construir_controlador(args, env, c_env)
        if env.observation_space.shape[0] != 3 and not args.aleatorio:
            print('Aviso: la observación no tiene 3 componentes; revisa OBS_Y / OBS_IR.')
        verdad = VerdadTerreno(c_env['SIM_IP'])
        os.makedirs(RUTA_LOGS_P2, exist_ok=True)
        sim = cargar_config_p2(args.config_p2).get('SIM', {})
        print(f'Evaluando "{etiqueta}". Simulador: multiplicador x{sim.get("MULTIPLICADOR")}, '
              f'físicas {sim.get("FISICAS")} (igual que en la evolución).')

        for config in args.configs:
            input(f'\nPon el guía en {config}: {CONFIGURACIONES[config]}\n'
                  'y pulsa Enter... ')
            ruta = os.path.join(RUTA_LOGS_P2, f'eval_{etiqueta}_{config}.csv')
            with open(ruta, 'w', newline='', encoding='utf-8') as f:
                escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
                escritor.writeheader()
                for num in range(1, EJECUCIONES[config] + 1):
                    episodio(env, verdad, politica, config, num, escritor, args)
                    f.flush()
            print(f'registro por paso: {ruta}')
    finally:
        if verdad is not None:
            verdad.cerrar()
        env.close()


if __name__ == '__main__':
    main()
