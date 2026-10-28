#!/usr/bin/env python3
"""
Práctica 2 — Controladores como funciones observación -> acción.

Todos reciben la observación del entorno de la P1 ([posx, size, posy],
normalizada) y devuelven la acción normalizada [a0, a1] en [-1, 1]. Así el
mismo bucle sirve para evolucionar, para el evaluador del anexo B y para la
comparación con la P1 (apartado 6.5):

    red NEAT       red prealimentada construida a partir de un genoma
    SB3            política final de la P1, en modo determinista
    reto           controlador programado a mano (referencia de calidad)
    aleatorio      acciones uniformes en el espacio de acción
"""

import json
import os
import pickle

import numpy as np
import neat

RUTA_CONFIG_P2 = os.path.join('config', 'p2_config.json')
RUTA_NEAT_INI = os.path.join('config', 'p2_neat.ini')


def cargar_config_p2(ruta=RUTA_CONFIG_P2):
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)


def cargar_config_neat(ruta=RUTA_NEAT_INI):
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                       neat.DefaultSpeciesSet, neat.DefaultStagnation, ruta)


def politica_red(genoma, config_neat):
    """Red de un genoma como función obs -> acción."""
    red = neat.nn.FeedForwardNetwork.create(genoma, config_neat)
    return lambda obs: np.array(red.activate(obs), dtype=np.float32)


def controlador_neat(ruta_genoma, ruta_ini=RUTA_NEAT_INI):
    config = cargar_config_neat(ruta_ini)
    with open(ruta_genoma, 'rb') as f:
        genoma = pickle.load(f)
    return politica_red(genoma, config)


def controlador_sb3(algoritmo, ruta_modelo):
    from stable_baselines3 import PPO, SAC
    modelo = (PPO if algoritmo == 'ppo' else SAC).load(ruta_modelo.removesuffix('.zip'))
    return lambda obs: modelo.predict(obs, deterministic=True)[0]


def controlador_reto(c_env, k_giro, k_vel):
    """Controlador programado a mano, sólo con la observación.

    giro:    a1 = K_GIRO * posx (en la P1, posx y el giro de la política SAC
             tienen correlación positiva: con posx > 0 se gira con a1 > 0).
    avance:  fracción de V_MAX = K_VEL * (SIZE_OBJ - size) / SIZE_OBJ, en [0, 1]:
             se para al llegar al size objetivo y acelera si la pelota se aleja.
    Sin pelota a la vista se detiene.
    """
    size_obj = c_env['SIZE_OBJ'] / c_env['SIZE_MAX']   # en unidades de obs[1]

    def politica(obs):
        posx, size = float(obs[0]), float(obs[1])
        if size <= 0.0:
            return np.array([-1.0, 0.0], dtype=np.float32)
        frac = np.clip(k_vel * (size_obj - size) / size_obj, 0.0, 1.0)
        a0 = 2.0 * frac - 1.0       # inversa de v = V_MIN + (a0+1)/2·(V_MAX-V_MIN)
        a1 = np.clip(k_giro * posx, -1.0, 1.0)
        return np.array([a0, a1], dtype=np.float32)

    return politica


def controlador_aleatorio(env):
    return lambda obs: env.action_space.sample()
