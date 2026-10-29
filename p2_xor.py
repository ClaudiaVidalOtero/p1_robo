#!/usr/bin/env python3
"""
Práctica 2 — Ejemplo XOR de NEAT-Python (comprobación de la imagen).

Es el ejemplo evolve-feedforward de NEAT-Python con su configuración
(config-feedforward), que se escribe en un archivo temporal. Evoluciona una
red prealimentada de 2 entradas y 1 salida (sigmoid) hasta que reproduce la
tabla del XOR. Calidad = 4 - suma de los errores al cuadrado en los 4 casos
(máximo 4); se para al llegar a fitness_threshold = 3,9.

No usa ROS ni el simulador: sirve para comprobar que NEAT-Python 2.0 está
instalado y funciona dentro del contenedor.

USO
    python3 p2_xor.py
    python3 p2_xor.py --semilla 3
"""

import argparse
import os
import tempfile

import neat

CONFIG = """
[NEAT]
fitness_criterion      = max
fitness_threshold      = 3.9
pop_size               = 150
reset_on_extinction    = False
no_fitness_termination = False

[DefaultGenome]
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.5
enabled_default         = True
enabled_mutate_rate     = 0.01
feed_forward            = True
initial_connection      = full
node_add_prob           = 0.2
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = 2
num_outputs             = 1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
min_species_size   = 2
"""

ENTRADAS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
SALIDAS = [0.0, 1.0, 1.0, 0.0]


def eval_genomes(genomas, config):
    for _, genoma in genomas:
        red = neat.nn.FeedForwardNetwork.create(genoma, config)
        genoma.fitness = 4.0 - sum((red.activate(x)[0] - y) ** 2
                                   for x, y in zip(ENTRADAS, SALIDAS))


def main():
    a = argparse.ArgumentParser(description='Ejemplo XOR de NEAT-Python.')
    a.add_argument('--semilla', type=int, default=1)
    a.add_argument('--generaciones', type=int, default=300)
    args = a.parse_args()

    ruta = os.path.join(tempfile.mkdtemp(), 'config-feedforward')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(CONFIG)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, ruta)

    poblacion = neat.Population(config, seed=args.semilla)
    poblacion.add_reporter(neat.StdOutReporter(False))
    ganador = poblacion.run(eval_genomes, args.generaciones)

    red = neat.nn.FeedForwardNetwork.create(ganador, config)
    print(f'\nGanador: calidad {ganador.fitness:.3f} de 4, '
          f'(nodos, conexiones) = {ganador.size()}')
    for x, y in zip(ENTRADAS, SALIDAS):
        print(f'  entrada {x} -> esperado {y:.0f}, obtenido {red.activate(x)[0]:.3f}')


if __name__ == '__main__':
    main()
