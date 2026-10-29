#!/usr/bin/env python3
"""
Práctica 2 — Evolución con NEAT-Python del controlador del seguidor.

Cada genoma es una red prealimentada (3 entradas, 2 salidas tanh) que
controla al seguidor en el entorno de la P1 (RoboboSeguidorEnv) durante
EPISODIOS_POR_INDIVIDUO episodios de DURACION_EVAL_S segundos reales. La
calidad usa información privilegiada del simulador (VerdadTerreno de
p1_evaluar.py): la separación y la desviación reales respecto a la pelota.
La red nunca la recibe; su observación sigue siendo [posx, size, posy].

CALIDAD (config/p2_config.json, formulación A, B o A0)
    En cada paso, con Δt el tiempo real del paso:
      banda(sep)  1 dentro de [0,20, 0,45] m; fuera, K_FUERA · rampa lineal
                  que vale 1 en el borde de la banda y 0 en la colisión
                  (0,12 m) y en guía perdido (0,90 m)
      alin(desv)  max(0, 1 - |desv| / DESV_MAX_DEG)
      paso        banda·(W_BANDA + W_DESV·alin)     si ALIN_POR_BANDA
                  W_BANDA·banda + W_DESV·alin       si no (A0, descartada)
    calidad = 100 · Σ paso·Δt / (T · (W_BANDA + W_DESV))
              - P_COLISION si choca - P_PERDIDO si pierde al guía
    Rango [-P, 100]. Con K_FUERA = 0,3 y la alineación ligada a la banda,
    seguir de lejos sin entrar en ella puntúa poco (en A0, seguir a 0,57 m
    puntuaba 73). El episodio acaba con los criterios del anexo B
    (colisión, guía perdido por distancia o > 3 s sin ver) o a los T s: los
    segundos no vividos no suman (penalización implícita), y la suma por
    tiempo distingue al que falla a los 2 s del que falla a los 8 s.

La configuración del guía (C1-C4) se cambia a mano en RoboboSim: robobosim no
permite fijarla. --guia sólo la anota en los registros. Evolución incremental:
se evoluciona en C3, se para, se pone C4 en RoboboSim y se reanuda.

SALIDA en p2_evol/<nombre>/
    p2_neat.ini, p2_config.json   copia de la configuración usada (con la semilla)
    evolucion.csv                 una fila por generación (curvas y coste)
    especies.csv                  tamaño de cada especie por generación
    episodios.csv                 una fila por episodio evaluado (causas de fin)
    chk/p2-N                      punto de control al empezar la generación N
    genomas/mejor_genNNN.pkl      mejor genoma de cada generación
    mejor_genoma.pkl              elegido por --reevaluar

USO (puente lanzado con modules:="['blob']"; desde la raíz del proyecto)
    python3 p2_neat.py --comprobar                       # 1 genoma aleatorio, 1 episodio
    python3 p2_neat.py --reto --guia C3 --episodios 3    # referencia del controlador a mano
    python3 p2_neat.py --evolucionar --nombre s0 --semilla 0 --generaciones 15 --guia C3
    python3 p2_neat.py --reanudar    --nombre s0 --generaciones 15 --guia C4 [--duracion 40]
    python3 p2_neat.py --reevaluar   --nombre s0 --candidatos 5 --episodios 3 --guia C4
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import shutil
import statistics
import time

import neat

from p1_evaluar import VerdadTerreno, medir
from p1_graficas_eval import PROTOCOLO
from p1_robobo_env import RUTA_CONFIG, RoboboSeguidorEnv, cargar_config
from p2_controladores import (
    RUTA_CONFIG_P2,
    RUTA_NEAT_INI,
    cargar_config_neat,
    cargar_config_p2,
    controlador_reto,
    politica_red,
)

RUTA_EVOL = 'p2_evol'
GUIAS = ['C1', 'C2', 'C3', 'C4']

# Contadores del proceso: los actualiza evaluar_individuo() y los lee el registro.
CONTADOR = {'pasos': 0, 'evaluaciones': 0}

# Contexto de la evaluación. Fuera de los registros a propósito: el
# Checkpointer serializa los registros con pickle y el entorno no se puede.
CTX = {'env': None, 'verdad': None, 'p2': None, 'calidad': None,
       'guia': None, 'gen': 0, 'ruta_episodios': None}

CAMPOS_EPISODIO = ['generacion', 'genoma', 'episodio', 'guia', 'calidad',
                   'desenlace', 'segundos', 'pasos', 'sep_media', 'pct_banda',
                   'desv_media']


# =====================================================================
# Calidad
# =====================================================================

def puntos_banda(sep, k_fuera=1.0):
    """1 dentro de la banda; fuera, k_fuera · rampa lineal hasta 0 en los umbrales de fallo."""
    p = PROTOCOLO
    if p['BANDA_MIN_M'] <= sep <= p['BANDA_MAX_M']:
        return 1.0
    if sep < p['BANDA_MIN_M']:
        rampa = (sep - p['COLISION_M']) / (p['BANDA_MIN_M'] - p['COLISION_M'])
    else:
        rampa = 1.0 - (sep - p['BANDA_MAX_M']) / (p['PERDIDO_M'] - p['BANDA_MAX_M'])
    return k_fuera * max(0.0, rampa)


def puntos_alineacion(desv, desv_max):
    return max(0.0, 1.0 - abs(desv) / desv_max)


def puntos_paso(sep, desv, cal):
    """Puntos de un paso (sin Δt), en [0, W_BANDA + W_DESV]."""
    banda = puntos_banda(sep, cal.get('K_FUERA', 1.0))
    alin = puntos_alineacion(desv, cal['DESV_MAX_DEG'])
    if cal.get('ALIN_POR_BANDA', False):
        return banda * (cal['W_BANDA'] + cal['W_DESV'] * alin)
    return cal['W_BANDA'] * banda + cal['W_DESV'] * alin


def episodio_calidad(env, verdad, politica, cal, duracion):
    """Un episodio con fin según el anexo B. Devuelve (calidad, resumen)."""
    p = PROTOCOLO
    obs, _ = env.reset()
    verdad.leer(nueva=True)            # posiciones posteriores al reinicio
    t0 = t_prev = t_visto = time.time()
    acum, pasos, desenlace = 0.0, 0, None
    seps, desvs, en_banda = [], [], 0

    while desenlace is None:
        obs, _, _, _, info = env.step(politica(obs))
        seguidor, _, pelota = verdad.leer()
        ahora = time.time()
        # El último paso no puede sumar más allá de T: calidad <= 100.
        dt = min(ahora, t0 + duracion) - min(t_prev, t0 + duracion)
        t_prev = ahora
        pasos += 1

        sep, desv = medir(seguidor, pelota)
        if info.get('visible'):
            t_visto = ahora
        if sep < p['COLISION_M']:
            desenlace = 'colision'
        elif sep > p['PERDIDO_M']:
            desenlace = 'perdido_distancia'
        elif ahora - t_visto > p['SIN_VER_S']:
            desenlace = 'perdido_sin_ver'
        elif ahora - t0 >= duracion:
            desenlace = 'completado'

        acum += puntos_paso(sep, desv, cal) * dt
        if CTX.get('detalle'):
            print(f'    t {ahora - t0:5.2f}  sep {sep:.3f} m  desv {desv:+6.1f}°  '
                  f'seg ({seguidor[0]:7.1f}, {seguidor[1]:7.1f})  '
                  f'pelota ({pelota[0]:7.1f}, {pelota[1]:7.1f})  '
                  f'blob {"sí" if info.get("visible") else "no"} posx {info.get("posx", 0.0):+.2f} '
                  f'size {info.get("size_raw", 0.0):5.1f}  v {info.get("v", 0.0):.2f} '
                  f'w {info.get("w", 0.0):+.2f}' + (f'  -> {desenlace}' if desenlace else ''))
        seps.append(sep)
        desvs.append(abs(desv))
        en_banda += int(p['BANDA_MIN_M'] <= sep <= p['BANDA_MAX_M'])

    env._detener()
    calidad = 100.0 * acum / (duracion * (cal['W_BANDA'] + cal['W_DESV']))
    if desenlace == 'colision':
        calidad -= cal['P_COLISION']
    elif desenlace.startswith('perdido'):
        calidad -= cal['P_PERDIDO']
    resumen = {
        'calidad': round(calidad, 3), 'desenlace': desenlace,
        'segundos': round(time.time() - t0, 2), 'pasos': pasos,
        'sep_media': round(statistics.mean(seps), 4),
        'pct_banda': round(100.0 * en_banda / pasos, 1),
        'desv_media': round(statistics.mean(desvs), 2),
    }
    return calidad, resumen


def evaluar_individuo(politica, etiqueta, episodios=None):
    """Calidad agregada de un controlador en varios episodios. Siempre un float."""
    p2 = CTX['p2']
    k = episodios or p2['EPISODIOS_POR_INDIVIDUO']
    calidades = []
    for e in range(1, k + 1):
        calidad, resumen = episodio_calidad(CTX['env'], CTX['verdad'], politica,
                                            CTX['calidad'], p2['DURACION_EVAL_S'])
        calidades.append(calidad)
        CONTADOR['pasos'] += resumen['pasos']
        CONTADOR['evaluaciones'] += 1
        print(f'  {etiqueta:>10} ep {e}: calidad {calidad:7.2f}  {resumen["desenlace"]:<17} '
              f'{resumen["segundos"]:5.1f} s  banda {resumen["pct_banda"]:5.1f} %  '
              f'sep {resumen["sep_media"]:.2f} m')
        if CTX['ruta_episodios']:
            with open(CTX['ruta_episodios'], 'a', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=CAMPOS_EPISODIO).writerow({
                    'generacion': CTX['gen'], 'genoma': etiqueta, 'episodio': e,
                    'guia': CTX['guia'], **resumen})
    if p2['AGREGACION'] == 'min':
        return float(min(calidades))
    return float(statistics.mean(calidades))


def eval_genomes(genomas, config):
    # Si el puente o RoboboSim se cuelgan, el entorno lanza RuntimeError y la
    # evolución se detiene: no se asigna calidad a un genoma por un fallo de
    # la infraestructura. Se reanuda desde el último punto de control.
    for gid, genoma in genomas:
        genoma.fitness = evaluar_individuo(politica_red(genoma, config), str(gid))


# =====================================================================
# Registro por generación
# =====================================================================

class RegistroGeneracion(neat.reporting.BaseReporter):
    """Una fila por generación en evolucion.csv y el tamaño de cada especie en
    especies.csv. Continúa los contadores si se reanuda. Sólo guarda rutas y
    números: el Checkpointer lo serializa con pickle."""

    CAMPOS = ['generacion', 'guia', 'calidad', 'duracion_s', 'cal_mejor',
              'cal_media', 'cal_desv', 'especies', 'nodos_mejor',
              'conexiones_mejor', 'evaluaciones', 'pasos', 'segundos',
              'segundos_gen']

    def __init__(self, carpeta, guia, calidad, duracion):
        self.ruta = os.path.join(carpeta, 'evolucion.csv')
        self.ruta_especies = os.path.join(carpeta, 'especies.csv')
        self.carpeta_genomas = os.path.join(carpeta, 'genomas')
        self.guia, self.calidad, self.duracion = guia, calidad, duracion
        os.makedirs(self.carpeta_genomas, exist_ok=True)
        self.base = {'evaluaciones': 0, 'pasos': 0, 'segundos': 0.0}
        if os.path.exists(self.ruta):
            with open(self.ruta, encoding='utf-8') as f:
                filas = list(csv.DictReader(f))
            if filas:
                self.base = {k: float(filas[-1][k]) for k in self.base}
        else:
            with open(self.ruta, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.CAMPOS)
            with open(self.ruta_especies, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['generacion', 'especie', 'tamano', 'cal_max'])
        self.inicio = dict(CONTADOR)
        self.t0 = self.t_gen = time.time()
        self.gen = 0

    def start_generation(self, generation):
        self.gen = generation
        self.t_gen = time.time()
        CTX['gen'] = generation

    def post_evaluate(self, config, population, species, best_genome):
        cal = [g.fitness for g in population.values()]
        nodos, conexiones = best_genome.size()
        fila = [self.gen, self.guia, self.calidad, self.duracion,
                round(best_genome.fitness, 3),
                round(statistics.mean(cal), 3), round(statistics.pstdev(cal), 3),
                len(species.species), nodos, conexiones,
                int(self.base['evaluaciones'] + CONTADOR['evaluaciones']
                    - self.inicio['evaluaciones']),
                int(self.base['pasos'] + CONTADOR['pasos'] - self.inicio['pasos']),
                round(self.base['segundos'] + time.time() - self.t0, 1),
                round(time.time() - self.t_gen, 1)]
        with open(self.ruta, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(fila)
        with open(self.ruta_especies, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            for sid, esp in sorted(species.species.items()):
                cals = [g.fitness for g in esp.members.values() if g.fitness is not None]
                w.writerow([self.gen, sid, len(esp.members),
                            round(max(cals), 3) if cals else ''])
        ruta = os.path.join(self.carpeta_genomas, f'mejor_gen{self.gen:03d}.pkl')
        with open(ruta, 'wb') as f:
            pickle.dump(best_genome, f)


# =====================================================================
# Modos
# =====================================================================

def preparar_contexto(args, carpeta=None):
    p2 = cargar_config_p2(os.path.join(carpeta, 'p2_config.json') if carpeta
                          else args.config_p2)
    if args.duracion:
        p2['DURACION_EVAL_S'] = args.duracion
    if args.calidad:
        p2['CALIDAD'] = args.calidad
    c_env = cargar_config(args.config)
    # Sin truncado del entorno: el fin lo deciden los criterios del anexo B.
    env = RoboboSeguidorEnv(pasos_max=10**9, usar_simulador=True, ruta_config=args.config)
    CTX.update(env=env, verdad=VerdadTerreno(c_env['SIM_IP']), p2=p2,
               calidad=p2['CALIDADES'][p2['CALIDAD']], guia=args.guia,
               detalle=args.detalle)
    return env, c_env, p2


def cerrar_contexto():
    if CTX['verdad'] is not None:
        CTX['verdad'].cerrar()
    if CTX['env'] is not None:
        CTX['env'].close()


def comprobar_dimensiones(env, config_neat):
    n_obs = env.observation_space.shape[0]
    if config_neat.genome_config.num_inputs != n_obs:
        raise SystemExit(f'num_inputs = {config_neat.genome_config.num_inputs} en la '
                         f'configuración de NEAT, pero la observación tiene {n_obs} '
                         'componentes (revisa OBS_Y / OBS_IR).')


def aviso_simulador(p2, args):
    s = p2.get('SIM', {})
    print(f'Guía: {args.guia} (cámbialo a mano en RoboboSim).  Simulador: '
          f'multiplicador x{s.get("MULTIPLICADOR")}, físicas {s.get("FISICAS")}.  '
          f'Calidad {p2["CALIDAD"]}, episodios de {p2["DURACION_EVAL_S"]:.0f} s, '
          f'{p2["EPISODIOS_POR_INDIVIDUO"]} por individuo ({p2["AGREGACION"]}).')


def estimar_coste(config_neat, p2, generaciones, t_reinicio=3.0):
    t_gen = (config_neat.pop_size * p2['EPISODIOS_POR_INDIVIDUO']
             * (p2['DURACION_EVAL_S'] + t_reinicio))
    print(f'Cota de coste: {t_gen / 60:.1f} min por generación, '
          f'{t_gen * generaciones / 3600:.1f} h para {generaciones} generaciones '
          f'(reinicio estimado en {t_reinicio:.0f} s).')


def preparar_carpeta(args):
    carpeta = os.path.join(RUTA_EVOL, args.nombre)
    if os.path.exists(os.path.join(carpeta, 'evolucion.csv')) and not args.forzar:
        raise SystemExit(f'{carpeta} ya tiene una evolución. Usa --reanudar o --forzar.')
    if args.forzar and os.path.isdir(carpeta):
        shutil.rmtree(carpeta)
    os.makedirs(os.path.join(carpeta, 'chk'), exist_ok=True)

    # Copia de la configuración usada, con la semilla escrita: es parte de las evidencias.
    with open(args.neat, encoding='utf-8') as f:
        ini = re.sub(r'(?m)^seed\s*=.*$', f'seed                   = {args.semilla}', f.read())
    with open(os.path.join(carpeta, 'p2_neat.ini'), 'w', encoding='utf-8') as f:
        f.write(ini)
    p2 = cargar_config_p2(args.config_p2)
    if args.duracion:
        p2['DURACION_EVAL_S'] = args.duracion
    if args.calidad:
        p2['CALIDAD'] = args.calidad
    with open(os.path.join(carpeta, 'p2_config.json'), 'w', encoding='utf-8') as f:
        json.dump(p2, f, indent=2, ensure_ascii=False)
    shutil.copy(args.config, os.path.join(carpeta, 'p1_config.json'))
    return carpeta


def ultimo_checkpoint(carpeta):
    chks = glob.glob(os.path.join(carpeta, 'chk', 'p2-*'))
    if not chks:
        raise SystemExit(f'No hay puntos de control en {carpeta}/chk.')
    return max(chks, key=lambda r: int(r.rsplit('-', 1)[1]))


def anadir_registros(poblacion, carpeta, guia, p2):
    poblacion.add_reporter(neat.StdOutReporter(True))
    poblacion.add_reporter(neat.Checkpointer(
        generation_interval=1, filename_prefix=os.path.join(carpeta, 'chk', 'p2-')))
    poblacion.add_reporter(RegistroGeneracion(carpeta, guia, p2['CALIDAD'],
                                              p2['DURACION_EVAL_S']))


def ejecutar(poblacion, carpeta, generaciones):
    CTX['ruta_episodios'] = os.path.join(carpeta, 'episodios.csv')
    if not os.path.exists(CTX['ruta_episodios']):
        with open(CTX['ruta_episodios'], 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CAMPOS_EPISODIO).writeheader()
    mejor = poblacion.run(eval_genomes, generaciones)
    with open(os.path.join(carpeta, 'mejor_run.pkl'), 'wb') as f:
        pickle.dump(mejor, f)
    print(f'\nFin. Mejor calidad registrada: {mejor.fitness:.2f} '
          f'(puede ser un episodio afortunado: usa --reevaluar).')


def modo_evolucionar(args):
    carpeta = preparar_carpeta(args)
    config_neat = cargar_config_neat(os.path.join(carpeta, 'p2_neat.ini'))
    env, _, p2 = preparar_contexto(args, carpeta)
    try:
        comprobar_dimensiones(env, config_neat)
        aviso_simulador(p2, args)
        estimar_coste(config_neat, p2, args.generaciones)
        poblacion = neat.Population(config_neat, seed=args.semilla)
        anadir_registros(poblacion, carpeta, args.guia, p2)
        ejecutar(poblacion, carpeta, args.generaciones)
    finally:
        cerrar_contexto()


def modo_reanudar(args):
    carpeta = os.path.join(RUTA_EVOL, args.nombre)
    chk = ultimo_checkpoint(carpeta)
    print(f'Reanudando desde {chk}')
    # El punto de control guarda su propia configuración de NEAT; con
    # --nueva-config se usa la de p2_evol/<nombre>/p2_neat.ini (editada a mano,
    # p. ej. para subir compatibility_threshold si hay demasiadas especies).
    nueva = (cargar_config_neat(os.path.join(carpeta, 'p2_neat.ini'))
             if args.nueva_config else None)
    poblacion = neat.Checkpointer.restore_checkpoint(chk, new_config=nueva)
    env, _, p2 = preparar_contexto(args, carpeta)
    try:
        comprobar_dimensiones(env, poblacion.config)
        aviso_simulador(p2, args)
        estimar_coste(poblacion.config, p2, args.generaciones)
        anadir_registros(poblacion, carpeta, args.guia, p2)
        ejecutar(poblacion, carpeta, args.generaciones)
    finally:
        cerrar_contexto()


def modo_comprobar(args):
    """Un genoma aleatorio completa un episodio y recibe una calidad."""
    config_neat = cargar_config_neat(args.neat)
    env, _, p2 = preparar_contexto(args)
    try:
        comprobar_dimensiones(env, config_neat)
        aviso_simulador(p2, args)
        poblacion = neat.Population(config_neat, seed=args.semilla)
        gid, genoma = next(iter(poblacion.population.items()))
        t0 = time.time()
        calidad = evaluar_individuo(politica_red(genoma, config_neat), f'aleat{gid}', 1)
        print(f'Calidad {calidad:.2f} en {time.time() - t0:.1f} s (con reinicio). '
              f'Nodos y conexiones: {genoma.size()}')
    finally:
        cerrar_contexto()


def modo_reto(args):
    """Referencia: el controlador programado a mano con la misma calidad."""
    env, c_env, p2 = preparar_contexto(args)
    try:
        aviso_simulador(p2, args)
        r = p2['RETO']
        politica = controlador_reto(c_env, r['K_GIRO'], r['K_VEL'])
        calidades = [evaluar_individuo(politica, 'reto', 1) for _ in range(args.episodios)]
        media = statistics.mean(calidades)
        print(f'Controlador del reto en {args.guia}: calidad media {media:.2f} '
              f'(min {min(calidades):.2f}, max {max(calidades):.2f}, {len(calidades)} ep.)')
        ruta = os.path.join(RUTA_EVOL, 'referencias.csv')
        os.makedirs(RUTA_EVOL, exist_ok=True)
        nuevo = not os.path.exists(ruta)
        with open(ruta, 'a', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(['controlador', 'guia', 'calidad', 'duracion_s',
                            'episodios', 'cal_media', 'cal_min', 'cal_max'])
            w.writerow(['reto', args.guia, p2['CALIDAD'], p2['DURACION_EVAL_S'],
                        len(calidades), round(media, 3), round(min(calidades), 3),
                        round(max(calidades), 3)])
        print(f'Anotado en {ruta}')
    finally:
        cerrar_contexto()


def modo_reevaluar(args):
    """Vuelve a evaluar a los mejores de la evolución y guarda el elegido."""
    carpeta = os.path.join(RUTA_EVOL, args.nombre)
    with open(os.path.join(carpeta, 'evolucion.csv'), encoding='utf-8') as f:
        filas = sorted(csv.DictReader(f), key=lambda r: -float(r['cal_mejor']))
    config_neat = cargar_config_neat(os.path.join(carpeta, 'p2_neat.ini'))
    candidatos, vistos = [], set()
    for fila in filas:
        ruta = os.path.join(carpeta, 'genomas', f'mejor_gen{int(fila["generacion"]):03d}.pkl')
        with open(ruta, 'rb') as f:
            genoma = pickle.load(f)
        if genoma.key not in vistos:
            vistos.add(genoma.key)
            candidatos.append((ruta, genoma))
        if len(candidatos) == args.candidatos:
            break

    env, _, p2 = preparar_contexto(args, carpeta)
    p2['AGREGACION'] = 'media'      # para elegir, la media de varios episodios
    try:
        aviso_simulador(p2, args)
        resultados = []
        for ruta, genoma in candidatos:
            print(f'\n{os.path.basename(ruta)} (genoma {genoma.key}, '
                  f'calidad en la evolución {genoma.fitness:.2f})')
            media = evaluar_individuo(politica_red(genoma, config_neat),
                                      str(genoma.key), args.episodios)
            resultados.append((media, ruta, genoma))
        resultados.sort(key=lambda r: -r[0])
        with open(os.path.join(carpeta, 'reevaluacion.csv'), 'a', newline='',
                  encoding='utf-8') as f:
            w = csv.writer(f)
            for media, ruta, genoma in resultados:
                w.writerow([args.guia, os.path.basename(ruta), genoma.key,
                            round(genoma.fitness, 3), args.episodios, round(media, 3)])
        media, ruta, genoma = resultados[0]
        with open(os.path.join(carpeta, 'mejor_genoma.pkl'), 'wb') as f:
            pickle.dump(genoma, f)
        print(f'\nElegido {os.path.basename(ruta)} (genoma {genoma.key}): '
              f'calidad media {media:.2f} en {args.episodios} episodios -> '
              f'{carpeta}/mejor_genoma.pkl')
    finally:
        cerrar_contexto()


def main():
    a = argparse.ArgumentParser(description='Evolución NEAT del seguidor (Práctica 2).')
    modo = a.add_mutually_exclusive_group(required=True)
    modo.add_argument('--evolucionar', action='store_true')
    modo.add_argument('--reanudar', action='store_true')
    modo.add_argument('--comprobar', action='store_true')
    modo.add_argument('--reto', action='store_true')
    modo.add_argument('--reevaluar', action='store_true')
    a.add_argument('--nombre', default='s0', help='carpeta p2_evol/<nombre>')
    a.add_argument('--semilla', type=int, default=0, help='semilla de NEAT')
    a.add_argument('--generaciones', type=int, default=15)
    a.add_argument('--guia', choices=GUIAS, required=True,
                   help='configuración del guía puesta en RoboboSim (sólo se anota)')
    a.add_argument('--calidad', choices=['A', 'B', 'A0'], default=None,
                   help='formulación de la calidad (por defecto, la de p2_config.json)')
    a.add_argument('--duracion', type=float, default=None,
                   help='segundos por episodio de evaluación (por defecto, DURACION_EVAL_S)')
    a.add_argument('--episodios', type=int, default=3, help='--reto y --reevaluar')
    a.add_argument('--candidatos', type=int, default=5, help='--reevaluar')
    a.add_argument('--forzar', action='store_true', help='sobrescribir la carpeta')
    a.add_argument('--detalle', action='store_true',
                   help='una línea por paso con la verdad del terreno y el blob')
    a.add_argument('--nueva-config', action='store_true',
                   help='--reanudar con el p2_neat.ini de la carpeta, no el del punto de control')
    a.add_argument('--neat', default=RUTA_NEAT_INI)
    a.add_argument('--config', default=RUTA_CONFIG, help='configuración del entorno (P1)')
    a.add_argument('--config-p2', default=RUTA_CONFIG_P2)
    args = a.parse_args()

    if args.evolucionar:
        modo_evolucionar(args)
    elif args.reanudar:
        modo_reanudar(args)
    elif args.comprobar:
        modo_comprobar(args)
    elif args.reto:
        modo_reto(args)
    else:
        modo_reevaluar(args)


if __name__ == '__main__':
    main()
