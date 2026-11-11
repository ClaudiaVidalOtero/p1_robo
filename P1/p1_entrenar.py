#!/usr/bin/env python3
"""
Práctica 1 — Entrenamiento con Stable-Baselines3 y curvas de aprendizaje.

Modos (uno por ejecución, desde la raíz del proyecto):

    python3 p1_entrenar.py --importar-calibracion   # calibracion.json -> p1_config.json
    python3 p1_entrenar.py --normalizacion          # configuración activa
    python3 p1_entrenar.py --comprobar              # check_env de Gymnasium
    python3 p1_entrenar.py --aleatorio --episodios 8
    python3 p1_entrenar.py --entrenar --algoritmo sac --pasos 20000 --semilla 0
    python3 p1_entrenar.py --evaluar  --algoritmo sac --semilla 0
    python3 p1_entrenar.py --graficas --grupos sac_s ppo_s --ventana 10

--aleatorio deja p1_logs/pasos_aleatorio_s<N>.csv, de donde --graficas toma
la recompensa media de referencia. --entrenar deja el Monitor
p1_logs/monitor_<alg>_s<N>.monitor.csv (una fila por episodio) y el modelo
modelos/p1_modelo_<alg>_s<N>.zip. La evaluación del anexo B está en
p1_evaluar.py.

Algoritmos: SAC (fuera de política, reutiliza cada transición desde el búfer
de repetición: aprovecha mejor pocos pasos, que aquí cuestan tiempo real) y
PPO (dentro de política, más estable pero necesita más pasos).
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import time

import numpy as np

from p1_graficas_eval import RUTA_LOGS
from p1_robobo_env import (
    RUTA_CONFIG,
    RUTA_MODELOS,
    RoboboSeguidorEnv,
    cargar_config,
    importar_calibracion,
)

RUTA_CHECKPOINTS = 'p1_checkpoints'


# =====================================================================
# Hiperparámetros
# =====================================================================
# Cada paso cuesta tiempo real: se ajustan sobre todo los lotes.

HIPER_SAC = {
    'learning_rate': 3e-4,
    'buffer_size': 50_000,
    'learning_starts': 500,      # pasos con acciones aleatorias antes de aprender
    'batch_size': 256,
    'tau': 0.005,
    'gamma': 0.99,
    'train_freq': 1,             # una actualización por paso del entorno
    'gradient_steps': 1,
}

HIPER_PPO = {
    'learning_rate': 3e-4,
    'n_steps': 512,              # pasos recogidos por actualización
    'batch_size': 64,
    'n_epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'ent_coef': 0.0,
}


def ruta_modelo(args):
    return os.path.join(RUTA_MODELOS, f'p1_modelo_{args.algoritmo}_s{args.semilla}')


def construir_algoritmo(args, env):
    from stable_baselines3 import PPO, SAC
    if args.algoritmo == 'ppo':
        return PPO('MlpPolicy', env, verbose=1, seed=args.semilla,
                   tensorboard_log=RUTA_LOGS, **HIPER_PPO)
    return SAC('MlpPolicy', env, verbose=1, seed=args.semilla,
               tensorboard_log=RUTA_LOGS, **HIPER_SAC)


def cargar_algoritmo(args, env):
    from stable_baselines3 import PPO, SAC
    clase = PPO if args.algoritmo == 'ppo' else SAC
    return clase.load(ruta_modelo(args), env=env)


def crear_env(args):
    return RoboboSeguidorEnv(
        pasos_max=args.pasos_max,
        usar_simulador=not args.sin_simulador,
        verbose=args.detalle,
        ruta_config=args.config)


# =====================================================================
# Modos
# =====================================================================

def modo_normalizacion(args):
    """Imprime la configuración activa y la normalización de la observación."""
    c = cargar_config(args.config)
    print(f'Configuración: {args.config}')
    print(json.dumps(c, indent=2, ensure_ascii=False))
    band = (c['SIZE_BANDA_MIN'], c['SIZE_BANDA_MAX'], c['SIZE_OBJ'])
    print('\nNormalización de la observación')
    print('  posx = (x - {:.0f}) / {:.0f}  -> [-1, 1]'.format(
        c['POSX_CENTRO'], c['POSX_CENTRO']))
    print('  size = size / {:.0f}          -> [0, 1]'.format(c['SIZE_MAX']))
    print('  banda objetivo en size = [{:.1f}, {:.1f}], centro {:.1f}'.format(*band))
    print('  posy = y / {:.0f}            -> [0, 1]  (OBS_Y={})'.format(c['IMG_ALTO'], c['OBS_Y']))
    print('  no visible -> posx = size = posy = 0')


def modo_comprobar(args):
    """Comprueba con check_env que el entorno cumple la interfaz de Gymnasium."""
    from stable_baselines3.common.env_checker import check_env
    env = crear_env(args)
    try:
        check_env(env, warn=True)
        print('\nEl entorno cumple la interfaz de Gymnasium.')
    finally:
        env.close()


def modo_aleatorio(args):
    """Línea base con acciones aleatorias en el entorno de entrenamiento."""
    env = crear_env(args)
    try:
        ruta = os.path.join(RUTA_LOGS, f'pasos_aleatorio_s{args.semilla}.csv')
        resumen(env, None, args.episodios, 'POLÍTICA ALEATORIA',
                ruta_pasos=ruta)
    finally:
        env.close()


def modo_entrenar(args):
    env = crear_env(args)
    try:
        from stable_baselines3.common.monitor import Monitor
        os.makedirs(RUTA_LOGS, exist_ok=True)
        # Monitor: una fila por episodio con r, l, t y la causa de fin.
        csv = os.path.join(RUTA_LOGS, f'monitor_{args.algoritmo}_s{args.semilla}')
        env = Monitor(env, filename=csv,
                      info_keywords=('choque', 'perdido'))

        modelo = construir_algoritmo(args, env)
        print(f'\nEntrenando {args.algoritmo.upper()} durante '
              f'{args.pasos} pasos (semilla {args.semilla}).')
        print('Ctrl-C interrumpe y guarda el modelo igualmente.\n')

        # Copias intermedias cada N llamadas a step() en p1_checkpoints/.
        callback = None
        if args.checkpoint > 0:
            from stable_baselines3.common.callbacks import CheckpointCallback
            nombre = f'{args.algoritmo}_s{args.semilla}'
            callback = CheckpointCallback(
                save_freq=args.checkpoint,
                save_path=os.path.join(RUTA_CHECKPOINTS, nombre),
                name_prefix=nombre)
            print(f'Copia intermedia cada {args.checkpoint} pasos en '
                  f'{os.path.join(RUTA_CHECKPOINTS, nombre)}/\n')

        t0 = time.time()
        try:
            modelo.learn(total_timesteps=args.pasos, progress_bar=False,
                         callback=callback)
        except KeyboardInterrupt:
            print('\nEntrenamiento interrumpido por el usuario.')
        except Exception:
            # Fallo del simulador o del puente: se guarda la red antes de propagar.
            ruta = ruta_modelo(args) + '_interrumpido'
            modelo.save(ruta)
            print(f'\nERROR durante el entrenamiento. Modelo guardado en {ruta}.zip')
            raise

        modelo.save(ruta_modelo(args))
        dt = time.time() - t0
        print(f'\nModelo guardado en {ruta_modelo(args)}.zip')
        print(f'Tiempo empleado: {dt / 60.0:.1f} min  '
              f'({dt / max(args.pasos, 1):.3f} s por paso)')
    finally:
        env.close()


def modo_evaluar(args):
    env = crear_env(args)
    try:
        if not os.path.exists(ruta_modelo(args) + '.zip'):
            raise SystemExit(f'No existe {ruta_modelo(args)}.zip. Entrena primero.')
        modelo = cargar_algoritmo(args, env)
        # deterministic=True: acción media, sin exploración.
        ruta = os.path.join(RUTA_LOGS, f'pasos_{args.algoritmo}_s{args.semilla}.csv')
        resumen(env,
                lambda obs: modelo.predict(obs, deterministic=True)[0],
                args.episodios,
                f'POLÍTICA APRENDIDA ({args.algoritmo.upper()})',
                ruta_pasos=ruta)
    finally:
        env.close()


# =====================================================================
# Episodios y métricas
# =====================================================================

def resumen(env, politica, episodios, titulo, ruta_pasos=None):
    """Ejecuta episodios con la política (None = aleatoria) y resume el resultado.

    Usa las terminaciones del propio entorno, no las del anexo B
    (p1_evaluar.py). Con ruta_pasos escribe un CSV con una fila por paso.
    """
    recompensas, longitudes = [], []
    completados = choques = perdidos = 0
    pasos_totales = pasos_banda = 0
    suma_size = 0.0

    if ruta_pasos:
        os.makedirs(os.path.dirname(ruta_pasos) or '.', exist_ok=True)
    with (open(ruta_pasos, 'w', newline='', encoding='utf-8') if ruta_pasos
          else contextlib.nullcontext()) as f:
        if f is not None:
            f.write('episodio,paso,t,posx,size_norm,size_raw,v,w,r,'
                    'visible,en_banda,choque,perdido,demasiado_cerca\n')

        print('\n' + titulo)
        print('-' * len(titulo))
        t0 = time.time()

        for ep in range(episodios):
            obs, _ = env.reset()
            total, pasos, info = 0.0, 0, {}
            terminado = truncado = False
            while not (terminado or truncado):
                accion = env.action_space.sample() if politica is None else politica(obs)
                obs, r, terminado, truncado, info = env.step(accion)
                total += r
                pasos += 1
                if info.get('en_banda'):
                    pasos_banda += 1
                suma_size += info.get('size_raw', 0.0)
                if f is not None:
                    f.write('{},{},{:.3f},{:.4f},{:.4f},{:.1f},{:.4f},{:.4f},'
                            '{:.4f},{},{},{},{},{}\n'.format(
                                ep + 1, pasos, time.time() - t0,
                                info.get('posx', 0.0), info.get('size', 0.0),
                                info.get('size_raw', 0.0), info.get('v', 0.0),
                                info.get('w', 0.0), r,
                                int(bool(info.get('visible'))),
                                int(bool(info.get('en_banda'))),
                                int(bool(info.get('choque'))),
                                int(bool(info.get('perdido'))),
                                int(bool(info.get('demasiado_cerca')))))

            pasos_totales += pasos
            recompensas.append(total)
            longitudes.append(pasos)
            if info.get('choque'):
                choques += 1
                desenlace = 'choque'
            elif info.get('perdido'):
                perdidos += 1
                desenlace = 'guía perdido'
            else:
                completados += 1
                desenlace = 'completado'
            print('episodio {:2d}   pasos {:3d}   recompensa {:8.2f}   size final '
                  '{:6.1f}   {}'.format(ep + 1, pasos, total,
                                         info.get('size_raw', 0.0), desenlace))

    if ruta_pasos:
        print(f'registro por paso: {ruta_pasos}')

    print(f'\nrecompensa media   {float(np.mean(recompensas)):.2f}  '
          f'(desviación {float(np.std(recompensas)):.2f})')
    print(f'longitud media     {float(np.mean(longitudes)):.1f} pasos')
    print(f'completados        {completados}/{episodios}')
    print(f'choques            {choques}/{episodios}')
    print(f'guía perdido       {perdidos}/{episodios}')
    if pasos_totales:
        print(f'pasos en banda     {100.0 * pasos_banda / pasos_totales:.1f} %   '
              f'(size medio {suma_size / pasos_totales:.1f})')
    print('coste              '
          f'{(time.time() - t0) / max(pasos_totales, 1):.3f} s/paso (incluye resets)')


# =====================================================================
# Curvas de aprendizaje (a partir de los CSV de Monitor)
# =====================================================================

def _leer_monitor(ruta):
    """Lee un CSV de Monitor (primera línea '#' y columnas r,l,t,choque,perdido).

    Devuelve arrays por episodio (pasos acumulados, recompensa, longitud,
    choque, perdido) y t_final, segundos desde el inicio al último episodio.
    """
    pasos, recompensas, longitudes, choques, perdidos = [], [], [], [], []
    acumulado, t_final = 0, 0.0
    with open(ruta, encoding='utf-8') as f:
        lineas = [x for x in f if x.strip() and not x.startswith('#')]
    for fila in csv.DictReader(lineas):
        l = int(float(fila['l']))
        acumulado += l
        pasos.append(acumulado)
        recompensas.append(float(fila['r']))
        longitudes.append(l)
        choques.append(fila.get('choque') == 'True')
        perdidos.append(fila.get('perdido') == 'True')
        t_final = float(fila['t'])
    return (np.array(pasos), np.array(recompensas), np.array(longitudes),
            np.array(choques, dtype=float), np.array(perdidos, dtype=float),
            t_final)


def _media_movil(y, ventana):
    v = int(ventana)
    if v <= 1:
        return y
    k = np.ones(v) / float(v)
    return np.convolve(y, k, mode='valid')


def _recompensa_aleatoria(semilla=0):
    """(Recompensa media por episodio, nº de episodios) de la política
    aleatoria a partir de pasos_aleatorio_s<N>.csv; (None, 0) si no existe."""
    ruta = os.path.join(RUTA_LOGS, f'pasos_aleatorio_s{semilla}.csv')
    if not os.path.exists(ruta):
        return None, 0
    por_ep = {}
    with open(ruta, encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            por_ep[fila['episodio']] = por_ep.get(fila['episodio'], 0.0) + float(fila['r'])
    if not por_ep:
        return None, 0
    return float(np.mean(list(por_ep.values()))), len(por_ep)


def modo_graficas(args):
    """Curvas de aprendizaje desde los Monitor: recompensa, duración y causa
    de fin frente a los pasos de step(); media y banda min-max por grupo de
    semillas, serie cruda tenue y línea de la política aleatoria."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    ax_r, ax_l, ax_c = ejes
    colores = plt.rcParams['axes.prop_cycle'].by_key()['color']
    estilos_causa = {'choque': '-', 'perdido': '--', 'tiempo': ':'}
    resumen_pie = []
    pasos_tot, segundos_tot = 0, 0.0   # coste medio por paso (incluye reinicios)

    for i, grupo in enumerate(args.grupos):
        rutas = sorted(glob.glob(os.path.join(RUTA_LOGS, f'monitor_{grupo}*.monitor.csv')))
        if not rutas:
            raise SystemExit(f'No hay {RUTA_LOGS}/monitor_{grupo}*.monitor.csv')
        color = colores[i % len(colores)]
        series = []   # por semilla: (x, {magnitud: media móvil})
        for ruta in rutas:
            pasos, r, l, choque, perdido, t_final = _leer_monitor(ruta)
            if len(pasos) < args.ventana:
                print(f'aviso: {ruta} tiene menos episodios que la ventana; se omite')
                continue
            pasos_tot += int(pasos[-1])
            segundos_tot += t_final
            tiempo = 1.0 - np.clip(choque + perdido, 0, 1)
            ax_r.plot(pasos, r, color=color, alpha=0.15, linewidth=0.7)
            ax_l.plot(pasos, l, color=color, alpha=0.15, linewidth=0.7)
            x = pasos[args.ventana - 1:]
            series.append((x, {
                'r': _media_movil(r, args.ventana),
                'l': _media_movil(l, args.ventana),
                'choque': _media_movil(choque, args.ventana),
                'perdido': _media_movil(perdido, args.ventana),
                'tiempo': _media_movil(tiempo, args.ventana)}))
            print(f'{ruta}: {len(pasos)} episodios, {int(pasos[-1])} pasos')
        if not series:
            continue
        n = len(series)
        nombre = grupo.upper().removesuffix('_S')   # 'sac_s' -> 'SAC'
        etiqueta = f'{nombre} ({n} semilla{"s" if n > 1 else ""})'
        resumen_pie.append(etiqueta)

        if n == 1:
            x, m = series[0]
            ax_r.plot(x, m['r'], color=color, label=etiqueta)
            ax_l.plot(x, m['l'], color=color)
            for causa, ls in estilos_causa.items():
                ax_c.plot(x, m[causa], color=color, linestyle=ls)
            continue

        # Rejilla común: desde donde todas tienen media móvil hasta donde
        # acaba la más corta (no se extrapola).
        x0 = max(s[0][0] for s in series)
        x1 = min(s[0][-1] for s in series)
        rejilla = np.linspace(x0, x1, 400)

        def interp(clave):
            return np.array([np.interp(rejilla, s[0], s[1][clave]) for s in series])

        for clave, ax in (('r', ax_r), ('l', ax_l)):
            y = interp(clave)
            ax.fill_between(rejilla, y.min(0), y.max(0), color=color, alpha=0.25,
                            label=f'{etiqueta}, min-max' if clave == 'r' else None)
            ax.plot(rejilla, y.mean(0), color=color,
                    label=f'{etiqueta}, media' if clave == 'r' else None)
        for causa, ls in estilos_causa.items():
            ax_c.plot(rejilla, interp(causa).mean(0), color=color, linestyle=ls)

    if args.sin_ref:
        ref, n_ref = None, 0
    elif args.ref is not None:
        ref, n_ref = args.ref, 0
    else:
        ref, n_ref = _recompensa_aleatoria()
    if ref is not None:
        txt = 'política aleatoria' + (f' ({n_ref} ep.)' if n_ref else '')
        ax_r.axhline(ref, color='red', linestyle='--', linewidth=1, label=txt)

    # Leyenda de causas: estilo de línea = causa.
    for causa, ls in estilos_causa.items():
        nombre = {'choque': 'choque', 'perdido': 'guía perdido',
                  'tiempo': 'límite de tiempo'}[causa]
        ax_c.plot([], [], color='0.3', linestyle=ls, label=nombre)

    ax_r.set_ylabel('Recompensa por episodio')
    s_paso = segundos_tot / pasos_tot if pasos_tot else None
    ax_l.set_ylabel('Duración del episodio (pasos'
                    + (f'; 1 paso ≈ {s_paso:.2f} s)' if s_paso else ')'))
    ax_c.set_ylabel('Fracción de episodios')
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlabel('Pasos del entorno (llamadas a step())')
    ax_r.legend(fontsize=8, loc='upper left')
    ax_c.legend(fontsize=8, loc='center right', title='causa de fin', title_fontsize=8)
    for ax in ejes:
        ax.grid(alpha=0.3)
    titulo = args.titulo or 'Curvas de aprendizaje'
    fig.suptitle(f'{titulo}\nmedia móvil de {args.ventana} episodios; serie cruda tenue')
    fig.tight_layout()
    salida = args.salida or 'p1_graficas.png'
    fig.savefig(salida, dpi=150)
    print(f'Figura guardada en {salida}')
    print('Para el pie: ' + '; '.join(resumen_pie)
          + f'; ventana {args.ventana} episodios'
          + (f'; {s_paso:.2f} s por paso medidos (reinicios incluidos)' if s_paso else '')
          + (f'; ref. aleatoria {ref:.1f}' if ref is not None else ''))


def main():
    p = argparse.ArgumentParser(
        description='Práctica 1 RIA: seguimiento del guía con RL (SB3).')

    modo = p.add_mutually_exclusive_group(required=True)
    modo.add_argument('--comprobar', action='store_true',
                      help='verifica la interfaz del entorno (check_env)')
    modo.add_argument('--normalizacion', action='store_true',
                      help='imprime la configuración y las constantes de normalización')
    modo.add_argument('--aleatorio', action='store_true',
                      help='episodios con acciones aleatorias (línea base)')
    modo.add_argument('--entrenar', action='store_true')
    modo.add_argument('--evaluar', action='store_true')
    modo.add_argument('--graficas', action='store_true',
                      help='curvas de aprendizaje desde p1_logs/monitor_*.csv')
    modo.add_argument('--importar-calibracion', action='store_true',
                      help='regenera config/p1_config.json desde config/calibracion.json')

    p.add_argument('--algoritmo', choices=['ppo', 'sac'], default='sac')
    p.add_argument('--pasos', type=int, default=20_000,
                   help='pasos totales de entrenamiento')
    p.add_argument('--checkpoint', type=int, default=2000,
                   help='copia intermedia del modelo cada N pasos (0 = ninguna)')
    p.add_argument('--episodios', type=int, default=8)
    p.add_argument('--pasos-max', type=int, default=300,
                   help='longitud máxima de un episodio en pasos')
    p.add_argument('--semilla', type=int, default=0)
    p.add_argument('--sin-simulador', action='store_true',
                   help='sin módulo sim; el seguidor se repone a mano')
    p.add_argument('--detalle', action='store_true',
                   help='imprime una línea por paso')
    p.add_argument('--config', default=RUTA_CONFIG,
                   help='archivo de configuración calibrada')

    # Solo para --graficas
    p.add_argument('--ventana', type=int, default=10,
                   help='ventana de la media móvil, en episodios')
    p.add_argument('--ref', type=float, default=None,
                   help='recompensa media de la política aleatoria (línea de referencia)')
    p.add_argument('--sin-ref', action='store_true',
                   help='no dibuja la línea de la política aleatoria')
    p.add_argument('--salida', default=None, help='PNG de salida de --graficas')
    p.add_argument('--grupos', nargs='+', default=['sac_s'],
                   help='prefijos de monitor a agrupar (p. ej. sac_s ppo_s0); '
                        'cada grupo = media y banda de sus semillas')
    p.add_argument('--titulo', default=None,
                   help='título de la figura (p. ej. la configuración del guía)')

    args = p.parse_args()

    if args.importar_calibracion:
        importar_calibracion(ruta_config=args.config)
    elif args.normalizacion:
        modo_normalizacion(args)
    elif args.comprobar:
        modo_comprobar(args)
    elif args.aleatorio:
        modo_aleatorio(args)
    elif args.entrenar:
        modo_entrenar(args)
    elif args.evaluar:
        modo_evaluar(args)
    elif args.graficas:
        modo_graficas(args)


if __name__ == '__main__':
    main()
