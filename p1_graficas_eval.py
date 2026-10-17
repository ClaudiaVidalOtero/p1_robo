#!/usr/bin/env python3
"""
Práctica 1 — Métricas, tablas y figuras de la EVALUACIÓN (anexo B).

Único sitio donde viven las métricas del anexo B, las tablas y las figuras.
p1_evaluar.py sólo ejecuta episodios y escribe los registros por paso
(p1_logs/eval_<etiqueta>_<C>.csv); aquí se leen esos CSV y se produce:

  1. fig_series_C1/C2/C3.png  separación y |desviación| frente al tiempo, con
                              la banda objetivo 0,20-0,45 m y los umbrales del
                              protocolo marcados (una línea por política).
  2. fig_desenlaces.png       barras apiladas de desenlaces por configuración
                              y causa de fin (una figura por política).
  3. tabla_anexo_b.md/.csv    las 4 métricas del anexo B (por configuración y
                              TOTAL) + velocidad real del guía + correlación
                              desviación <-> posx.
  4. (consola) la misma tabla y los avisos sobre la velocidad del guía.

No necesita ROS 2, ni simulador, ni la política: trabaja sólo con los CSV.
Por eso este módulo define el PROTOCOLO y el esquema del CSV (COLUMNAS), que
p1_evaluar.py importa desde aquí.

USO
    python3 p1_graficas_eval.py                          # todo lo de p1_logs
    python3 p1_graficas_eval.py --etiquetas sac_s0 --salida p1_graficas
    python3 p1_graficas_eval.py --etiquetas sac_s0 aleatorio
    python3 p1_graficas_eval.py --solo-tablas            # sin matplotlib
    python3 p1_graficas_eval.py --sin-consola            # sin tabla en consola

Salida en --salida (por defecto p1_graficas/):
    fig_series_C1.png, fig_series_C2.png, fig_series_C3.png
    fig_desenlaces.png
    tabla_anexo_b.md, tabla_anexo_b.csv
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

RUTA_LOGS = 'p1_logs'

# Parámetros del protocolo (anexo B). Comunes a todos los grupos y ajenos a la
# calibración: por eso no están en p1_config.json.
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

# Esquema del registro por paso que escribe p1_evaluar.py.
COLUMNAS = ['config', 'episodio', 'paso', 't',
            'seg_x_mm', 'seg_z_mm', 'seg_rumbo_deg',
            'pelota_x_mm', 'pelota_z_mm', 'guia_x_mm', 'guia_z_mm',
            'separacion_m', 'en_banda', 'desviacion_deg',
            'visible', 'blob_posx', 'blob_size', 't_sin_ver',
            'a0', 'a1', 'v', 'w',
            'entorno_choque', 'entorno_perdido', 'desenlace']

# Orden fijo de las configuraciones del anexo B.
ORDEN_CONFIG = ['C1', 'C2', 'C3']

# Causas de fin, en el orden en que se apilan. 'sin_terminar' es un episodio
# cuyo CSV no tiene desenlace (p. ej. se abortó la ejecución).
CAUSAS = ['completado', 'colision', 'perdido_distancia',
          'perdido_sin_ver', 'sin_terminar']
ETIQUETA_CAUSA = {
    'completado': 'completado',
    'colision': 'colisión',
    'perdido_distancia': 'guía perdido (distancia)',
    'perdido_sin_ver': 'guía perdido (vista)',
    'sin_terminar': 'sin terminar',
}
COLOR_CAUSA = {
    'completado': '#43a047',
    'colision': '#e53935',
    'perdido_distancia': '#fb8c00',
    'perdido_sin_ver': '#fdd835',
    'sin_terminar': '#9e9e9e',
}

PATRON = re.compile(r'^eval_(?P<et>.+)_(?P<cfg>C[123])\.csv$')


# =====================================================================
# Lectura de los registros
# =====================================================================

def leer_csv(ruta):
    with open(ruta, newline='') as f:
        return list(csv.DictReader(f))


def descubrir(ruta_logs=RUTA_LOGS):
    """Devuelve {etiqueta: {config: [filas]}} a partir de p1_logs/eval_*.csv."""
    datos = {}
    for ruta in sorted(glob.glob(os.path.join(ruta_logs, 'eval_*.csv'))):
        m = PATRON.match(os.path.basename(ruta))
        if not m:
            continue
        filas = leer_csv(ruta)
        if not filas:
            continue
        datos.setdefault(m['et'], {})[m['cfg']] = filas
    return datos


# =====================================================================
# Métricas del anexo B (siempre a partir de los CSV por paso)
# =====================================================================

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


def metricas_etiqueta(por_config):
    """Métricas por configuración (en orden C1, C2, C3) más la fila TOTAL."""
    ordenadas = sorted(por_config.items(),
                       key=lambda kv: ORDEN_CONFIG.index(kv[0]))
    filas = {cfg: metricas(v) for cfg, v in ordenadas}
    filas['TOTAL'] = metricas([f for v in por_config.values() for f in v])
    return filas


def velocidad_guia(filas):
    """Velocidad media real del guía (m/s) y su perfil por tramos de 10 s."""
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


# =====================================================================
# Tablas (consola, Markdown y CSV)
# =====================================================================

def _fila_tabla(nombre, m):
    """Columnas de la tabla del anexo B a partir de un dict de métricas."""
    d = m['desenlaces']
    return {
        'config': nombre,
        'episodios': m['episodios'],
        'completados': m['completados'],
        'pct_completados': round(m['pct_completados'], 1),
        'colision': d.get('colision', 0),
        'perdido_distancia': d.get('perdido_distancia', 0),
        'perdido_sin_ver': d.get('perdido_sin_ver', 0),
        'pct_en_banda': round(m['pct_en_banda'], 1),
        'separacion_media_m': round(m['separacion_media_m'], 3),
        'desviacion_media_deg': round(m['desviacion_media_deg'], 1),
        'pct_desv_bajo_umbral': round(m['pct_desv_bajo_umbral'], 1),
    }


CABECERA_CSV = ['etiqueta', 'config', 'episodios', 'completados',
                'pct_completados', 'colision', 'perdido_distancia',
                'perdido_sin_ver', 'pct_en_banda', 'separacion_media_m',
                'desviacion_media_deg', 'pct_desv_bajo_umbral']


def imprimir_tabla(etiqueta, filas):
    titulo = f'ANEXO B — {etiqueta}'
    print('\n' + titulo)
    print('=' * len(titulo))
    print(f'{"":8s}{"episodios":>10s}{"complet.":>10s}{"colisión":>10s}'
          f'{"perd.dist":>10s}{"perd.vista":>11s}{"% banda":>9s}'
          f'{"sep. media":>11s}{"|desv| °":>10s}{"% <15°":>8s}')
    for nombre, m in filas.items():
        d = m['desenlaces']
        print(f'{nombre:8s}{m["episodios"]:>10d}'
              f'{m["completados"]:>5d} ({m["pct_completados"]:3.0f}%)'
              f'{d.get("colision", 0):>9d}{d.get("perdido_distancia", 0):>10d}'
              f'{d.get("perdido_sin_ver", 0):>11d}{m["pct_en_banda"]:>8.1f}%'
              f'{m["separacion_media_m"]:>9.3f} m{m["desviacion_media_deg"]:>10.1f}'
              f'{m["pct_desv_bajo_umbral"]:>7.1f}%')
    print('\nCausas: colisión (< 0,12 m), guía perdido por distancia (> 0,90 m) '
          'o por pelota sin detectar > 3 s.')


def avisar_velocidad_guia(por_config_filas):
    """Velocidad del guía por configuración; avisa si dos configuraciones
    distintas tienen el mismo perfil (configuración no aplicada) o si C1
    (velocidad constante) fluctúa."""
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
        for b in sorted(perfiles):
            if b == 'C1':
                continue
            if abs(perfiles['C1'][0][0] - perfiles[b][0][0]) < 0.01:
                print(f'  AVISO: C1 y {b} tienen la misma velocidad media del '
                      'guía. ¿Se aplicó la configuración del guía?')
        perfil = perfiles['C1'][0][1]
        if len(perfil) > 1 and max(perfil) - min(perfil) > 0.03:
            print('  AVISO: en C1 (10-10, constante) la velocidad del guía '
                  'fluctúa. ¿Se aplicó la configuración del guía?')


def imprimir_consola(etiqueta, por_config):
    imprimir_tabla(etiqueta, metricas_etiqueta(por_config))
    avisar_velocidad_guia(por_config)
    cal = comprobar_calibracion([f for v in por_config.values() for f in v])
    if cal is not None:
        # El signo es negativo porque la x del blob crece hacia el lado
        # contrario al de la desviación; lo que comprueba la calibración es
        # que |r| sea alto.
        print(f'Correlación desviación real <-> posx del blob: {cal[0]:+.3f} '
              f'({cal[1]} pasos con pelota visible). |r| debe ser alto '
              '(signo esperado: negativo).')


def escribir_tablas(datos, dir_salida):
    """Tabla del anexo B (todas las etiquetas) en Markdown y CSV."""
    filas_csv = []
    lineas = [
        '| Etiqueta | Config | Episodios | Completados | Colisión | '
        'Perd. dist. | Perd. vista | % en banda | Sep. media (m) | '
        '\\|desv\\| (°) | % <15° |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    bloque_vel, bloque_cor = [], []
    for et, por_config in datos.items():
        for nombre, m in metricas_etiqueta(por_config).items():
            v = _fila_tabla(nombre, m)
            filas_csv.append({'etiqueta': et, **v})
            lineas.append(
                f"| {et} | {v['config']} | {v['episodios']} | "
                f"{v['completados']} ({v['pct_completados']:.0f}%) | "
                f"{v['colision']} | {v['perdido_distancia']} | "
                f"{v['perdido_sin_ver']} | {v['pct_en_banda']:.1f}% | "
                f"{v['separacion_media_m']:.3f} | "
                f"{v['desviacion_media_deg']:.1f} | "
                f"{v['pct_desv_bajo_umbral']:.1f}% |")

        # Velocidad real del guía (detecta una configuración mal aplicada).
        for cfg, ss in sorted(por_config.items(),
                              key=lambda kv: ORDEN_CONFIG.index(kv[0])):
            media, perfil = velocidad_guia(ss)
            if media is not None:
                bloque_vel.append(
                    f'- {et} / {cfg}: {media:.3f} m/s   (por 10 s: '
                    + ' '.join(f'{x:.2f}' for x in perfil) + ')')

        # Correlación desviación real <-> posx del blob (comprueba calibración).
        cal = comprobar_calibracion([f for v in por_config.values() for f in v])
        if cal is not None:
            bloque_cor.append(
                f'- {et}: r = {cal[0]:+.3f} ({cal[1]} pasos con pelota '
                'visible). |r| debe ser alto (signo esperado: negativo).')

    ruta_md = os.path.join(dir_salida, 'tabla_anexo_b.md')
    with open(ruta_md, 'w', encoding='utf-8') as f:
        f.write('# Tabla del anexo B (evaluación)\n\n')
        f.write('Separación en banda 0,20-0,45 m; colisión < 0,12 m; guía '
                'perdido > 0,90 m o pelota sin detectar > 3 s; '
                '|desviación| deseada < 15°.\n\n')
        f.write('\n'.join(lineas) + '\n')
        f.write('\n## Velocidad real del guía (verdad del terreno)\n\n')
        f.write('\n'.join(bloque_vel) + '\n')
        if bloque_cor:
            f.write('\n## Correlación desviación <-> posx del blob\n\n')
            f.write('\n'.join(bloque_cor) + '\n')

    ruta_csv = os.path.join(dir_salida, 'tabla_anexo_b.csv')
    with open(ruta_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CABECERA_CSV)
        w.writeheader()
        w.writerows(filas_csv)

    print(f'Tabla Markdown: {ruta_md}')
    print(f'Tabla CSV:      {ruta_csv}')


# =====================================================================
# Figuras
# =====================================================================

def figura_series(datos, dir_salida, cfg):
    """Separación y |desviación| frente al tiempo, por configuración.

    Una línea por etiqueta (política); la banda objetivo y los umbrales del
    protocolo quedan marcados como referencia.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_s.axhspan(PROTOCOLO['BANDA_MIN_M'], PROTOCOLO['BANDA_MAX_M'],
                 color='#43a047', alpha=0.12, label='banda objetivo')
    ax_s.axhline(PROTOCOLO['COLISION_M'], color='#e53935', ls=':', lw=1,
                 label='colisión 0,12 m')
    ax_s.axhline(PROTOCOLO['PERDIDO_M'], color='#fb8c00', ls=':', lw=1,
                 label='guía perdido 0,90 m')
    ax_d.axhline(PROTOCOLO['DESV_UMBRAL_DEG'], color='0.4', ls=':', lw=1,
                 label='15°')

    dibujadas = False
    for et, por_config in datos.items():
        if cfg not in por_config:
            continue
        episodios = {}
        for fila in por_config[cfg]:
            episodios.setdefault(fila['episodio'], []).append(fila)
        for i, (_, fs) in enumerate(sorted(episodios.items(), key=lambda kv: int(kv[0]))):
            t = [float(x['t']) for x in fs]
            s = [float(x['separacion_m']) for x in fs]
            d = [abs(float(x['desviacion_deg'])) for x in fs]
            etiqueta = et if i == 0 else None   # una entrada de leyenda por política
            ax_s.plot(t, s, lw=1.4, label=etiqueta)
            ax_d.plot(t, d, lw=1.4, label=etiqueta)
            dibujadas = True
    if not dibujadas:
        plt.close(fig)
        return None

    ax_s.set_ylabel('separación (m)')
    ax_s.set_title(f'Configuración {cfg}: {CONFIGURACIONES[cfg]}')
    ax_d.set_ylabel('|desviación| (°)')
    ax_d.set_xlabel('tiempo (s)')
    ax_s.legend(fontsize=8, loc='best', ncol=2)
    for ax in (ax_s, ax_d):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    salida = os.path.join(dir_salida, f'fig_series_{cfg}.png')
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f'Figura: {salida}')


def figura_desenlaces(datos, dir_salida):
    """Barras apiladas de desenlaces por configuración (una figura por política)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    etiquetas = list(datos)
    fig, ejes = plt.subplots(1, len(etiquetas), figsize=(4.2 * len(etiquetas), 4),
                             squeeze=False)
    for ax, et in zip(ejes[0], etiquetas):
        configs = [c for c in ORDEN_CONFIG if c in datos[et]]
        base = np.zeros(len(configs))
        for causa in CAUSAS:
            vals = np.array([
                metricas(datos[et][c])['desenlaces'].get(causa, 0)
                for c in configs], dtype=float)
            if vals.sum() == 0:
                continue
            ax.bar(configs, vals, bottom=base, color=COLOR_CAUSA[causa],
                   label=ETIQUETA_CAUSA[causa])
            base += vals
        ax.set_title(et)
        ax.set_xlabel('configuración')
        ax.set_ylabel('episodios')
        ax.grid(alpha=0.3, axis='y')
    # Leyenda común (una vez).
    asas, textos = ejes[0][0].get_legend_handles_labels()
    fig.legend(asas, textos, loc='upper center', ncol=len(CAUSAS),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salida = os.path.join(dir_salida, 'fig_desenlaces.png')
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f'Figura: {salida}')


# =====================================================================
# Línea de órdenes
# =====================================================================

def main():
    a = argparse.ArgumentParser(
        description='Métricas, tablas y figuras del anexo B a partir de '
                    'p1_logs/eval_*.csv (sin simulador).')
    a.add_argument('--etiquetas', nargs='+', default=None,
                   help='etiquetas a incluir (p. ej. sac_s0 aleatorio). '
                        'Por defecto, todas las que haya en p1_logs')
    a.add_argument('--salida', default='p1_graficas',
                   help='carpeta donde guardar figuras y tablas')
    a.add_argument('--solo-tablas', action='store_true',
                   help='no genera figuras (no requiere matplotlib)')
    a.add_argument('--sin-consola', action='store_true',
                   help='no imprime las tablas y avisos en la consola')
    args = a.parse_args()

    datos = descubrir()
    if args.etiquetas:
        datos = {et: v for et, v in datos.items() if et in args.etiquetas}
    if not datos:
        raise SystemExit(
            'No hay registros de evaluación en p1_logs/eval_*.csv'
            + (f' para {args.etiquetas}' if args.etiquetas else '') + '.')

    os.makedirs(args.salida, exist_ok=True)
    print('Etiquetas: ' + ', '.join(datos))

    escribir_tablas(datos, args.salida)
    if not args.sin_consola:
        for et, por_config in datos.items():
            imprimir_consola(et, por_config)
    if not args.solo_tablas:
        for cfg in ORDEN_CONFIG:
            figura_series(datos, args.salida, cfg)
        figura_desenlaces(datos, args.salida)


if __name__ == '__main__':
    main()
