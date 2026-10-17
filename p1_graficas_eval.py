#!/usr/bin/env python3
"""
Práctica 1 — Métricas, tablas y figuras del anexo B.

Lee los CSV por paso que escribe p1_evaluar.py (p1_logs/eval_<etiqueta>_<C>.csv)
y genera, sin ROS 2 ni simulador:
    fig_series_C1/C2/C3.png  separación y |desviación| frente al tiempo
    fig_desenlaces.png       desenlaces por configuración y causa de fin
    fig_calibracion.png      desviación real frente a la x del blob (B.1)
    tabla_anexo_b.md/.csv    las 4 métricas por configuración y TOTAL
    tabla_final.md/.csv, fig_final_series.png   política elegida (--final)

Define PROTOCOLO, CONFIGURACIONES y COLUMNAS, que importa p1_evaluar.py.
La etiqueta 'aleatorio' sólo se incluye si se pide con --etiquetas.

USO
    python3 p1_graficas_eval.py                          # todo lo de p1_logs
    python3 p1_graficas_eval.py --etiquetas sac_s0 --salida p1_graficas
    python3 p1_graficas_eval.py --final C1=sac_s1 C2=sac_s2 C3=sac_s1
    python3 p1_graficas_eval.py --solo-tablas            # sin matplotlib
    python3 p1_graficas_eval.py --sin-consola            # sin tabla en consola

Salida en --salida (por defecto p1_graficas/):
    fig_series_C1.png, fig_series_C2.png, fig_series_C3.png
    fig_desenlaces.png, fig_calibracion.png
    tabla_anexo_b.md, tabla_anexo_b.csv
    tabla_final.md, tabla_final.csv, fig_final_series.png   (con --final)
"""

import argparse
import csv
import glob
import os
import re

import numpy as np

RUTA_LOGS = 'p1_logs'

# Parámetros del anexo B (no dependen de la calibración).
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

# Causas de fin de B.3 más 'otra' (otra terminación o CSV sin desenlace).
CAUSAS = ['completado', 'colision', 'perdido_distancia',
          'perdido_sin_ver', 'otra']
ETIQUETA_CAUSA = {
    'completado': 'completado',
    'colision': 'colisión',
    'perdido_distancia': 'guía perdido (distancia)',
    'perdido_sin_ver': 'guía perdido (vista)',
    'otra': 'otra causa / sin terminar',
}
COLOR_CAUSA = {
    'completado': '#43a047',
    'colision': '#e53935',
    'perdido_distancia': '#fb8c00',
    'perdido_sin_ver': '#fdd835',
    'otra': '#9e9e9e',
}

# Etiquetas que no entran en la comparativa salvo que se pidan con --etiquetas.
EXCLUIDAS = {'aleatorio'}

PATRON = re.compile(r'^eval_(?P<et>.+)_(?P<cfg>C[123])\.csv$')


# =====================================================================
# Lectura de los registros
# =====================================================================

def leer_csv(ruta):
    with open(ruta, newline='', encoding='utf-8') as f:
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
        d = pasos[-1]['desenlace']
        if d not in CAUSAS:
            d = 'otra'      # otra terminación o episodio sin desenlace
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
    """(r, nº de pasos) entre la desviación real y la posx del blob (B.1).

    Con una buena calibración |r| es alto; el signo es negativo por el
    convenio de ángulos del simulador.
    """
    pares = [(float(f['desviacion_deg']), float(f['blob_posx']))
             for f in filas if f['visible'] == '1']
    if len(pares) < 10:
        return None
    d, x = np.array(pares).T
    if d.std() == 0 or x.std() == 0:
        return None
    return float(np.corrcoef(d, x)[0, 1]), len(pares)


def episodios_incompletos(datos):
    """Episodios cuyo CSV acaba sin desenlace (evaluación abortada)."""
    avisos = []
    for et, por_config in datos.items():
        for cfg, filas in por_config.items():
            episodios = {}
            for fila in filas:
                episodios.setdefault(fila['episodio'], []).append(fila)
            for ep, pasos in episodios.items():
                if not pasos[-1]['desenlace']:
                    avisos.append(f'{et} / {cfg} ep {ep}: sin desenlace, '
                                  f'acaba a los {float(pasos[-1]["t"]):.1f} s')
    return avisos


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
        'otra': d.get('otra', 0),
        'pct_en_banda': round(m['pct_en_banda'], 1),
        'separacion_media_m': round(m['separacion_media_m'], 3),
        'desviacion_media_deg': round(m['desviacion_media_deg'], 1),
        'pct_desv_bajo_umbral': round(m['pct_desv_bajo_umbral'], 1),
    }


CABECERA_CSV = ['etiqueta', 'config', 'episodios', 'completados',
                'pct_completados', 'colision', 'perdido_distancia',
                'perdido_sin_ver', 'otra', 'pct_en_banda', 'separacion_media_m',
                'desviacion_media_deg', 'pct_desv_bajo_umbral']


def imprimir_tabla(etiqueta, filas):
    titulo = f'ANEXO B — {etiqueta}'
    print('\n' + titulo)
    print('=' * len(titulo))
    print(f'{"":8s}{"episodios":>10s}{"complet.":>10s}{"colisión":>10s}'
          f'{"perd.dist":>10s}{"perd.vista":>11s}{"otra":>6s}{"% banda":>9s}'
          f'{"sep. media":>11s}{"|desv| °":>10s}{"% <15°":>8s}')
    for nombre, m in filas.items():
        d = m['desenlaces']
        print(f'{nombre:8s}{m["episodios"]:>10d}'
              f'{m["completados"]:>5d} ({m["pct_completados"]:3.0f}%)'
              f'{d.get("colision", 0):>9d}{d.get("perdido_distancia", 0):>10d}'
              f'{d.get("perdido_sin_ver", 0):>11d}{d.get("otra", 0):>6d}'
              f'{m["pct_en_banda"]:>8.1f}%'
              f'{m["separacion_media_m"]:>9.3f} m{m["desviacion_media_deg"]:>10.1f}'
              f'{m["pct_desv_bajo_umbral"]:>7.1f}%')
    print('\nCausas: colisión (< 0,12 m), guía perdido por distancia (> 0,90 m) '
          'o por pelota sin detectar > 3 s; otra = otra terminación o '
          'episodio sin terminar (no cuenta como completado).')


def avisar_velocidad_guia(por_config_filas):
    """Velocidad real del guía por episodio; avisa si parece que no se aplicó
    la configuración del guía (C1 igual que otra, o C1 no constante)."""
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
    # Sólo se compara C1: C2 y C3 comparten rango y pueden tener la misma media.
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
        print(f'Correlación desviación real <-> posx del blob: {cal[0]:+.3f} '
              f'({cal[1]} pasos con pelota visible). |r| debe ser alto '
              '(signo esperado: negativo).')


CABECERA_MD = [
    '| Etiqueta | Config | Episodios | Completados | Colisión | '
    'Perd. dist. | Perd. vista | Otra | % en banda | Sep. media (m) | '
    '\\|desv\\| (°) | % <15° |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]


def _linea_md(et, v):
    return (f"| {et} | {v['config']} | {v['episodios']} | "
            f"{v['completados']} ({v['pct_completados']:.0f}%) | "
            f"{v['colision']} | {v['perdido_distancia']} | "
            f"{v['perdido_sin_ver']} | {v['otra']} | {v['pct_en_banda']:.1f}% | "
            f"{v['separacion_media_m']:.3f} | "
            f"{v['desviacion_media_deg']:.1f} | "
            f"{v['pct_desv_bajo_umbral']:.1f}% |")


def _escribir_csv(ruta, filas_csv):
    with open(ruta, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CABECERA_CSV)
        w.writeheader()
        w.writerows(filas_csv)


def escribir_tablas(datos, dir_salida):
    """Tabla del anexo B (todas las etiquetas) en Markdown y CSV."""
    filas_csv = []
    lineas = list(CABECERA_MD)
    bloque_vel, bloque_cor = [], []
    for et, por_config in datos.items():
        for nombre, m in metricas_etiqueta(por_config).items():
            v = _fila_tabla(nombre, m)
            filas_csv.append({'etiqueta': et, **v})
            lineas.append(_linea_md(et, v))

        for cfg, ss in sorted(por_config.items(),
                              key=lambda kv: ORDEN_CONFIG.index(kv[0])):
            media, perfil = velocidad_guia(ss)
            if media is not None:
                bloque_vel.append(
                    f'- {et} / {cfg}: {media:.3f} m/s   (por 10 s: '
                    + ' '.join(f'{x:.2f}' for x in perfil) + ')')

        cal =comprobar_calibracion([f for v in por_config.values() for f in v])
        if cal is not None:
            bloque_cor.append(
                f'- {et}: r = {cal[0]:+.3f} ({cal[1]} pasos con pelota '
                'visible). |r| debe ser alto (signo esperado: negativo).')

    ruta_md = os.path.join(dir_salida, 'tabla_anexo_b.md')
    with open(ruta_md, 'w', encoding='utf-8') as f:
        f.write('# Tabla del anexo B (evaluación)\n\n')
        f.write(LEYENDA_MD)
        f.write('\n'.join(lineas) + '\n')
        f.write('\n## Velocidad real del guía (verdad del terreno)\n\n')
        f.write('\n'.join(bloque_vel) + '\n')
        if bloque_cor:
            f.write('\n## Correlación desviación <-> posx del blob\n\n')
            f.write('\n'.join(bloque_cor) + '\n')
        incompletos = episodios_incompletos(datos)
        if incompletos:
            f.write('\n## Episodios sin terminar (repetir la evaluación)\n\n')
            f.write('\n'.join(f'- {x}' for x in incompletos) + '\n')

    ruta_csv = os.path.join(dir_salida, 'tabla_anexo_b.csv')
    _escribir_csv(ruta_csv, filas_csv)

    print(f'Tabla Markdown: {ruta_md}')
    print(f'Tabla CSV:      {ruta_csv}')


LEYENDA_MD = ('Separación en banda 0,20-0,45 m; colisión < 0,12 m; guía '
              'perdido > 0,90 m o pelota sin detectar > 3 s; '
              '|desviación| deseada < 15°. "Otra" = otra terminación o '
              'episodio sin terminar (no cuenta como completado).\n\n')


def parsear_final(pares, datos):
    """'C1=sac_s1 C2=...' -> {config: etiqueta}, comprobando que existan."""
    final = {}
    for par in pares:
        cfg, sep, et = par.partition('=')
        if not sep or cfg not in CONFIGURACIONES:
            raise SystemExit(f'--final: "{par}" no tiene la forma C1=etiqueta')
        if et not in datos or cfg not in datos[et]:
            raise SystemExit(f'--final: no existe p1_logs/eval_{et}_{cfg}.csv')
        final[cfg] = et
    return dict(sorted(final.items(), key=lambda kv: ORDEN_CONFIG.index(kv[0])))


def escribir_tabla_final(datos, final, dir_salida):
    """Tabla de la solución final: una política por configuración y su TOTAL."""
    filas_csv, lineas = [], list(CABECERA_MD)
    todas = []
    for cfg, et in final.items():
        v = _fila_tabla(cfg, metricas(datos[et][cfg]))
        filas_csv.append({'etiqueta': et, **v})
        lineas.append(_linea_md(et, v))
        todas.extend(datos[et][cfg])
    v = _fila_tabla('TOTAL', metricas(todas))
    filas_csv.append({'etiqueta': 'final', **v})
    lineas.append(_linea_md('final', v))

    ruta_md = os.path.join(dir_salida, 'tabla_final.md')
    with open(ruta_md, 'w', encoding='utf-8') as f:
        f.write('# Tabla del anexo B — solución final por configuración\n\n')
        f.write(LEYENDA_MD)
        f.write('\n'.join(lineas) + '\n')
    ruta_csv = os.path.join(dir_salida, 'tabla_final.csv')
    _escribir_csv(ruta_csv, filas_csv)
    print(f'Tabla final:    {ruta_md}')
    print(f'Tabla final:    {ruta_csv}')


# =====================================================================
# Figuras
# =====================================================================

def _dibujar_series(ax_s, ax_d, series):
    """Separación (ax_s) y |desviación| (ax_d) frente al tiempo, con la banda y
    los umbrales del protocolo. series = [(etiqueta, filas)]. Devuelve si se
    dibujó algo."""
    ax_s.axhspan(PROTOCOLO['BANDA_MIN_M'], PROTOCOLO['BANDA_MAX_M'],
                 color='#43a047', alpha=0.12, label='banda objetivo')
    ax_s.axhline(PROTOCOLO['COLISION_M'], color='#e53935', ls=':', lw=1,
                 label='colisión 0,12 m')
    ax_s.axhline(PROTOCOLO['PERDIDO_M'], color='#fb8c00', ls=':', lw=1,
                 label='guía perdido 0,90 m')
    ax_d.axhline(PROTOCOLO['DESV_UMBRAL_DEG'], color='0.4', ls=':', lw=1,
                 label='15°')
    dibujadas = False
    for et, filas in series:
        episodios = {}
        for fila in filas:
            episodios.setdefault(fila['episodio'], []).append(fila)
        for i, (_, fs) in enumerate(sorted(episodios.items(), key=lambda kv: int(kv[0]))):
            t = [float(x['t']) for x in fs]
            s = [float(x['separacion_m']) for x in fs]
            d = [abs(float(x['desviacion_deg'])) for x in fs]
            etiqueta = et if i == 0 else None   # una entrada de leyenda por política
            ax_s.plot(t, s, lw=1.4, label=etiqueta)
            ax_d.plot(t, d, lw=1.4, label=etiqueta)
            dibujadas = True
    ax_s.set_ylabel('separación (m)')
    ax_d.set_ylabel('|desviación| (°)')
    ax_d.set_xlabel('tiempo (s)')
    for ax in (ax_s, ax_d):
        ax.set_xlim(0, PROTOCOLO['DURACION_S'])
        ax.grid(alpha=0.3)
    return dibujadas


def figura_series(datos, dir_salida, cfg):
    """Series de todas las políticas en la configuración cfg."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax_s, ax_d) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    dibujadas = _dibujar_series(ax_s, ax_d, [(et, v[cfg]) for et, v in datos.items()
                                             if cfg in v])
    if not dibujadas:
        plt.close(fig)
        return None
    ax_s.set_title(f'Configuración {cfg}: {CONFIGURACIONES[cfg]}')
    ax_s.legend(fontsize=8, loc='best', ncol=2)
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
        ax.yaxis.get_major_locator().set_params(integer=True)
    # Leyenda común con todas las causas.
    from matplotlib.patches import Patch
    asas = [Patch(color=COLOR_CAUSA[c], label=ETIQUETA_CAUSA[c]) for c in CAUSAS]
    fig.legend(handles=asas, loc='upper center', ncol=len(CAUSAS),
               fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    salida = os.path.join(dir_salida, 'fig_desenlaces.png')
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f'Figura: {salida}')


def figura_calibracion(datos, dir_salida):
    """Desviación real frente a la x del blob en los pasos con la pelota
    visible, con su ajuste lineal y r (B.1)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    todos = []
    for et, por_config in datos.items():
        pares = [(float(f['desviacion_deg']), float(f['blob_posx']))
                 for v in por_config.values() for f in v if f['visible'] == '1']
        if not pares:
            continue
        d, x = np.array(pares).T
        ax.scatter(d, x, s=6, alpha=0.4, label=et)
        todos.extend(pares)
    if len(todos) < 10:
        plt.close(fig)
        return None
    d, x = np.array(todos).T
    pend, orig = np.polyfit(d, x, 1)
    r = np.corrcoef(d, x)[0, 1]
    xs = np.linspace(d.min(), d.max(), 2)
    ax.plot(xs, pend * xs + orig, color='k', lw=1.2,
            label=f'ajuste: posx = {pend:+.4f}·desv {orig:+.3f}  (r = {r:+.2f})')
    ax.set_xlabel('desviación real (°)')
    ax.set_ylabel('posx del blob (normalizada)')
    ax.set_title(f'Calibración: desviación real vs. x del blob ({len(todos)} pasos)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    salida = os.path.join(dir_salida, 'fig_calibracion.png')
    fig.savefig(salida, dpi=150)
    plt.close(fig)
    print(f'Figura: {salida}')


def figura_final_series(datos, final, dir_salida):
    """Separación y |desviación| de la solución final, una fila por configuración."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ejes = plt.subplots(len(final), 2, figsize=(11, 2.8 * len(final)),
                             squeeze=False)
    for (ax_s, ax_d), (cfg, et) in zip(ejes, final.items()):
        _dibujar_series(ax_s, ax_d, [(et, datos[et][cfg])])
        m = metricas(datos[et][cfg])
        ax_s.set_title(f'{cfg} ({et}): {m["pct_en_banda"]:.0f} % en banda, '
                       f'sep. media {m["separacion_media_m"]:.2f} m', fontsize=9)
        ax_d.set_title(f'{cfg}: |desv| media {m["desviacion_media_deg"]:.1f}°, '
                       f'{m["pct_desv_bajo_umbral"]:.0f} % < 15°', fontsize=9)
    ejes[0][0].legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    salida = os.path.join(dir_salida, 'fig_final_series.png')
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
                   help='etiquetas a incluir (p. ej. sac_s0 sac_s1). Por '
                        'defecto, todas las de p1_logs salvo "aleatorio"')
    a.add_argument('--final', nargs='+', default=None, metavar='Cn=ETIQUETA',
                   help='política elegida por configuración, p. ej. '
                        'C1=sac_s1 C2=sac_s2 C3=sac_s1')
    a.add_argument('--salida', default='p1_graficas',
                   help='carpeta donde guardar figuras y tablas')
    a.add_argument('--solo-tablas', action='store_true',
                   help='no genera figuras (no requiere matplotlib)')
    a.add_argument('--sin-consola', action='store_true',
                   help='no imprime las tablas y avisos en la consola')
    args = a.parse_args()

    todos = descubrir()
    final = parsear_final(args.final, todos) if args.final else None
    if args.etiquetas:
        datos = {et: v for et, v in todos.items() if et in args.etiquetas}
    else:
        datos ={et: v for et, v in todos.items() if et not in EXCLUIDAS}
    if not datos:
        raise SystemExit(
            'No hay registros de evaluación en p1_logs/eval_*.csv'
            + (f' para {args.etiquetas}' if args.etiquetas else '') + '.')

    os.makedirs(args.salida, exist_ok=True)
    print('Etiquetas: ' + ', '.join(datos))

    escribir_tablas(datos, args.salida)
    if final:
        escribir_tabla_final(todos, final, args.salida)
    if not args.sin_consola:
        for et, por_config in datos.items():
            imprimir_consola(et, por_config)
        if final:
            imprimir_tabla('solución final (' + ', '.join(
                f'{c}={e}' for c, e in final.items()) + ')',
                {**{c: metricas(todos[e][c]) for c, e in final.items()},
                 'TOTAL': metricas([f for c, e in final.items() for f in todos[e][c]])})
    for aviso in episodios_incompletos(datos):
        print(f'AVISO: {aviso}. Cuenta como "otra"; repite esa evaluación.')
    if not args.solo_tablas:
        for cfg in ORDEN_CONFIG:
            figura_series(datos, args.salida, cfg)
        figura_desenlaces(datos, args.salida)
        figura_calibracion(datos, args.salida)
        if final:
            figura_final_series(todos, final, args.salida)


if __name__ == '__main__':
    main()
