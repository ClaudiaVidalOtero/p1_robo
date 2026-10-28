#!/usr/bin/env python3
"""
Práctica 2 — Figuras y tablas a partir de los registros (sin ROS ni simulador).

EVOLUCIÓN (p2_evol/<nombre>/: evolucion.csv, especies.csv, episodios.csv)
    fig_evolucion.png          calidad del mejor y media de la población por
                               generación; con varias semillas, media entre
                               semillas + banda min-max y cada semilla tenue.
                               Línea horizontal: controlador del reto
                               (p2_evol/referencias.csv). Línea vertical: cambio
                               de configuración del guía (C3 -> C4).
    fig_evolucion_<n>.png      una evolución: mejor y media ± desviación típica
    fig_especies_<n>.png       individuos por especie (áreas apiladas)
    fig_complejidad.png        nodos y conexiones activas del mejor por generación
    fig_causas_<n>.png         fracción de episodios por causa de fin y generación
    resumen_evolucion.md       datos para el pie: población, episodios, duración,
                               generaciones, evaluaciones, pasos y horas
RED
    fig_red_<n>.png            topología del genoma (grosor ~ |peso|, verde > 0)
    red_<n>.txt                nodos, sesgos y conexiones, para interpretarla
ANEXO B (p2_logs/eval_<etiqueta>_<C>.csv, escritos por p2_evaluar.py)
    tabla_anexo_b_p2.md/.csv   métricas de la P1 por configuración + TOTAL, y
                               las paradas de C4

USO
    python3 p2_graficas.py --evol s0 s1 s2
    python3 p2_graficas.py --red p2_evol/s0/mejor_genoma.pkl
    python3 p2_graficas.py --anexo-b                    # todas las etiquetas de p2_logs/
    python3 p2_graficas.py --anexo-b neat_s0 p1_sac_s0 reto aleatorio
"""

import argparse
import configparser
import csv
import glob
import json
import os
import pickle
import re

import numpy as np

from p1_graficas_eval import CAUSAS, COLOR_CAUSA, ETIQUETA_CAUSA, leer_csv, metricas

RUTA_EVOL = 'p2_evol'
RUTA_LOGS_P2 = 'p2_logs'
RUTA_SALIDA = 'p2_graficas'
ORDEN_CONFIG = ['C1', 'C2', 'C3', 'C4']
PATRON = re.compile(r'^eval_(?P<et>.+)_(?P<cfg>C[1-4])\.csv$')

NOMBRES_ENTRADA = ['posx', 'size', 'posy']
NOMBRES_SALIDA = ['avance a0', 'giro a1']

# Parada del guía (anexo B.1) y paradas superadas (B.4).
V_PARADA = 0.01          # m/s: el guía está parado por debajo de esto
T_PARADA_MIN = 0.5       # s: duración mínima de una parada
T_TRAS_PARADA = 10.0     # s: sin fallo tras reanudar la marcha
V_SEGUIDOR_PARADO = 0.02  # m/s: el seguidor "se detiene" si baja de esto


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# =====================================================================
# Lectura de una evolución
# =====================================================================

def leer_evolucion(nombre):
    carpeta = os.path.join(RUTA_EVOL, nombre)
    filas = leer_csv(os.path.join(carpeta, 'evolucion.csv'))
    if not filas:
        raise SystemExit(f'{carpeta}/evolucion.csv está vacío.')
    ev = {k: np.array([float(f[k]) for f in filas])
          for k in ('generacion', 'cal_mejor', 'cal_media', 'cal_desv', 'especies',
                    'nodos_mejor', 'conexiones_mejor', 'evaluaciones', 'pasos',
                    'segundos', 'duracion_s')}
    ev['guia'] = [f['guia'] for f in filas]
    ev['calidad'] = filas[-1]['calidad']
    ev['nombre'] = nombre
    ini = configparser.ConfigParser()
    ini.read(os.path.join(carpeta, 'p2_neat.ini'), encoding='utf-8')
    ev['pop_size'] = ini.getint('NEAT', 'pop_size')
    ev['semilla'] = ini.get('NEAT', 'seed', fallback='?')
    with open(os.path.join(carpeta, 'p2_config.json'), encoding='utf-8') as f:
        ev['episodios_ind'] = json.load(f)['EPISODIOS_POR_INDIVIDUO']
    ev['carpeta'] = carpeta
    return ev


def cambios_guia(ev):
    """[(generación, configuración)] cada vez que cambia la configuración del guía."""
    cambios = []
    for g, guia in zip(ev['generacion'], ev['guia']):
        if not cambios or cambios[-1][1] != guia:
            cambios.append((int(g), guia))
    return cambios


def leer_referencias():
    ruta = os.path.join(RUTA_EVOL, 'referencias.csv')
    return leer_csv(ruta) if os.path.exists(ruta) else []


def pie(evs):
    """Texto del pie de figura con el coste de las evoluciones."""
    e = evs[0]
    dur = sorted({f'{d:g}' for ev in evs for d in ev['duracion_s']})
    guias = ' -> '.join(f'{g} (gen {n})' for n, g in cambios_guia(e))
    pasos = sum(ev['pasos'][-1] for ev in evs)
    horas = sum(ev['segundos'][-1] for ev in evs) / 3600.0
    return (f'Población {e["pop_size"]}, {e["episodios_ind"]} episodio(s) por individuo, '
            f'episodios de {"/".join(dur)} s como máximo, calidad {e["calidad"]}. '
            f'Guía: {guias}. Semillas: {", ".join(str(ev["semilla"]) for ev in evs)}. '
            f'Total: {int(pasos)} pasos del entorno, {horas:.1f} h.')


# =====================================================================
# Figuras de la evolución
# =====================================================================

def figura_evolucion(evs, salida):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = min(len(ev['generacion']) for ev in evs)
    gen = evs[0]['generacion'][:n]
    if len(evs) == 1:
        ev = evs[0]
        ax.plot(ev['generacion'], ev['cal_mejor'], color='tab:blue', label='mejor')
        ax.plot(ev['generacion'], ev['cal_media'], color='tab:orange', label='media de la población')
        ax.fill_between(ev['generacion'], ev['cal_media'] - ev['cal_desv'],
                        ev['cal_media'] + ev['cal_desv'], color='tab:orange', alpha=0.2,
                        label='media ± desv. típica')
        banda = 'desviación típica de la población'
    else:
        for clave, color, nombre in (('cal_mejor', 'tab:blue', 'mejor'),
                                     ('cal_media', 'tab:orange', 'media de la población')):
            m = np.array([ev[clave][:n] for ev in evs])
            for fila in m:
                ax.plot(gen, fila, color=color, alpha=0.25, lw=0.8)
            ax.plot(gen, m.mean(axis=0), color=color, lw=2,
                    label=f'{nombre} (media de {len(evs)} semillas)')
            ax.fill_between(gen, m.min(axis=0), m.max(axis=0), color=color, alpha=0.15)
        banda = 'mínimo y máximo entre semillas'
    for ref in leer_referencias():
        if ref['controlador'] == 'reto' and ref['calidad'] == evs[0]['calidad']:
            ax.axhline(float(ref['cal_media']), color='grey', ls='--', lw=1)
            ax.text(gen[0], float(ref['cal_media']), f' reto {ref["guia"]}',
                    va='bottom', fontsize=8, color='grey')
    for g, guia in cambios_guia(evs[0])[1:]:
        ax.axvline(g - 0.5, color='black', ls=':', lw=1)
        ax.text(g - 0.4, ax.get_ylim()[1], f'guía {guia}', va='top', fontsize=8)
    ax.set_xlabel('Generación')
    ax.set_ylabel(f'Calidad {evs[0]["calidad"]} (adimensional, máx. 100)')
    ax.set_title(f'Curva de evolución (banda: {banda})')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(alpha=0.3)
    fig.text(0.01, -0.02, pie(evs), fontsize=7, wrap=True, va='top')
    fig.savefig(salida, dpi=200, bbox_inches='tight')
    plt.close(fig)


def figura_especies(ev, salida):
    plt = _plt()
    filas = leer_csv(os.path.join(ev['carpeta'], 'especies.csv'))
    gens = sorted({int(f['generacion']) for f in filas})
    especies = sorted({int(f['especie']) for f in filas})
    tam = np.zeros((len(especies), len(gens)))
    for f in filas:
        tam[especies.index(int(f['especie'])), gens.index(int(f['generacion']))] = int(f['tamano'])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.stackplot(gens, tam, labels=[f'especie {s}' for s in especies])
    ax.set_xlabel('Generación')
    ax.set_ylabel('Individuos por especie')
    ax.set_title(f'Especiación ({ev["nombre"]}, semilla {ev["semilla"]})')
    if len(especies) <= 12:
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig(salida, dpi=200, bbox_inches='tight')
    plt.close(fig)


def figura_complejidad(evs, salida):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, ev in enumerate(evs):
        c = f'C{i}'
        ax.plot(ev['generacion'], ev['nodos_mejor'], color=c, ls='-',
                label=f'{ev["nombre"]} nodos')
        ax.plot(ev['generacion'], ev['conexiones_mejor'], color=c, ls='--',
                label=f'{ev["nombre"]} conexiones')
    ax.set_xlabel('Generación')
    ax.set_ylabel('Nº (nodos sin entradas / conexiones activas)')
    ax.set_title('Complejidad del mejor individuo')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(salida, dpi=200, bbox_inches='tight')
    plt.close(fig)


def figura_causas(ev, salida):
    plt = _plt()
    ruta = os.path.join(ev['carpeta'], 'episodios.csv')
    if not os.path.exists(ruta):
        return False
    filas = leer_csv(ruta)
    gens = sorted({int(f['generacion']) for f in filas})
    frac = {c: np.zeros(len(gens)) for c in CAUSAS}
    for i, g in enumerate(gens):
        de_g = [f['desenlace'] for f in filas if int(f['generacion']) == g]
        for d in de_g:
            frac[d if d in frac else 'otra'][i] += 1.0 / len(de_g)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.stackplot(gens, [frac[c] for c in CAUSAS],
                 labels=[ETIQUETA_CAUSA[c] for c in CAUSAS],
                 colors=[COLOR_CAUSA[c] for c in CAUSAS])
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fracción de episodios')
    ax.set_ylim(0, 1)
    ax.set_title(f'Causa de fin de los episodios de evaluación ({ev["nombre"]})')
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.savefig(salida, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return True


def resumen_evolucion(evs, salida):
    lineas = ['| Evolución | Semilla | Calidad | Generaciones | Evaluaciones | Pasos | Horas | Mejor (gen.) | Especies final |',
              '|---|---|---|---|---|---|---|---|---|']
    for ev in evs:
        i = int(np.argmax(ev['cal_mejor']))
        lineas.append(f'| {ev["nombre"]} | {ev["semilla"]} | {ev["calidad"]} | '
                      f'{len(ev["generacion"])} | {int(ev["evaluaciones"][-1])} | '
                      f'{int(ev["pasos"][-1])} | {ev["segundos"][-1] / 3600:.2f} | '
                      f'{ev["cal_mejor"][i]:.1f} ({int(ev["generacion"][i])}) | '
                      f'{int(ev["especies"][-1])} |')
    lineas += ['', 'Pie: ' + pie(evs)]
    with open(salida, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas) + '\n')
    print('\n'.join(lineas))


def modo_evol(nombres, dir_salida):
    evs = [leer_evolucion(n) for n in nombres]
    figura_evolucion(evs, os.path.join(dir_salida, 'fig_evolucion.png'))
    figura_complejidad(evs, os.path.join(dir_salida, 'fig_complejidad.png'))
    for ev in evs:
        figura_evolucion([ev], os.path.join(dir_salida, f'fig_evolucion_{ev["nombre"]}.png'))
        figura_especies(ev, os.path.join(dir_salida, f'fig_especies_{ev["nombre"]}.png'))
        figura_causas(ev, os.path.join(dir_salida, f'fig_causas_{ev["nombre"]}.png'))
    resumen_evolucion(evs, os.path.join(dir_salida, 'resumen_evolucion.md'))


# =====================================================================
# Topología de la red
# =====================================================================

def dibujar_red(genoma, config, nombres_entrada, nombres_salida, ruta):
    """Red por capas (función de las indicaciones técnicas, apartado 7)."""
    plt = _plt()
    gc = config.genome_config
    entradas, salidas = list(gc.input_keys), list(gc.output_keys)
    activas = [c.key for c in genoma.connections.values() if c.enabled]
    # Profundidad de cada nodo: camino más largo desde las entradas.
    prof = {n: 0 for n in entradas}
    for _ in range(len(genoma.nodes) + 1):
        for (a, b) in activas:
            nueva = prof[a] + 1 if a in prof and a != b else -1
            if prof.get(b, -1) < nueva <= len(genoma.nodes):
                prof[b] = nueva
    ocultas = [n for n in prof if n not in entradas and n not in salidas]
    capas = [entradas]
    for p in sorted({prof[n] for n in ocultas}):
        capas.append(sorted(n for n in ocultas if prof[n] == p))
    capas.append(salidas)
    pos = {n: (i, j - (len(c) - 1) / 2)
           for i, c in enumerate(capas) for j, n in enumerate(c)}
    fig, ax = plt.subplots(figsize=(6, 4))
    for (a, b) in activas:
        if a in pos and b in pos:
            w = genoma.connections[(a, b)].weight
            color = 'tab:green' if w > 0 else 'tab:red'
            # shrink: la punta de la flecha queda fuera del círculo del nodo.
            ax.annotate('', xy=pos[b], xytext=pos[a], arrowprops=dict(
                arrowstyle='->', lw=0.5 + 1.5 * min(abs(w), 3) / 3, color=color,
                shrinkA=10, shrinkB=10))
            xa, ya = pos[a]
            xb, yb = pos[b]
            # Peso escrito cerca del destino; el punto depende de la altura
            # del origen para que no coincida donde se cruzan dos conexiones.
            frac = 0.6 + 0.08 * ya
            ax.text(xa + frac * (xb - xa), ya + frac * (yb - ya), f'{w:+.2f}',
                    fontsize=7, color=color, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none'))
    nombres = dict(zip(entradas, nombres_entrada))
    nombres.update(zip(salidas, nombres_salida))
    for n, (x, y) in pos.items():
        ax.scatter(x, y, s=300, color='lightgrey', edgecolor='black', zorder=3)
        ax.text(x, y + 0.25, nombres.get(n, str(n)), ha='center', fontsize=8)
    ax.axis('off')
    fig.savefig(ruta, dpi=200, bbox_inches='tight')
    plt.close(fig)


def describir_red(genoma, config, nombres_entrada, nombres_salida):
    """Texto con sesgos y pesos para interpretar la red."""
    gc = config.genome_config
    nombres = dict(zip(gc.input_keys, nombres_entrada))
    nombres.update(zip(gc.output_keys, nombres_salida))
    nodos, conexiones = genoma.size()
    lineas = [f'Genoma {genoma.key}: {nodos} nodos (sin entradas), {conexiones} conexiones '
              f'activas, calidad {genoma.fitness}', '', 'Nodos (activación, sesgo):']
    for k, n in sorted(genoma.nodes.items()):
        lineas.append(f'  {nombres.get(k, k)!s:>10}: {n.activation}, sesgo {n.bias:+.3f}')
    lineas += ['', 'Conexiones activas (origen -> destino: peso):']
    for (a, b), c in sorted(genoma.connections.items(), key=lambda kv: -abs(kv[1].weight)):
        if c.enabled:
            lineas.append(f'  {nombres.get(a, a)!s:>10} -> {nombres.get(b, b)!s:<10}: {c.weight:+.3f}')
    return '\n'.join(lineas)


def modo_red(ruta_genoma, ruta_ini, dir_salida):
    from p2_controladores import cargar_config_neat
    ini = ruta_ini or os.path.join(os.path.dirname(ruta_genoma), 'p2_neat.ini')
    config = cargar_config_neat(ini)
    with open(ruta_genoma, 'rb') as f:
        genoma = pickle.load(f)
    nombre = os.path.basename(os.path.dirname(os.path.abspath(ruta_genoma)))
    dibujar_red(genoma, config, NOMBRES_ENTRADA, NOMBRES_SALIDA,
                os.path.join(dir_salida, f'fig_red_{nombre}.png'))
    texto = describir_red(genoma, config, NOMBRES_ENTRADA, NOMBRES_SALIDA)
    with open(os.path.join(dir_salida, f'red_{nombre}.txt'), 'w', encoding='utf-8') as f:
        f.write(texto + '\n')
    print(texto)


# =====================================================================
# Anexo B con C4
# =====================================================================

def descubrir(ruta_logs=RUTA_LOGS_P2):
    """{etiqueta: {config: [filas]}} a partir de p2_logs/eval_*.csv."""
    datos = {}
    for ruta in sorted(glob.glob(os.path.join(ruta_logs, 'eval_*.csv'))):
        m = PATRON.match(os.path.basename(ruta))
        filas = leer_csv(ruta) if m else []
        if filas:
            datos.setdefault(m['et'], {})[m['cfg']] = filas
    return datos


def _velocidad(t, x, z):
    """Velocidad (m/s) entre filas consecutivas a partir de posiciones en mm."""
    dt = np.maximum(np.diff(t), 1e-6)
    return np.hypot(np.diff(x), np.diff(z)) / 1000.0 / dt


def paradas_episodio(pasos):
    """Paradas del guía de un episodio y si el seguidor las supera (B.4).

    Parada: tramo con velocidad del guía < V_PARADA durante más de
    T_PARADA_MIN. Superada: el seguidor se detiene (velocidad mínima durante
    la parada < V_SEGUIDOR_PARADO), sin colisión, y no falla en los
    T_TRAS_PARADA s siguientes a que el guía reanude la marcha. Una parada
    que dura hasta el final del episodio no se puede juzgar ('sin_reanudar').
    """
    t = np.array([float(f['t']) for f in pasos])
    if len(t) < 3:
        return []
    vg = _velocidad(t, np.array([float(f['guia_x_mm']) for f in pasos]),
                    np.array([float(f['guia_z_mm']) for f in pasos]))
    vs = _velocidad(t, np.array([float(f['seg_x_mm']) for f in pasos]),
                    np.array([float(f['seg_z_mm']) for f in pasos]))
    t_fin, desenlace = t[-1], pasos[-1]['desenlace']
    resultado, i = [], 0
    while i < len(vg):
        if vg[i] >= V_PARADA:
            i += 1
            continue
        j = i
        while j < len(vg) and vg[j] < V_PARADA:
            j += 1
        ts, te = t[i], t[j] if j < len(t) else t_fin
        if te - ts > T_PARADA_MIN:
            reanuda = j < len(vg)
            se_detiene = float(vs[i:j].min()) < V_SEGUIDOR_PARADO
            sin_fallo = desenlace == 'completado' or t_fin >= te + T_TRAS_PARADA
            if not reanuda:
                estado = 'sin_reanudar'
            elif se_detiene and sin_fallo:
                estado = 'superada'
            else:
                estado = 'fallida'
            resultado.append({'inicio_s': round(float(ts), 1), 'fin_s': round(float(te), 1),
                              'estado': estado})
        i = j
    return resultado


def metricas_paradas(filas):
    episodios = {}
    for f in filas:
        episodios.setdefault(f['episodio'], []).append(f)
    todas = [p for pasos in episodios.values() for p in paradas_episodio(pasos)]
    return {'paradas': len(todas),
            'superadas': sum(p['estado'] == 'superada' for p in todas),
            'sin_reanudar': sum(p['estado'] == 'sin_reanudar' for p in todas),
            'detalle': todas}


CABECERA = ['etiqueta', 'config', 'episodios', 'completados', 'pct_completados',
            'colision', 'perdido_distancia', 'perdido_sin_ver', 'otra',
            'pct_en_banda', 'separacion_media_m', 'desviacion_media_deg',
            'pct_desv_bajo_15', 'paradas', 'paradas_superadas', 'paradas_sin_reanudar']


def fila_tabla(etiqueta, config, filas, paradas=None):
    m = metricas(filas)
    d = m['desenlaces']
    fila = {'etiqueta': etiqueta, 'config': config, 'episodios': m['episodios'],
            'completados': m['completados'], 'pct_completados': round(m['pct_completados'], 1),
            'colision': d.get('colision', 0), 'perdido_distancia': d.get('perdido_distancia', 0),
            'perdido_sin_ver': d.get('perdido_sin_ver', 0), 'otra': d.get('otra', 0),
            'pct_en_banda': round(m['pct_en_banda'], 1),
            'separacion_media_m': round(m['separacion_media_m'], 3),
            'desviacion_media_deg': round(m['desviacion_media_deg'], 1),
            'pct_desv_bajo_15': round(m['pct_desv_bajo_umbral'], 1),
            'paradas': '', 'paradas_superadas': '', 'paradas_sin_reanudar': ''}
    if paradas is not None:
        fila.update(paradas=paradas['paradas'], paradas_superadas=paradas['superadas'],
                    paradas_sin_reanudar=paradas['sin_reanudar'])
    return fila


def modo_anexo_b(etiquetas, dir_salida):
    datos = descubrir()
    if etiquetas:
        datos = {e: datos[e] for e in etiquetas if e in datos}
    if not datos:
        raise SystemExit(f'No hay registros eval_*.csv en {RUTA_LOGS_P2}/.')
    tabla = []
    for et, por_config in datos.items():
        for cfg in [c for c in ORDEN_CONFIG if c in por_config]:
            par = metricas_paradas(por_config[cfg]) if cfg == 'C4' else None
            tabla.append(fila_tabla(et, cfg, por_config[cfg], par))
            if par:
                for p in par['detalle']:
                    print(f'  {et} C4 parada {p["inicio_s"]}-{p["fin_s"]} s: {p["estado"]}')
        tabla.append(fila_tabla(et, 'TOTAL', [f for v in por_config.values() for f in v]))

    with open(os.path.join(dir_salida, 'tabla_anexo_b_p2.csv'), 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CABECERA)
        w.writeheader()
        w.writerows(tabla)
    md = ['| Controlador | Config | Completados | Colisión | Perdido (dist.) | Perdido (vista) | Otra '
          '| % en banda | Sep. media (m) | Desv. media (°) | % desv. < 15° | Paradas superadas |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in tabla:
        par = (f'{r["paradas_superadas"]}/{r["paradas"]}'
               + (f' ({r["paradas_sin_reanudar"]} sin reanudar)' if r['paradas_sin_reanudar'] else '')
               if r['paradas'] != '' else '—')
        md.append(f'| {r["etiqueta"]} | {r["config"]} | {r["completados"]}/{r["episodios"]} '
                  f'({r["pct_completados"]} %) | {r["colision"]} | {r["perdido_distancia"]} | '
                  f'{r["perdido_sin_ver"]} | {r["otra"]} | {r["pct_en_banda"]} | '
                  f'{r["separacion_media_m"]} | {r["desviacion_media_deg"]} | '
                  f'{r["pct_desv_bajo_15"]} | {par} |')
    md += ['', 'Banda 0,20-0,45 m; colisión < 0,12 m; guía perdido > 0,90 m o > 3 s sin ver; '
           f'parada del guía: < {V_PARADA} m/s durante > {T_PARADA_MIN} s; superada: el '
           f'seguidor baja de {V_SEGUIDOR_PARADO} m/s sin colisión y no falla en los '
           f'{T_TRAS_PARADA:.0f} s siguientes a que el guía reanude.']
    with open(os.path.join(dir_salida, 'tabla_anexo_b_p2.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))


def main():
    a = argparse.ArgumentParser(description='Figuras y tablas de la Práctica 2.')
    a.add_argument('--evol', nargs='+', metavar='NOMBRE', help='evoluciones de p2_evol/')
    a.add_argument('--red', metavar='GENOMA.pkl', help='genoma a dibujar')
    a.add_argument('--neat-ini', default=None,
                   help='configuración de NEAT del genoma (por defecto, la de su carpeta)')
    a.add_argument('--anexo-b', nargs='*', metavar='ETIQUETA',
                   help='tablas del anexo B (sin etiquetas: todas las de p2_logs/)')
    a.add_argument('--salida', default=RUTA_SALIDA)
    args = a.parse_args()
    if not (args.evol or args.red or args.anexo_b is not None):
        a.error('indica --evol, --red y/o --anexo-b')
    os.makedirs(args.salida, exist_ok=True)
    if args.evol:
        modo_evol(args.evol, args.salida)
    if args.red:
        modo_red(args.red, args.neat_ini, args.salida)
    if args.anexo_b is not None:
        modo_anexo_b(args.anexo_b, args.salida)
    print(f'\nSalida en {args.salida}/')


if __name__ == '__main__':
    main()
