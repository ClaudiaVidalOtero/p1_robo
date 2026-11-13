| Controlador | Tamaño | Parámetros | Pasos del entorno | Horas de reloj |
|---|---|---|---|---|
| P1 SAC (p1_modelo_sac_s0.zip) | red fija, 338184 parámetros en la política | 67844 (actor) | 19804 (1 entrenamiento/s) | 2.45 |
| P2 NEAT (p2_evol/s8/mejor_genoma_C3.pkl) | 2 nodos, 8 conexiones activas | 10 (pesos + sesgos) | 63036 (1 evolución/es) | 11.11 |

Pasos y horas: suma de todos los entrenamientos/evoluciones indicados (--monitor, --evol). Parámetros de NEAT: conexiones activas + sesgos de los nodos no de entrada (la respuesta no muta).
