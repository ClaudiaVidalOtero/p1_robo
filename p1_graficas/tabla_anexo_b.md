# Tabla del anexo B (evaluación)

Separación en banda 0,20-0,45 m; colisión < 0,12 m; guía perdido > 0,90 m o pelota sin detectar > 3 s; |desviación| deseada < 15°. "Otra" = otra terminación o episodio sin terminar (no cuenta como completado).

| Etiqueta | Config | Episodios | Completados | Colisión | Perd. dist. | Perd. vista | Otra | % en banda | Sep. media (m) | \|desv\| (°) | % <15° |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_s0 | C1 | 1 | 0 (0%) | 1 | 0 | 0 | 0 | 42.9% | 0.337 | 10.6 | 57.1% |
| ppo_s0 | C2 | 1 | 0 (0%) | 0 | 0 | 1 | 0 | 75.5% | 0.426 | 11.7 | 76.1% |
| ppo_s0 | C3 | 1 | 0 (0%) | 0 | 0 | 1 | 0 | 91.3% | 0.400 | 24.1 | 52.2% |
| ppo_s0 | TOTAL | 3 | 0 (0%) | 1 | 0 | 2 | 0 | 76.2% | 0.420 | 13.2 | 72.5% |
| sac_s0 | C1 | 1 | 0 (0%) | 0 | 0 | 1 | 0 | 30.8% | 0.289 | 19.0 | 46.2% |
| sac_s0 | C2 | 1 | 0 (0%) | 0 | 1 | 0 | 0 | 32.4% | 0.523 | 29.3 | 52.9% |
| sac_s0 | C3 | 1 | 0 (0%) | 0 | 1 | 0 | 0 | 51.6% | 0.482 | 32.5 | 51.6% |
| sac_s0 | TOTAL | 3 | 0 (0%) | 0 | 2 | 1 | 0 | 39.7% | 0.468 | 28.8 | 51.3% |
| sac_s1 | C1 | 1 | 1 (100%) | 0 | 0 | 0 | 0 | 99.3% | 0.353 | 3.6 | 98.0% |
| sac_s1 | C2 | 1 | 0 (0%) | 0 | 1 | 0 | 0 | 68.1% | 0.441 | 14.7 | 86.1% |
| sac_s1 | C3 | 1 | 0 (0%) | 0 | 0 | 1 | 0 | 78.1% | 0.406 | 12.2 | 88.6% |
| sac_s1 | TOTAL | 3 | 1 (33%) | 0 | 1 | 1 | 0 | 89.7% | 0.378 | 7.2 | 94.0% |
| sac_s2 | C1 | 1 | 0 (0%) | 0 | 0 | 1 | 0 | 97.8% | 0.341 | 9.9 | 87.2% |
| sac_s2 | C2 | 1 | 1 (100%) | 0 | 0 | 0 | 0 | 64.1% | 0.436 | 5.4 | 99.7% |
| sac_s2 | C3 | 1 | 0 (0%) | 0 | 0 | 0 | 1 | 70.7% | 0.417 | 13.7 | 58.5% |
| sac_s2 | TOTAL | 3 | 1 (33%) | 0 | 0 | 1 | 1 | 76.2% | 0.402 | 7.6 | 92.1% |

## Velocidad real del guía (verdad del terreno)

- ppo_s0 / C1: 0.033 m/s   (por 10 s: 0.03)
- ppo_s0 / C2: 0.137 m/s   (por 10 s: 0.12 0.12 0.12 0.15 0.15 0.17 0.15)
- ppo_s0 / C3: 0.110 m/s   (por 10 s: 0.11)
- sac_s0 / C1: 0.040 m/s   (por 10 s: 0.04)
- sac_s0 / C2: 0.162 m/s   (por 10 s: 0.17 0.15)
- sac_s0 / C3: 0.142 m/s   (por 10 s: 0.13 0.19)
- sac_s1 / C1: 0.050 m/s   (por 10 s: 0.04 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05)
- sac_s1 / C2: 0.131 m/s   (por 10 s: 0.17 0.08 0.14)
- sac_s1 / C3: 0.128 m/s   (por 10 s: 0.13 0.14 0.11 0.12 0.15)
- sac_s2 / C1: 0.049 m/s   (por 10 s: 0.04 0.05 0.05 0.05 0.05 0.05 0.05 0.05)
- sac_s2 / C2: 0.151 m/s   (por 10 s: 0.18 0.08 0.14 0.12 0.12 0.22 0.16 0.16 0.22 0.16 0.12 0.13 0.22)
- sac_s2 / C3: 0.140 m/s   (por 10 s: 0.10 0.20)

## Correlación desviación <-> posx del blob

- ppo_s0: r = -0.890 (171 pasos con pelota visible). |r| debe ser alto (signo esperado: negativo).
- sac_s0: r = -0.735 (50 pasos con pelota visible). |r| debe ser alto (signo esperado: negativo).
- sac_s1: r = -0.891 (471 pasos con pelota visible). |r| debe ser alto (signo esperado: negativo).
- sac_s2: r = -0.901 (508 pasos con pelota visible). |r| debe ser alto (signo esperado: negativo).

## Episodios sin terminar (repetir la evaluación)

- sac_s2 / C3 ep 1: sin desenlace, acaba a los 16.5 s
