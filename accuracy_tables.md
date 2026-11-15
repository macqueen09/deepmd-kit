# Ordinary Torch / BASE / PR accuracy

Primary-input self-repeat min–max ranges, five repeats per process. These are self-repeat errors, NOT compression errors against Torch.
Units: energy/atom and virial/atom in eV/atom; force_max in eV/Å; stress in eV/Å³.
Torch has two deterministic-reference processes per cell; BASE/PR have four, or six for Neo4096. Observed Torch zeros are not a universal determinism claim.

| Model/atoms | Metric | Ordinary Torch | BASE | PR |
|---|---|---|---|---|
| Neo/4096 | energy_per_atom | 0.000000e+00–0.000000e+00 | 2.328306e-10–4.610047e-08 | 1.513399e-09–4.970934e-08 |
| Neo/4096 | force_rmse | 0.000000e+00–0.000000e+00 | 1.920985e-06–2.002909e-06 | 1.931576e-06–1.989329e-06 |
| Neo/4096 | force_max | 0.000000e+00–0.000000e+00 | 7.195398e-06–9.271316e-06 | 7.493887e-06–9.398209e-06 |
| Neo/4096 | virial_per_atom | 0.000000e+00–0.000000e+00 | 1.072021e-08–1.237877e-07 | 2.387515e-08–1.515627e-07 |
| Neo/4096 | stress_max | 0.000000e+00–0.000000e+00 | 1.889662e-09–2.182017e-08 | 3.616441e-09–2.671610e-08 |
| Nano/1000 | energy_per_atom | 0.000000e+00–0.000000e+00 | 9.536743e-10–2.727509e-07 | 1.907349e-08–2.164841e-07 |
| Nano/1000 | force_rmse | 0.000000e+00–0.000000e+00 | 3.193263e-06–3.370047e-06 | 3.150967e-06–3.318457e-06 |
| Nano/1000 | force_max | 0.000000e+00–0.000000e+00 | 1.075654e-05–1.390022e-05 | 1.102476e-05–1.518335e-05 |
| Nano/1000 | virial_per_atom | 0.000000e+00–0.000000e+00 | 3.588200e-08–4.521459e-07 | 7.156655e-08–4.876554e-07 |
| Nano/1000 | stress_max | 0.000000e+00–0.000000e+00 | 6.324952e-09–7.970018e-08 | 1.169775e-08–8.595946e-08 |
| Nano/4096 | energy_per_atom | 0.000000e+00–0.000000e+00 | 4.656613e-10–9.872019e-08 | 1.164153e-09–9.872019e-08 |
| Nano/4096 | force_rmse | 0.000000e+00–0.000000e+00 | 3.253862e-06–3.337766e-06 | 3.257438e-06–3.362581e-06 |
| Nano/4096 | force_max | 0.000000e+00–0.000000e+00 | 1.239823e-05–1.511769e-05 | 1.205527e-05–1.514726e-05 |
| Nano/4096 | virial_per_atom | 0.000000e+00–0.000000e+00 | 2.407887e-08–2.247198e-07 | 3.087189e-08–2.416755e-07 |
| Nano/4096 | stress_max | 0.000000e+00–0.000000e+00 | 4.152783e-09–3.961156e-08 | 5.441816e-09–4.260036e-08 |
| Mini/1000 | energy_per_atom | 0.000000e+00–0.000000e+00 | 3.337860e-09–9.918213e-08 | 4.768372e-10–8.916855e-08 |
| Mini/1000 | force_rmse | 0.000000e+00–0.000000e+00 | 2.116186e-06–2.267348e-06 | 2.115592e-06–2.250872e-06 |
| Mini/1000 | force_max | 0.000000e+00–0.000000e+00 | 7.173279e-06–9.774463e-06 | 7.284107e-06–9.193551e-06 |
| Mini/1000 | virial_per_atom | 0.000000e+00–0.000000e+00 | 3.674999e-08–2.405979e-07 | 2.638251e-08–2.623312e-07 |
| Mini/1000 | stress_max | 0.000000e+00–0.000000e+00 | 5.277579e-09–4.241041e-08 | 4.560506e-09–4.624136e-08 |
| Mini/4096 | energy_per_atom | 0.000000e+00–0.000000e+00 | 1.047738e-09–5.366746e-08 | 0.000000e+00–4.284084e-08 |
| Mini/4096 | force_rmse | 0.000000e+00–0.000000e+00 | 2.410863e-06–2.491531e-06 | 2.405475e-06–2.503602e-06 |
| Mini/4096 | force_max | 0.000000e+00–0.000000e+00 | 9.103911e-06–1.078541e-05 | 9.237556e-06–1.328089e-05 |
| Mini/4096 | virial_per_atom | 0.000000e+00–0.000000e+00 | 1.300600e-08–1.769740e-07 | 1.673425e-08–1.013505e-07 |
| Mini/4096 | stress_max | 0.000000e+00–0.000000e+00 | 9.740889e-10–3.119537e-08 | 2.949761e-09–1.786514e-08 |
| Neo/1000 | energy_per_atom | 0.000000e+00–0.000000e+00 | 2.861023e-09–6.341934e-08 | 2.861023e-09–9.679794e-08 |
| Neo/1000 | force_rmse | 0.000000e+00–0.000000e+00 | 1.685580e-06–1.792604e-06 | 1.659227e-06–1.764255e-06 |
| Neo/1000 | force_max | 0.000000e+00–0.000000e+00 | 5.438225e-06–7.774914e-06 | 5.492009e-06–7.583527e-06 |
| Neo/1000 | virial_per_atom | 0.000000e+00–0.000000e+00 | 2.154708e-08–2.020523e-07 | 2.714992e-08–1.778528e-07 |
| Neo/1000 | stress_max | 0.000000e+00–0.000000e+00 | 3.798123e-09–3.561594e-08 | 4.785740e-09–3.135028e-08 |
| Air/1000 | energy_per_atom | 0.000000e+00–0.000000e+00 | 0.000000e+00–3.623962e-08 | 2.861023e-09–9.059906e-08 |
| Air/1000 | force_rmse | 0.000000e+00–0.000000e+00 | 1.367460e-06–1.472897e-06 | 1.331835e-06–1.469679e-06 |
| Air/1000 | force_max | 0.000000e+00–0.000000e+00 | 4.698755e-06–6.669201e-06 | 4.329951e-06–6.138580e-06 |
| Air/1000 | virial_per_atom | 0.000000e+00–0.000000e+00 | 1.909211e-08–2.222881e-07 | 1.592003e-08–2.821013e-07 |
| Air/1000 | stress_max | 0.000000e+00–0.000000e+00 | 2.489071e-09–3.918292e-08 | 2.716604e-09–4.972626e-08 |
| Air/4096 | energy_per_atom | 0.000000e+00–0.000000e+00 | 2.328306e-10–2.328306e-08 | 1.164153e-10–3.550667e-08 |
| Air/4096 | force_rmse | 0.000000e+00–0.000000e+00 | 1.533444e-06–1.572136e-06 | 1.547811e-06–1.589209e-06 |
| Air/4096 | force_max | 0.000000e+00–0.000000e+00 | 5.648937e-06–7.148134e-06 | 5.675713e-06–8.946285e-06 |
| Air/4096 | virial_per_atom | 0.000000e+00–0.000000e+00 | 8.549932e-09–1.729450e-07 | 1.101216e-08–1.093194e-07 |
| Air/4096 | stress_max | 0.000000e+00–0.000000e+00 | 1.507104e-09–3.048517e-08 | 1.549989e-09–1.926984e-08 |
| Plus/1000 | energy_per_atom | 0.000000e+00–0.000000e+00 | 3.814697e-09–1.058578e-07 | 9.536743e-10–5.531311e-08 |
| Plus/1000 | force_rmse | 0.000000e+00–0.000000e+00 | 1.531944e-06–1.683019e-06 | 1.567077e-06–1.655507e-06 |
| Plus/1000 | force_max | 0.000000e+00–0.000000e+00 | 5.462673e-06–7.387716e-06 | 5.133683e-06–7.124618e-06 |
| Plus/1000 | virial_per_atom | 0.000000e+00–0.000000e+00 | 2.460368e-08–2.103895e-07 | 1.871213e-08–2.631918e-07 |
| Plus/1000 | stress_max | 0.000000e+00–0.000000e+00 | 3.337969e-09–3.708555e-08 | 3.298405e-09–4.639305e-08 |
| Plus/4096 | energy_per_atom | 0.000000e+00–0.000000e+00 | 2.328306e-10–2.770685e-08 | 4.656613e-10–3.864989e-08 |
| Plus/4096 | force_rmse | 0.000000e+00–0.000000e+00 | 1.855416e-06–1.903907e-06 | 1.850569e-06–1.917732e-06 |
| Plus/4096 | force_max | 0.000000e+00–0.000000e+00 | 6.577931e-06–9.801239e-06 | 6.936491e-06–9.438023e-06 |
| Plus/4096 | virial_per_atom | 0.000000e+00–0.000000e+00 | 8.112693e-09–1.832827e-07 | 7.465133e-09–1.326384e-07 |
| Plus/4096 | stress_max | 0.000000e+00–0.000000e+00 | 1.276327e-09–3.230742e-08 | 1.315886e-09–2.338030e-08 |
| Neo/8000 | energy_per_atom | 0.000000e+00–0.000000e+00 | 2.026558e-09–3.153086e-08 | 1.192093e-10–2.115965e-08 |
| Neo/8000 | force_rmse | 0.000000e+00–0.000000e+00 | 1.937152e-06–1.986420e-06 | 1.936569e-06–1.982548e-06 |
| Neo/8000 | force_max | 0.000000e+00–0.000000e+00 | 7.508090e-06–9.926036e-06 | 7.782830e-06–1.024897e-05 |
| Neo/8000 | virial_per_atom | 0.000000e+00–0.000000e+00 | 8.191448e-09–9.285752e-08 | 1.544715e-08–8.945074e-08 |
| Neo/8000 | stress_max | 0.000000e+00–0.000000e+00 | 1.443914e-09–1.636808e-08 | 1.842917e-09–1.576756e-08 |

## Gate counts

Original optimization: 41/46 PASS. Four failed pairs have direct numerical failures; one has passing cross-arm case metrics but propagates self-repeat FAIL.
Against ordinary Torch: 1,472/1,472 numeric cases pass the compression gate; overall 178/184 PASS, with six artifact-self-status-only failures.
BASE self-repeat FAIL: 2/46 processes; PR: 1/46. Neither changes the original paired failures.
Extra dynamic requests: 32/33 PASS; atomic outputs: 12/12 PASS. Nano multi-frame virial has elementwise ratio 1.267517.
Fixed identical Nano graphs: 72 cross-arm E/F/V arrays bitwise identical; shared-search graph fields: 24/24 PASS. These diagnostics do not override pipeline failures.

## Strict paired failures

| Cell / round | Direct failure |
|---|---|
| Nano1000/r0 | seed7 virial ratio 1.0873797; rotated E/atom 1.0281801e-6; BASE self-repeat also FAIL |
| Plus1000/r0 | primary virial elementwise ratio 1.0299906 |
| Nano4096/r1 | all eight paired numeric cases PASS; BASE and PR self-repeat status FAIL |
| Mini1000/r1 | seed7 virial elementwise ratio 1.1455652 |
| Nano4096/r2 | rotated E/atom 1.0728836e-6 |

No thresholds were relaxed. Root causes are not fully isolated; baseline variability and fixed-graph equality do not rule out an effect on independently rebuilt graph order/reductions.
Full cross-process ranges, exact comparison commands and gate reports are in results.json.gz.
