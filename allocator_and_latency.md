# Allocator and latency details

All 46 control-valid pairs are included, including numerical FAIL pairs. Peaks are the maximum across processes; latency is the median of per-process percentiles.

| Cell | BASE/PR P90 ms | BASE/PR steady reserved MiB | BASE/PR including-first reserved MiB |
|---|---|---|---|
| Neo/4096 | 1.612548 / 1.564059 | 102 / 82 | 102 / 82 |
| Nano/1000 | 1.181858 / 1.131922 | 24 / 24 | 24 / 24 |
| Nano/4096 | 1.295878 / 1.248460 | 80 / 80 | 80 / 80 |
| Mini/1000 | 1.257961 / 1.216820 | 44 / 24 | 44 / 24 |
| Mini/4096 | 1.393389 / 1.354578 | 92 / 80 | 92 / 80 |
| Neo/1000 | 1.234553 / 1.193455 | 48 / 48 | 48 / 48 |
| Air/1000 | 1.285156 / 1.252087 | 62 / 62 | 62 / 62 |
| Air/4096 | 1.768553 / 1.739366 | 116 / 96 | 116 / 96 |
| Plus/1000 | 1.461866 / 1.417291 | 92 / 92 | 92 / 92 |
| Plus/4096 | 2.304175 / 2.258597 | 166 / 146 | 166 / 146 |
| Neo/8000 | 2.060863 / 1.907076 | 166 / 168 | 166 / 168 |

Neo8000 reserved rises 166→168 MiB in this independent-master matrix. The historical C01-stacked Plus4096 case rises 136→144 MiB. These are distinct baselines and runs.
