# Research scripts

Scripts are grouped by experiment family; see `docs/protocols/SCRIPT_INDEX.md` for the directory map and primary commands.

For Python scripts, prefer module execution from the repository root:

```bash
python -m scripts.<group>.<module> --help
```

This keeps the repository root on `sys.path` and makes imports from `scripts.lib` and the primary model package deterministic. Shell launchers already resolve their categorized Python entry points explicitly.

The two reported scaling experiments are archived under `scripts/scaling/`:

```bash
python -m scripts.scaling.benchmark_full_wallclock_runtime_gpu1 --help
python -m scripts.scaling.run_rld_training_population_scaling_v2 --help
```

Some baseline and scaling scripts preserve the absolute paths used for the
original run as provenance. Replace their dataset, checkpoint, and upstream
repository paths before rerunning on another machine. The primary model
package itself is portable and installed from `neurid/`.
