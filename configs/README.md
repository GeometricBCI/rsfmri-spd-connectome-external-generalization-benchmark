# Configuration hierarchy

The stable benchmark command reads one complete YAML file with
`--config`. Paths in YAML are resolved relative to that YAML file. The
subdirectories separate examples and protocol notes without implying that
restricted dataset files are shipped here.

Resolution precedence is:

1. explicit command-line option;
2. YAML value;
3. `RSFMRI_SPD_*` environment variable;
4. documented local default (`../rsfmri_spd_data/` and
   `results/benchmark_csv/`).

See `examples/synthetic_dry_run.yaml` for the complete supported schema.
Dataset and atlas names for benchmark execution are validated by
`spd_connectome_benchmark.datasets`; unknown names are rejected.

The `release/` subdirectory is a separate, fail-closed Zenodo packaging
configuration. In that policy, unknown datasets are non-public by default.
Benchmark configuration does not grant dataset-release permission.

`RSFMRI_SPD_OUTPUT_ROOT` is a shared results root and contributes its
`benchmark_csv/` child. `RSFMRI_SPD_BENCHMARK_OUTPUT_ROOT`, YAML `output_root`,
and CLI `--output-dir` identify the exact benchmark output directory.
