# Local patches to cl_gym 1.0.3

1. `cl_gym/benchmarks/utils.py`: `DEFAULT_DATASET_DIR` reads the environment
   variable `CL_DATA_DIR` when it is set (default unchanged), so that datasets
   live outside the source tree.
2. `cl_gym/utils/loggers.py`: the `comet_ml` import is optional; only the Comet
   logger, which this project does not use, needs it.

Nothing else is changed. The package is imported from this directory (the
runner puts `third_party/cl-gym` on `sys.path`); its PyPI pins are not installed.
