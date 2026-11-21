import os
dirname = os.path.dirname(__file__)
# Patched: the data root comes from CL_DATA_DIR when set, so that datasets live
# outside the source tree (an rsync push with --delete would erase them there).
DEFAULT_DATASET_DIR = os.environ.get('CL_DATA_DIR', os.path.join(dirname, '..', 'data'))
