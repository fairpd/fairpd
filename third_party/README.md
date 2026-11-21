# Third-party code

| directory | upstream | version | licence |
|---|---|---|---|
| `FSW/` | https://github.com/jyparkkr/FSW | commit `6b19d10d1ac0ccc33c3d4ab16d4d070ac9de5edf` | none stated: not redistributed; `scripts/get_fsw.sh` fetches it and applies `patches/fsw-sensitive-indentation.patch` |
| `cl-gym/` | https://pypi.org/project/cl-gym/ (1.0.3, the version FSW installs); licence from https://github.com/imirzadeh/CL-Gym | 1.0.3 | MIT (`cl-gym/LICENSE.txt`) |

CL-Gym is vendored because its pinned dependencies (torch 1.8.1, numpy 1.19)
cannot be installed with current Python; its two local changes are listed in
`cl-gym/PATCHES.md`.
