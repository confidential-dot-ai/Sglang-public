# Recipes

Launch recipes for configurations served from this branch. Each recipe is a
thin wrapper around an existing launcher. It pins the flags for one
configuration so the command does not live only in a report or a shell history.

`deepseek-v41/container/` holds the production deployment: the image, compose
file and systemd units for one PD pair.

| Model | Folder | Launcher it wraps |
| --- | --- | --- |
| DeepSeek-V4.1 | [`deepseek-v41/`](deepseek-v41/README.md) | `recipes/deepseek-v41/launcher/launch.py` |

Every recipe serves the checkout it lives in. Extra arguments and environment
overrides pass through to the underlying launcher; see each folder's README.
