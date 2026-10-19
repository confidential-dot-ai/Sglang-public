# Recipes

Launch recipes for configurations brought up on this fork. Each recipe is a
thin wrapper around an existing launcher. It pins the flags for one
configuration so the command does not live only in a report or a shell history.

These are bring-up and qualification configurations, not deployment advice.
The recommended deployment for a model lives on its `docs/cookbook/` page.

| Model | Folder | Launcher it wraps |
| --- | --- | --- |
| DeepSeek-V4.1 | [`deepseek-v41/`](deepseek-v41/README.md) | `benchmark/deepseek_v41_pd/launch.py` |

Every recipe serves the checkout it lives in. Extra arguments and environment
overrides pass through to the underlying launcher; see each folder's README.
