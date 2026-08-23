# Managing the engine's TP model cache

The veloGB10 engine copies required files — model weights, model shards, and the config files that
belong to those models — to node machines **automatically**. These files are saved as **blobs** in the
engine's content-addressed cache.

After you have used several different models with the engine, the cache directory will be full of
model shards. It is a good idea to inspect what is there and to remove shards you will no longer use.

> **If you accidentally delete something you did not mean to delete, do not try to fix it manually.**
> Just run the node and the head again; any missing files will be copied over to the right locations
> automatically.

`gb10_inference` exposes three command-line arguments for managing these cached files:

| Argument | What it does |
|---|---|
| `--cached-models-list` | List cached TP models (name, total size, blob count) |
| `--cached-models-remove <ID>` | Remove **one** cached model (name / unique prefix) |
| `--cached-models-remove-all` | Clear the whole TP model cache |

## List the cache

```bash
./gb10_inference --cached-models-list
```

This may return something like:

```
cached models (2):
  3.8-27b-nvfp4-full-all     15.20 GiB  13 blob(s)
  Qwen3.8-27B-DFlash2      3.58 GiB  6 blob(s)
cache ~/.cache/gb10_tp/blobs — 18 blob(s), 18.78 GiB total
```

## Remove one cached model

To delete one of those entries, reference it by name (or a unique prefix):

```bash
./gb10_inference --cached-models-remove Qwen3.8-27B-DFlash2
```

This removes only `Qwen3.8-27B-DFlash2` from the cache.

## Clear the whole cache

```bash
./gb10_inference --cached-models-remove-all
```

This clears the entire TP model cache.
