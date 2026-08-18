# Native build.

- 1) Build the simulator and run a smoke test:
  - ./scripts/setup.sh
- 2) Download benchmark matrices:

```
python3 scripts/download_matrices.py
```

Docker (alternative). A pre-configured container is also available:

```
docker compose build
docker compose run artifact \
  ./scripts/run_all.sh
```

