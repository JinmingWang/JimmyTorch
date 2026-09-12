# Experiment Manager contracts

Read this reference for the monitor server, training client, SQLite storage, or dashboard frontend.

## Process boundary

Run the standalone service with:

```bash
python -m Training.ExperimentManager [--host HOST] [--port PORT] [--runs-root PATH]
```

Defaults are loopback host `127.0.0.1`, starting port `9000`, and `./Runs`. The server finds a free port at or above the requested one. Clients do not discover that fallback; configure `experiment_manager_server_port` to the port printed by the service.

`ExperimentManagerClient` never hosts HTTP. Its daemon reporter posts the latest snapshot and polls for a pending learning-rate request. Network errors update reachability state but do not propagate into training.

## Persistence

`ExperimentLogger` owns `<run_dir>/status_and_log.sqlite` and records metadata, sampled scalar series, PNG figure blobs, and lifecycle status. `comments.txt` and `model_arch.txt` remain readable companion files and legacy fallbacks.

`Runs/Experiment_GUI_Status.sqlite` stores dashboard-wide JSON settings such as theme, selection/curve configuration, and notes. Run comments, stars, and colors are run metadata. Storage applies caps to scalar and figure history.

## Live and historical state

- `LiveState` represents only the last registered live run.
- Historical run summaries, scalars, figures, and architecture are read from run directories/databases.
- The watcher broadcasts run-tree changes.
- The browser uses REST for data/mutations and WebSocket for live/tree updates.
- Optional `psutil` and NVML support add system fields; missing metrics are valid.

## Mutations and safety

The dashboard can request a learning rate, edit run comments, set stars/colors, open a folder, and delete a run through a confirmation-token flow. Treat learning-rate requests as training mutations and deletion as destructive user-data removal. Do not exercise either in diagnostic or screenshot work without explicit authorization.

The default loopback binding is the safe operating assumption. A non-loopback binding exposes operational endpoints and must be protected outside the application.

## Validation strategy

- Use a temporary `Runs` root for server/storage tests.
- Verify training still completes with the server absent.
- Test logger data separately from live snapshots.
- Check REST response shapes against frontend types and state stores.
- For screenshots or demos, create sanitized temporary run metadata rather than exposing real project names or notes.
