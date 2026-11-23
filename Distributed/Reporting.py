"""HTTP-only reporting. Background workers never call distributed collectives."""
import json
import threading
import urllib.request
from Training.ExperimentManager.client import ExperimentManagerClient, _sampleSystem


class DistributedRunClient(ExperimentManagerClient):
    def __init__(self, *args, run_id, topology, **kwargs):
        self.run_id = run_id
        self.topology = topology
        self.registration_payload = None
        self.registered = False
        super().__init__(*args, **kwargs)

    def sampleSystem(self):
        # The local node reporter supplies primary-machine telemetry too.
        return {}

    def _post_silently(self, path, payload):
        if path != "/api/ingest/finish" and self._stop_event.is_set():
            return None
        payload = dict(payload, run_id=self.run_id)
        if path == "/api/ingest/register":
            payload["topology"] = self.topology
            payload["telemetry_interval"] = self.refresh_interval
        if path == "/api/ingest/update":
            payload.pop("system", None)
        if path == "/api/ingest/register":
            self.registration_payload = payload
            response = super()._post_silently(path, payload)
            self.registered = response is not None
            return response
        if path == "/api/ingest/update" and not self.registered:
            self.retryRegistration()
        response = super()._post_silently(path, payload)
        if path == "/api/ingest/update" and response is None:
            # Recover after a server restart, but never reclaim a newer run.
            self.retryRegistration()
        return response

    def retryRegistration(self):
        if self._stop_event.is_set():
            return
        snapshot = super()._get_silently("/api/status")
        if (not self._stop_event.is_set() and snapshot is not None
                and not snapshot.get("run") and self.registration_payload):
            response = super()._post_silently("/api/ingest/register", self.registration_payload)
            self.registered = response is not None
        elif snapshot and (snapshot.get("run") or {}).get("id") == self.run_id:
            self.registered = True


    def _get_silently(self, path):
        if self._stop_event.is_set():
            return None
        return super()._get_silently(path + "?run_id=" + self.run_id)


class NodeReporter:
    def __init__(self, host, port, run_id, node_id, device_assignments, refresh_interval=1.0):
        self.url = f"http://{host}:{port}/api/ingest/node"
        self.run_id = run_id
        self.node_id = node_id
        self.device_assignments = device_assignments
        self.refresh_interval = refresh_interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.reportLoop, name="NodeTelemetry", daemon=True)
        self.thread.start()

    def reportLoop(self):
        while not self.stop_event.is_set():
            try:
                payload = {"run_id": self.run_id, "node_id": self.node_id,
                           "system": _sampleSystem(self.device_assignments)}
                if self.stop_event.is_set():
                    return
                request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=1.5) as response:
                    response.read()
            except Exception:
                pass  # Optional telemetry must never interrupt model training.
            self.stop_event.wait(self.refresh_interval)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
