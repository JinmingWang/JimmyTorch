import asyncio
from types import SimpleNamespace

import pytest
from Training.ExperimentManager import client as client_module
from Distributed.Context import DistributedContext


class FakeNvml:
    def __init__(self, count=3, fail_index=None):
        self.count = count
        self.fail_index = fail_index

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetUUID(self, handle):
        return f"GPU-id-{handle}".encode()

    def nvmlDeviceGetName(self, handle):
        return f"Test GPU {handle}".encode()

    def nvmlDeviceGetMemoryInfo(self, handle):
        if handle == self.fail_index:
            raise RuntimeError("unavailable")
        return SimpleNamespace(used=100 + handle, total=1000)

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=handle * 20)


def test_gpuIdentityReorderingAndLegacyFields(monkeypatch):
    monkeypatch.setattr(client_module, "_pynvml", FakeNvml())
    assignments = [{"rank": 0, "device_id": "GPU-id-2", "device_type": "cuda"},
                   {"rank": 1, "device_id": "GPU-id-0", "device_type": "cuda"}]
    stats = client_module._sampleSystem(assignments)
    assert [gpu["training_ranks"] for gpu in stats["gpus"]] == [[1], [], [0]]
    assert [gpu["index"] for gpu in stats["gpus"]] == [0, 1, 2]
    assert stats["gpu_mem_used"] == 100
    assert stats["gpu_util"] == 0
    assert stats["gpus"][2]["name"] == "Test GPU 2"


@pytest.mark.parametrize("count", [0, 1, 8])
def test_gpuCounts(monkeypatch, count):
    monkeypatch.setattr(client_module, "_pynvml", FakeNvml(count))
    assert len(client_module._sampleSystem([])["gpus"]) == count


def test_partialFailureAndUnknownAssignments(monkeypatch):
    monkeypatch.setattr(client_module, "_pynvml", FakeNvml(fail_index=1))
    stats = client_module._sampleSystem([{"rank": 0, "device_id": None, "device_type": "cuda"}])
    assert stats["gpus"][1]["memory_used"] is None
    assert stats["gpus"][1]["utilization"] == 20
    assert stats["gpus"][2]["memory_used"] == 102
    assert all(gpu["training_ranks"] is None for gpu in stats["gpus"])
    monkeypatch.setattr(client_module, "_pynvml", None)
    assert client_module._sampleSystem()["gpus"] is None


def test_deviceAssignmentsUseCudaUuid(monkeypatch):
    import torch
    context = DistributedContext.__new__(DistributedContext)
    context.device = torch.device("cuda", 1)
    context.rank, context.world_size, context.is_distributed = 1, 1, False
    context.node_id, context.hostname, context.local_rank, context.local_world_size = "0", "test", 0, 1
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-physical-7"))
    assert context.deviceAssignments()[0]["device_id"] == "GPU-physical-7"


def test_deviceSamplingDoesNotHoldTrainingLock(monkeypatch):
    monkeypatch.setattr(client_module.ExperimentManagerClient, "_register_and_start", lambda self: None)
    client = client_module.ExperimentManagerClient(1, 1)

    def sampleWithoutLock(assignments):
        assert not client._lock.locked()
        return {"gpus": []}

    monkeypatch.setattr(client_module, "_sampleSystem", sampleWithoutLock)
    assert client.snapshot()["system"]["gpus"] == []


def test_gpuHttpAndWebsocketPropagation(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from Training.ExperimentManager.server import build_app

    async def exerciseServer():
        async with TestClient(TestServer(build_app(tmp_path))) as client:
            websocket = await client.ws_connect("/ws")
            await websocket.receive_json()
            await websocket.receive_json()
            stats = {"gpus": [{"id": "GPU-test", "index": 3, "name": "Test GPU",
                              "memory_used": 10, "memory_total": 100, "utilization": 25,
                              "training_ranks": [0]}]}
            response = await client.post("/api/ingest/update", json={"system": stats})
            assert response.status == 200
            message = await websocket.receive_json(timeout=5)
            while message["type"] != "live_snapshot":
                message = await websocket.receive_json(timeout=5)
            assert message["payload"]["system"] == stats
            response = await client.get("/api/status")
            assert (await response.json())["system"] == stats
            await websocket.close()
    asyncio.run(exerciseServer())


def test_multiNodeHttpIsolationAndStaleness(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from Training.ExperimentManager.server import build_app

    async def exerciseServer():
        app = build_app(tmp_path)
        async with TestClient(TestServer(app)) as client:
            topology = [{"rank": index, "node_id": str(index), "hostname": f"host-{index}"} for index in range(2)]
            registration = {"run_id": "new-run", "topology": topology, "total_steps": 5}
            assert (await client.post("/api/ingest/register", json=registration)).status == 200
            websocket = await client.ws_connect("/ws")
            await websocket.receive_json()
            await websocket.receive_json()
            for index in range(2):
                response = await client.post("/api/ingest/node", json={"run_id": "new-run", "node_id": str(index),
                    "system": {"gpus": [], "cpu_mem_bytes": index + 100}, "status": "error"})
                assert response.status == 200
            snapshot = (await (await client.get("/api/status")).json())
            assert snapshot["status"] == "idle"  # Nodes cannot change training status.
            assert snapshot["system"]["cpu_mem_bytes"] == 100
            assert [node["system"]["cpu_mem_bytes"] for node in snapshot["nodes"]] == [100, 101]
            assert all(node["connected"] for node in snapshot["nodes"])
            for endpoint in ("node", "update", "finish", "status"):
                response = await client.post(f"/api/ingest/{endpoint}", json={"run_id": "old-run", "node_id": "0"})
                assert response.status == 409
            assert (await client.get("/api/live/learning-rate?run_id=old-run")).status == 409
            assert (await client.post("/api/ingest/node", json={"run_id": "new-run", "node_id": "unknown"})).status == 400
            app["live_state"].snapshot["nodes"][1]["received_at"] -= 30
            snapshot = await app["live_state"].get_snapshot()
            assert snapshot["nodes"][0]["connected"]
            assert not snapshot["nodes"][1]["connected"]
            assert snapshot["nodes"][1]["system"]["cpu_mem_bytes"] == 101
            # Re-registration by the same owner must preserve other machines.
            await client.post("/api/ingest/register", json=registration)
            assert len((await app["live_state"].get_snapshot())["nodes"]) == 2
            received_nodes = False
            for _ in range(10):
                message = await websocket.receive_json(timeout=5)
                if message["type"] == "live_snapshot" and len(message["payload"].get("nodes", [])) == 2:
                    received_nodes = True
                    break
            assert received_nodes
            await client.post("/api/ingest/finish", json={"run_id": "new-run", "status": "done"})
            snapshot = await app["live_state"].get_snapshot()
            assert not snapshot["connected"]
            assert all(not node["connected"] for node in snapshot["nodes"])
            await websocket.close()
    asyncio.run(exerciseServer())


def test_realBackgroundReportersAndReconnect(tmp_path, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer
    from Training.ExperimentManager.server import build_app
    from Distributed.Reporting import DistributedRunClient, NodeReporter

    monkeypatch.setattr(client_module, "_pynvml", FakeNvml(2))

    async def exerciseReporters():
        async with TestClient(TestServer(build_app(tmp_path))) as http_client:
            host, port = http_client.server.host, http_client.server.port
            topology = [{"rank": index, "node_id": str(index), "hostname": f"host-{index}",
                         "device_id": f"GPU-id-{1 - index}", "device_type": "cuda"} for index in range(2)]
            run_client = await asyncio.to_thread(DistributedRunClient, 5, 2, host=host, port=port,
                run_id="reporter-run", topology=topology, refresh_interval=0.03)
            reporters = [NodeReporter(host, port, "reporter-run", str(index), [topology[index]], 0.03)
                         for index in range(2)]
            try:
                run_client.update(0, 0)
                for _ in range(100):
                    snapshot = await http_client.server.app["live_state"].get_snapshot()
                    if snapshot["progress"]["overall"] == 1 and all(node["received_at"] for node in snapshot["nodes"]):
                        break
                    await asyncio.sleep(0.03)
                assert snapshot["status"] == "training"
                assert len(snapshot["nodes"]) == 2
                assert [gpu["training_ranks"] for gpu in snapshot["nodes"][0]["system"]["gpus"]] == [[], [0]]
                assert [gpu["training_ranks"] for gpu in snapshot["nodes"][1]["system"]["gpus"]] == [[1], []]
                # An empty server state simulates a dashboard restart during training.
                from Training.ExperimentManager.server import LiveState
                http_client.server.app["live_state"] = LiveState()
                for _ in range(100):
                    snapshot = await http_client.server.app["live_state"].get_snapshot()
                    if snapshot["run"] and snapshot["progress"]["overall"] == 1 and all(node["received_at"] for node in snapshot["nodes"]):
                        break
                    await asyncio.sleep(0.03)
                assert snapshot["run"]["id"] == "reporter-run"
                assert snapshot["progress"]["overall"] == 1
                assert len(snapshot["nodes"]) == 2
                await http_client.post("/api/learning-rate", json={"learning_rate": 0.025})
                requested = None
                for _ in range(100):
                    requested = run_client.consume_learning_rate_request()
                    if requested is not None:
                        break
                    await asyncio.sleep(0.03)
                assert requested == 0.025
                run_client.set_status("done")
            finally:
                for reporter in reporters:
                    await asyncio.to_thread(reporter.close)
                    assert not reporter.thread.is_alive()
                await asyncio.to_thread(run_client.close)
                assert not run_client._reporter_thread.is_alive()
            snapshot = await http_client.server.app["live_state"].get_snapshot()
            assert snapshot["status"] == "done"
            assert not snapshot["connected"]
    asyncio.run(exerciseReporters())


def test_failedDeviceDoesNotHideOthers(monkeypatch):
    class BrokenNvml(FakeNvml):
        def nvmlDeviceGetHandleByIndex(self, index):
            if index == 1:
                raise RuntimeError("device is unavailable")
            return index
        def nvmlDeviceGetName(self, handle):
            raise RuntimeError("name unavailable")

    monkeypatch.setattr(client_module, "_pynvml", BrokenNvml())
    stats = client_module._sampleSystem([])
    assert len(stats["gpus"]) == 3
    assert stats["gpus"][1]["utilization"] is None
    assert stats["gpus"][0]["memory_used"] == 100
    assert stats["gpus"][2]["memory_used"] == 102
    assert all(gpu["name"] is None for gpu in stats["gpus"])
