import { useEffect } from "react";
import { Route, Switch } from "wouter";
import { useLiveStore } from "./state/liveStore";
import { useUIStore } from "./state/uiStore";
import { connectLiveWebSocket } from "./api/ws";
import { Dashboard } from "./dashboard/Dashboard";
import { ArchViewer } from "./arch/ArchViewer";

export function App() {
  const loadGlobalSettings = useUIStore((s) => s.loadGlobalSettings);
  const applyTreeUpdate = useUIStore((s) => s.applyTreeUpdate);

  useEffect(() => {
    const setSnapshot = useLiveStore.getState().setSnapshot;
    const setWsState = useLiveStore.getState().setWsState;
    return connectLiveWebSocket({
      onMessage: (msg) => {
        if (msg.type === "live_snapshot") setSnapshot(msg.payload);
        else if (msg.type === "tree_updated") applyTreeUpdate(msg.payload);
      },
      onStateChange: (s) => setWsState(s),
    });
  }, [applyTreeUpdate]);

  useEffect(() => {
    void loadGlobalSettings();
  }, [loadGlobalSettings]);

  return (
    <Switch>
      <Route path="/arch/:dataset/:model/:run" component={ArchViewer} />
      <Route path="/" component={Dashboard} />
      <Route component={Dashboard} />
    </Switch>
  );
}
