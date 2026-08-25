import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
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
                if (msg.type === "live_snapshot")
                    setSnapshot(msg.payload);
                else if (msg.type === "tree_updated")
                    applyTreeUpdate(msg.payload);
            },
            onStateChange: (s) => setWsState(s),
        });
    }, [applyTreeUpdate]);
    useEffect(() => {
        void loadGlobalSettings();
    }, [loadGlobalSettings]);
    return (_jsxs(Switch, { children: [_jsx(Route, { path: "/arch/:dataset/:model/:run", component: ArchViewer }), _jsx(Route, { path: "/", component: Dashboard }), _jsx(Route, { component: Dashboard })] }));
}
