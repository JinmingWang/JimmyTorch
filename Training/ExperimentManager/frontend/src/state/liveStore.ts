import { create } from "zustand";
import type { LiveSnapshot } from "../api/types";
import type { WsConnectionState } from "../api/ws";

interface LiveState {
  snapshot: LiveSnapshot | null;
  wsState: WsConnectionState;
  setSnapshot: (s: LiveSnapshot) => void;
  setWsState: (s: WsConnectionState) => void;
}

export const useLiveStore = create<LiveState>((set) => ({
  snapshot: null,
  wsState: "connecting",
  setSnapshot: (s) => set({ snapshot: s }),
  setWsState: (s) => set({ wsState: s }),
}));
