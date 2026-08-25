import { create } from "zustand";
export const useLiveStore = create((set) => ({
    snapshot: null,
    wsState: "connecting",
    setSnapshot: (s) => set({ snapshot: s }),
    setWsState: (s) => set({ wsState: s }),
}));
