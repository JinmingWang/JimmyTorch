export function connectLiveWebSocket(opts) {
    let ws = null;
    let closed = false;
    let retryDelay = 500;
    const url = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;
    const open = () => {
        if (closed)
            return;
        opts.onStateChange("connecting");
        ws = new WebSocket(url);
        ws.onopen = () => {
            retryDelay = 500;
            opts.onStateChange("open");
        };
        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                opts.onMessage(msg);
            }
            catch {
                // ignore malformed frames
            }
        };
        ws.onclose = () => {
            opts.onStateChange("closed");
            if (closed)
                return;
            window.setTimeout(open, retryDelay);
            retryDelay = Math.min(retryDelay * 2, 8000);
        };
        ws.onerror = () => {
            // onclose will fire next.
        };
    };
    open();
    return () => {
        closed = true;
        ws?.close();
    };
}
export async function postLearningRate(lr) {
    await fetch("/api/learning-rate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ learning_rate: lr }),
    });
}
