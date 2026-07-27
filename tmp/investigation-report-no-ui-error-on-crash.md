# Investigation Report: No UI Error on Backend Crash

## Summary

The frontend's status circle and error state are driven **exclusively by WebSocket messages**. When the backend crashes, no message is sent, so the UI stays at its last-known state — it never transitions to an error indicator.

## Evidence

### 1. The "red circle" only reflects `trainingStatus.status`

In `frontend/src/App.jsx` (lines 181–199), the circle color is derived from `trainingStatus.status`:

| Status value            | Circle color |
|------------------------|-------------|
| `idle`                 | gray         |
| `ready_to_promote`     | blue         |
| `deploying`            | yellow       |
| `deployment_success`   | green        |
| `deployment_error`     | red          |

This status is set **only** by `ws.onmessage` — when the backend sends a JSON update via the WebSocket. The circle has no connection-health semantics; it is purely a deployment-state indicator.

### 2. WebSocket `onerror` and `onclose` do not update any React state

`frontend/src/App.jsx` lines 22–44:

```jsx
ws.onerror = (err) => {
  console.error("WebSocket error:", err);  // console only — no UI state change
};
ws.onclose = () => {
  console.log("WebSocket connection closed"); // console only — no UI state change
};
```

When the backend crashes, the WebSocket drops. The browser fires `onerror` then `onclose`. Both handlers only log to the browser console. **No React state variable is updated**, so `trainingStatus` remains at whatever value the last WebSocket message set it to (e.g., `{ status: 'idle', ... }`). The circle stays gray.

### 3. No reconnection logic

There is no reconnection attempt in the WebSocket setup. Once the connection drops, it's gone. The user must reload the page (which re-creates the WebSocket) to see updated status.

### 4. The `/analyze` endpoint *does* show errors — but only as text

For text analysis specifically, `handleAnalyze` in `App.jsx` (lines 92–103) does catch network errors from `fetch`:

```jsx
try {
  const data = await analyzeText(text);
  setResults(data);
} catch (err) {
  setError(err.message);  // ← shows red text below the input
}
```

This renders as a red `<div className="error-message">` — **plain text, not the circle indicator**. If the backend crashes while the user is actively analyzing text, they would see a red error message like "Failed to fetch". But the **status circle** in the expert panel would not change.

### 5. Backend crash leaves stale connections in `ConnectionManager`

In `backend/app/main.py`, `ConnectionManager.broadcast()` silently swallows all send errors (`except: pass`). When the backend crashes, dead WebSocket connections remain in `active_connections`. On restart, old connections are not cleaned up, and the frontend's WebSocket object is stale (it won't automatically reconnect).

### 6. Other silently-ignored failures

| Scenario | What the user sees |
|----------|-------------------|
| Model promote returns HTTP error (non-2xx) | Nothing — no `else` branch after `if (response.ok)` |
| WebSocket receives malformed JSON | Uncaught `JSON.parse` error — no user feedback |
| Training subprocess crashes | Status stays "training" forever — no timeout or watchdog |

## Root Cause

**The UI has no mechanism to detect backend unreachability.** The status circle is a deployment-state indicator, not a health indicator. When the backend crashes:

1. The WebSocket drops → `onclose` fires → logs to console only → `trainingStatus` stays at last value → circle color unchanged.
2. No reconnection logic exists → the WebSocket is dead until page reload.
3. No health-check or heartbeat mechanism exists → the frontend never learns that the backend is gone.

The result: the user sees a **stale, gray circle** (or whatever color it was before the crash) and has no visible indication that anything is wrong.

## Ruled Out

| Hypothesis | Why eliminated |
|-----------|---------------|
| "The `/analyze` catch block isn't working" | It works correctly — but only for the analysis flow, not the WebSocket/expert-mode circle. The user was likely looking at the circle, not running analysis. |
| "Backend sends an error WebSocket message before crashing" | No such message exists. The backend has no graceful shutdown hook or crash-notification mechanism. |
| "The circle should turn red on WebSocket close" | There is no code path that sets `trainingStatus.status` to any error value on `onclose`/`onerror`. The red color is only triggered by `status === 'deployment_error'`, which comes from a WebSocket message. |
| "There's a connection-status component I missed" | Thorough search of App.jsx and all components confirms: no connection-status component, no heartbeat, no reconnection logic anywhere. |

## Recommended Fix

1. **Add a `connected` state variable** in `App.jsx`, initialized to `false` and set to `true` on `ws.onopen`, `false` on `ws.onclose`/`ws.onerror`. Use it to change the circle color (e.g., show a disconnected indicator or overlay) when the backend is unreachable.

2. **Add WebSocket reconnection logic** — on `onclose`, attempt to reconnect after a short delay (e.g., 3–5 seconds) with exponential backoff. Reset `connected` on successful reconnection.

3. **Show connection status visually** — either repurpose the existing circle (add a new status like `disconnected`), or add a banner/toast that appears when `connected === false`.

4. **Add a React Error Boundary** wrapping the app to catch unhandled runtime errors (like the `JSON.parse` in `onmessage`) and display a fallback UI instead of a white screen.

5. **Add a `try/catch` around `JSON.parse` in `ws.onmessage`** to prevent runtime crashes from malformed messages, and surface the error to the user.

6. (Backend) **Add a `finally` block to the WebSocket endpoint** that always calls `manager.disconnect(websocket)`, even on unexpected exceptions, to prevent connection leaks.