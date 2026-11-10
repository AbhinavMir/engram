"""Local server for Engram. Binds to 127.0.0.1 only.
Usage: python server.py [--model DIR] [--device mps|cuda|cpu] [--port 7860]"""
import argparse, atexit, json, os, shutil, socket, subprocess, threading, traceback, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from engine import Engine, BenchError, HERE

LOCK = threading.Lock(); E = {"engine": None, "last": None}
def need():
    if E["engine"] is None: raise BenchError("Load a model first.")
    return E["engine"]

def retrace(e):
    return e.trace(e.run["prompt"], e.run["chat"]) if e.run else dict(info=e.info())

def route(path, b):
    if path == "/api/load":
        E["engine"] = None; E["engine"] = Engine.load(b["path"], ARGS.device); return dict(info=E["engine"].info())
    e = need(); pos = b.get("pos")
    if path == "/api/trace": return e.trace(b["prompt"], bool(b.get("chat")))
    if path == "/api/track": return e.track(b.get("target"), b.get("foil"), int(pos))
    if path == "/api/matrix": return e.matrix(b["name"], int(pos or 0))
    if path == "/api/pin": e.pin(b["group"], int(pos)); return dict(info=e.info())
    if path == "/api/unpin": e.unpin(int(b["id"])); return dict(info=e.info())
    if path == "/api/edit": e.apply(b); return retrace(e)
    if path == "/api/toggle": e.toggle(int(b["id"]), b["on"]); return retrace(e)
    if path == "/api/remove": e.remove(int(b["id"])); return retrace(e)
    if path == "/api/export": return e.export(b.get("name"))
    if path == "/api/import": e.import_delta(b["path"]); return retrace(e)
    raise BenchError(f"Unknown route {path}")

AI = {"port": None}
def start_ai():
    folder = os.path.join(HERE, "ai"); bun = shutil.which("bun") or os.path.expanduser("~/.bun/bin/bun")
    if not os.path.exists(os.path.join(folder, "node_modules")) or not os.path.exists(bun): return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); AI["port"] = sock.getsockname()[1]
    proc = subprocess.Popen([bun, "sidecar.mjs"], cwd=folder, env={**os.environ, "PORT": str(AI["port"])})
    atexit.register(proc.terminate)

def ai_call(path, data=None):
    if AI["port"] is None: raise BenchError(f"The AI process is not running. Run bun install in {os.path.join(HERE, 'ai')} and restart the server.")
    req = urllib.request.Request(f"http://127.0.0.1:{AI['port']}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=240) as r: return r.status, r.read()
    except urllib.error.HTTPError as err: return err.code, err.read()

class H(BaseHTTPRequestHandler):
    def allowed(self):
        hosts = (f"127.0.0.1:{ARGS.port}", f"localhost:{ARGS.port}"); origin = self.headers.get("Origin")
        return self.headers.get("Host") in hosts and (origin is None or origin in tuple("http://" + h for h in hosts))
    def log_message(self, *a): pass
    def send(self, code, data, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if not self.allowed(): return self.send(403, b"{}")
        if self.path.startswith("/api/ai/"):
            try: code, out = ai_call(self.path[7:])
            except BenchError as err: code, out = 400, json.dumps(dict(error=str(err))).encode()
            return self.send(code, out)
        if self.path == "/":
            return self.send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if self.path == "/api/state":
            e = E["engine"]; return self.send(200, json.dumps(dict(info=e.info() if e else None, suggest=ARGS.model and os.path.abspath(ARGS.model))).encode())
        self.send(404, b"{}")
    def do_POST(self):
        if not self.allowed(): return self.send(403, json.dumps(dict(error="This request came from another site.")).encode())
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            if self.path.startswith("/api/ai/"):
                code, out = ai_call(self.path[7:], raw); return self.send(code, out)
            body = json.loads(raw)
            with LOCK: out = route(self.path, body)
            self.send(200, json.dumps(out).encode())
        except BenchError as err: self.send(400, json.dumps(dict(error=str(err))).encode())
        except Exception as err:
            traceback.print_exc(); self.send(500, json.dumps(dict(error=f"{type(err).__name__}: {err}")).encode())

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model"); p.add_argument("--device"); p.add_argument("--port", type=int, default=7860); ARGS = p.parse_args()
    if ARGS.model: E["engine"] = Engine.load(ARGS.model, ARGS.device); print("loaded", E["engine"].name, "on", E["engine"].device, flush=True)
    start_ai()
    print(f"Engram: http://127.0.0.1:{ARGS.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()
