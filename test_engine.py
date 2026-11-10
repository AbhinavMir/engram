"""Checks: run with the experiment venv from this folder. Uses the 0.5B model and a tiny random GPT-2."""
import sys, json, os, tempfile, torch
from engine import Engine, BenchError
M05 = "../experiment/models/Qwen2.5-0.5B-Instruct"; M15 = "../experiment/models/Qwen2.5-1.5B-Instruct"
ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg) or c
res = []
e = Engine.load(M05)
p = "The capital of France is"
t = e.trace(p, False)
ids = e.run["ids"]; logits = e.model(ids).logits[0]; last = e.run["base"][-1].to(e.device)
res.append(ok(float((e.lens_logits(last) - logits).abs().max()) < 1e-3, "last-layer lens equals model logits"))
m = next(x for x in e.mats.values() if x.kind.endswith("down_proj") and x.layer == e.n_layers - 4)
orig = e._W(m).clone(); pos = t["n_prompt"] - 1
tgt = t["base"][-1][pos][0][2]; foil = t["base"][-1][pos][1][2]
e.apply(dict(op="rank_one", module=m.name, alpha=4.0, target=dict(id=tgt), foil=dict(id=foil), key="position", pos=pos))
d = e._W(m) - orig; s = torch.linalg.svdvals(d[:64, :64])
res.append(ok(abs(float(d.double().norm()) - 4.0) < 1e-3, "rank-one write has Frobenius norm alpha"))
res.append(ok(float(s[1] / s[0]) < 1e-4, "delta is rank one"))
tk = e.track(dict(id=tgt), dict(id=foil), pos); e.trace(p, False); tk = e.track(dict(id=tgt), dict(id=foil), pos)
res.append(ok(tk["edited"] is not None and tk["edited"][-1] != tk["base"][-1], "edit changes the final margin"))
e.pin("A", pos); e.pin("B", 0)
e.apply(dict(op="scale", module=m.name, factor=0, rows="", cols="0:16"))
res.append(ok(float(e._W(m)[:, :16].abs().max()) == 0, "scale 0 clears the selected columns"))
e.apply(dict(op="remove_direction", module=m.name, strength=1.0, source="pins"))
out = e.export("test_roundtrip"); total = len(e.edits)
sd = {k: v.clone() for k, v in e.model.state_dict().items() if k == m.param}
for x in list(e.edits): e.remove(x["id"])
res.append(ok(e.residual == 0.0, f"rollback restores exact values (residual {e.residual})"))
e.import_delta(out["path"])
res.append(ok(float((e.params[m.param].detach().cpu() - sd[m.param].cpu()).abs().max()) < 1e-5, "export then import reproduces the edited weights"))
os.remove(out["path"]); os.remove(out["path"][:-12] + ".json")
e.remove(e.edits[0]["id"])
try: e.apply(dict(op="rank_one", module=next(x.name for x in e.mats.values() if x.kind.endswith("up_proj")), alpha=1, target=dict(id=1), foil=dict(id=2), key="position", pos=0)); res.append(ok(False, "bad target matrix rejected"))
except BenchError as err: res.append(ok(True, "bad target matrix rejected"))

# other architecture: Conv1D, transformer.h, ln_f
from transformers import GPT2Config, GPT2LMHeadModel
g = GPT2LMHeadModel(GPT2Config(vocab_size=len(e.tok), n_embd=32, n_layer=3, n_head=2, n_positions=256)).eval()
eg = Engine(g, e.tok, "tiny-gpt2", "cpu"); tg = eg.trace("hello there", False)
res.append(ok(eg.layers_path == "transformer.h" and tg["n_prompt"] > 0, "GPT-2 layout found and traced"))
mg = next(x for x in eg.mats.values() if x.kind == "attn.c_proj"); og = g.state_dict()[mg.param].clone()
eg.apply(dict(op="rank_one", module=mg.name, alpha=2.0, target=dict(id=5), foil=dict(id=6), key="position", pos=0))
dg = g.state_dict()[mg.param] - og
res.append(ok(mg.flip and abs(float(dg.double().norm()) - 2.0) < 1e-3, "Conv1D orientation handled"))
eg.matrix(mg.name, 0)
del e, eg, g

# paper reproduction
if "--paper" in sys.argv:
    e = Engine.load(M15); sys.path.insert(0, "../experiment"); os.chdir("../experiment")
    from common import chat_texts; from tf_probe import build
    ex = {json.loads(l)["id"]: json.loads(l) for l in open("data/test.jsonl")}["f5_0002"]
    texts, meta = build(e.tok, [ex]); ids = e.tok(texts, return_tensors="pt", add_special_tokens=False).input_ids.to(e.device)
    P, s_, g_ = [x for x in meta[0][1] if x[1] == "sore throat"][0]
    from tf_probe import class_ids; c = class_ids(e.tok)
    def go():
        lg = e.forward(ids)[2][P]; return float(lg[c["absent"]] - lg[c["not mentioned"]])
    b = go(); e.import_delta("edits/selected_c03.safetensors"); a = go()
    res.append(ok(abs(b + 1.57) < 0.02 and abs(a - 2.14) < 0.02, f"paper edit reproduced on note f5_0002: {b:+.2f} to {a:+.2f}"))
    spec = json.load(open("edits/selected_c03.json"))["ops"][0]
    u, _, _ = e._token_direction(dict(id=c["absent"]), dict(id=c["not mentioned"]))
    res.append(ok(float(u @ torch.tensor(spec["u"]).float()) > 0.999, "token direction matches the paper's u"))
print("ALL PASS" if all(res) else "FAILURES", len(res)); sys.exit(0 if all(res) else 1)
