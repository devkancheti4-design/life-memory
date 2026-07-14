#!/usr/bin/env python3
"""Two-session demo against any local ollama model: the assistant that
doesn't forget — and doesn't bluff.

  SESSION 1: facts are stored into the Life (routed by the harness, exactly
             as a tool-running agent following README §1 would run
             `python3 life.py put ...`). The process then DIES.
  SESSION 2: a FRESH process. The bare model is asked each question — it
             cannot know (nothing in context). Then the Life-fused path:
             exact recall in ~µs, the ONE fact injected, model voices it.
             An unstored question must come back "not stored", not a guess.

Honest framing: the Life does the remembering (deterministic, byte-exact);
the model does the phrasing. The bare-model arm shows what any LLM alone
does across sessions: nothing to recall from.

Needs: ollama running locally. Stdlib only otherwise.
Run:   python3 examples/agent_ollama.py [--model llama3.2:3b]
"""
import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from life import Life

FACTS = {
    "wifi.password": "SunFlower42",
    "server.ip": "192.168.7.203",
    "mom.birthday": "March 11",
    "locker.code": "48-19-06",
    "dentist.appointment": "Friday 3pm",
}
DB = os.path.join(ROOT, "demo_life.json")


def ask(model, prompt):
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        json.dumps({"model": model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0}}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


def session1():
    life = Life()
    for k, v in FACTS.items():
        life.learn(k, v)          # == python3 life.py put "k" "v"
    life.save(DB)
    print(f"SESSION 1: stored {len(FACTS)} facts -> {DB}  sha {life.sha()}")
    print("SESSION 1 process exits. Context window: gone. Model state: gone.\n")


def session2(model):
    life = Life(DB)               # fresh process, memory reloaded from disk
    print(f"SESSION 2 (fresh process): loaded sha {life.sha()}")
    bare_ok = fused_ok = 0
    for k, v in FACTS.items():
        q = f"What is my {k.replace('.', ' ')}? Answer with just the value."
        bare = ask(model, q)
        bare_ok += v.lower() in bare.lower()
        fact = life.recall(k)     # == python3 life.py get "k"  (~µs, exact)
        fused = ask(model, f"Known fact: {k} = {fact}\n{q}")
        fused_ok += v in fused    # verbatim requirement
        print(f"  {k:22s} bare: {bare[:34]!r:38s} fused: {fused[:34]!r}")
    # the honesty arm: an unstored fact must ABSTAIN, not be guessed
    missing = life.recall("car.plate")
    print(f"  {'car.plate':22s} life: {'ABSTAIN' if missing is None else missing}"
          "  -> model must answer 'not stored', it has nothing to guess from")
    n = len(FACTS)
    print(f"\nRESULT  bare model: {bare_ok}/{n}   Life-fused: {fused_ok}/{n}")
    print("(the Life recalled deterministically; the model only voiced it)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--session", choices=["1", "2", "both"], default="both")
    a = ap.parse_args()
    if a.session in ("1", "both"):
        if a.session == "both":
            # true process death: session 1 runs in its own python process
            subprocess.run([sys.executable, __file__, "--session", "1"],
                           check=True)
        else:
            session1()
    if a.session in ("2", "both"):
        session2(a.model)
        os.path.exists(DB) and os.unlink(DB)
