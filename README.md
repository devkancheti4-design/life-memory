# Life — exact, permanent, honest memory for any LLM

One Python file. Zero dependencies. Works with whatever model you already have —
Claude, GPT, Llama, Qwen, anything local or cloud. Your model keeps the reasoning;
the Life keeps the facts: **exact** (verbatim, not paraphrased), **permanent**
(survives the conversation, the context window, and the process), and **honest**
(on a fact it doesn't have, it says `ABSTAIN` — it cannot guess).

## 1. The prompt — this is the product

Get [`life.py`](life.py) into your working directory, then paste this into any
assistant that can run commands (Claude Code, Cursor, Codex, Aider, an agent
wrapper around a local model — anything):

```
You now have an exact external memory: ./life.py (one stdlib Python file, no install).
Follow these rules exactly:

1. STORE — when the user states a fact worth keeping (a name, number, path,
   decision, preference — anything they'd be annoyed to repeat), run:
     python3 life.py put "subject.attribute" "exact value"
   Use lowercase dot-notation keys, e.g. wifi.password, server.ip, mom.birthday.

2. RECALL — before answering anything that depends on a stored fact, run:
     python3 life.py get "subject.attribute"
   • It prints a value  -> use it VERBATIM. Never paraphrase a stored value.
   • It prints ABSTAIN  -> say you don't have that stored. NEVER guess a stored fact.

3. REVISE — store again with the same key; the newest value wins and the old
   ones stay inspectable via: python3 life.py history "key"

4. RELATE — python3 life.py link "a" "b" connects facts;
   python3 life.py chain "a" follows the whole dependency chain.

5. The memory is ./life.json. It outlives this conversation, your context
   window, and your process. Never edit it by hand, never delete it.

If you cannot run commands, print the exact command for the user to run and
ask them to paste the output back. The protocol is the same either way.
```

That's the whole integration. The prompt adapts to whatever model reads it:
a tool-running agent executes the commands itself; a chat-only model falls back
to printing them for you. Either way the model now has memory that never
paraphrases and never bluffs.

## 2. Install & smoke test

```bash
git clone https://github.com/devkancheti4-design/life-memory
cd life-memory
python3 smoke_test.py        # ~10 s, stdlib only, deterministic pass/fail
```

The smoke test proves every claim on this page on your machine: 10,000/10,000
exact recalls, revision with retained history, 1,000/1,000 abstentions on
unseen keys, byte-identical twins (golden SHA `0ffd7ccc8e97f01b` — the same on
your machine as on ours), recall across process death, 3-hop relational chains,
gated fusion math, and 100k-fact scale.

Try it by hand:

```bash
python3 life.py put "wifi.password" "SunFlower42"
python3 life.py get "wifi.password"     # -> SunFlower42, in ~microseconds
python3 life.py get "wifi.pasword"      # -> ABSTAIN (exact-key: typos abstain, not guess)
python3 life.py sha                     # -> identity hash; twins match byte-exactly
```

## 3. Why it's not "just a dict"

Honest answer first: the storage core **is** an exact-key count table — on
purpose. The proof below shows exact lookup is precisely the property gradient
training cannot deliver, so the winning move is to keep it exact, not to make
it clever. What a plain dict does **not** give you:

| Property | dict | Life |
|---|---|---|
| Revision keeps full history (`history`) | ✗ overwrites | ✓ newest wins, history retained |
| Abstention wired into the model contract | `None`, unused | ✓ `ABSTAIN` → the model must say "not stored" instead of guessing |
| **Confidence-gated logit fusion** (`blend`) | ✗ | ✓ counts fold into the model's softmax; unseen keys leave the model untouched |
| Relational multi-hop retrieval (`chain`) | ✗ one value | ✓ follows dependency chains |
| Canonical identity (`sha`) | ✗ | ✓ byte-exact twins, byte-exact across process death |

The fusion row is the one that matters. Measured end-to-end with buried facts:
**bare model 0/40 · facts-in-context 36/40 · fused 40/40.** The fused system
ties the fair baseline and adds an exactness edge — because the table's counts
override the softmax on keys it knows and leave it alone on keys it doesn't.

## 4. Why an LLM needs it — the Required-Fusion Proof

[`proof/PROOF.md`](proof/PROOF.md) argues from measurement that an LLM+exact-store
fusion is not a convenience but the design the mathematics forces. Reproduce it:

```bash
bash proof/run_proof.sh          # pillars I–IV, ~2 min, stdlib only, deterministic
bash proof/run_proof.sh --llm    # pillar V on real local LLMs (needs ollama)
```

The five pillars, one line each (numbers from the deterministic experiments):

1. **Exact recall is optimizer-inaccessible.** The training gradient vanishes
   proportionally to remaining error (measured over 5 orders of magnitude);
   on near-duplicate keys attention leaks 85% of its mass at cosine 0.99.
   The exact table: zero leak, zero training.
2. **No fixed-size differentiable memory substitutes.** Fast-weight recall
   cliffs at K = d writes (100% → 37%), and the damage is retroactive —
   new writes physically degrade the oldest memories (100% → 31%).
3. **Softmax cannot abstain.** A net at its generalizing best answers pure
   garbage with 0.974 mean confidence (36/40 above 90%). The table abstains
   60/60 — "I don't know" is structural, not trained.
4. **Writing facts into weights is conditionally destructive — decidably.**
   Agreement-bearing streams self-heal (75%→92%); conflicting streams collapse
   to 8% retention. The table holds 100% in both regimes. That decidable split
   is the routing law of the fusion.
5. **It's behaviorally real in production models.** llama3.2:3b and llama3:8b
   at temperature 0 drop from 100% to ~88% per-fact recall as queried facts
   grow 4 → 64, and ~80% of the errors are another fact's value, confidently
   delivered. More parameters did not fix it.

The Life holds exactly the complementary corner — and *only* that corner
(see Honest limits).

## 5. Three ways to fuse

**a) Prompt fusion — any model (the prompt in §1).** Zero code. The model
runs `put`/`get` and obeys `ABSTAIN`. Works with cloud APIs and local models
alike; nothing about your facts enters any training set or context you didn't
choose.

**b) Logit fusion — local models.** [`examples/fuse_logits.py`](examples/fuse_logits.py)
shows `blend()` folding the table into a real model's next-token distribution —
the mechanism behind the 40/40. Needs a local model you can read logits from
(transformers, MLX, llama.cpp).

**c) Life + RAG + model — the full memory system.** The Life is exact-key
*only*. Measured: on 20 buried facts, entity-named queries hit 10/10 through
the Life, but paraphrased queries hit **0/10** — while embedding-based RAG got
those 10/10. They're different organs: Life = exact/permanent/honest, RAG =
fuzzy/semantic, model = reasoning. [`examples/agent_ollama.py`](examples/agent_ollama.py)
runs the two-session demo (store facts, kill the process, fresh session
recalls) against any local ollama model.

## 6. Measured numbers

`[EXACT]` = deterministic, reproduces bit-for-bit on your machine.
`[HW]` = measured on an Apple M4 laptop; yours will differ.

| Measurement | Value | Label |
|---|---|---|
| Recall correctness, 10k facts | 10,000/10,000 verbatim | [EXACT] |
| Abstention on unseen keys | 1,000/1,000 (0 guesses) | [EXACT] |
| Twin determinism | SHA `0ffd7ccc8e97f01b` on any machine | [EXACT] |
| Recall across process death | byte-exact (cross-process SHA match) | [EXACT] |
| Fused recall vs fair baseline | bare 0/40 · in-context 36/40 · fused 40/40 | [EXACT protocol, HW model] |
| Recall latency @100k facts | ~1 µs, O(1) flat | [HW] |
| Recall latency @15M facts | 0.2–1.4 µs (measured, still flat) | [HW] |
| Memory cost | ~400 B/fact RAM, ~40 B/fact on disk | [HW] |
| Store scale tested | 30M facts, 0 key collisions | [EXACT] |
| Context-window comparison | 1M facts ≈ 6M tokens — no context window holds it; the Life holds it in ~410 MB RAM | [EXACT arithmetic] |

## 7. Honest limits — read before you rely on it

- **Exact-key only.** `wifi.pasword` (typo) abstains. A paraphrase ("what's
  that plant-named password?") scores **0/10** where semantic RAG scores 10/10.
  Pair with RAG if you need fuzzy retrieval.
- **It does not generalize and does not reason.** 0% held-out on unseen keys —
  by design. The model is the reasoner; the Life is the memory.
- **It ties, not crushes, the fair baseline.** A model *given* the facts
  in-context already gets ~90% (36/40). The Life's win is the last 10%
  exactness, plus permanence past the context window, plus honest abstention —
  not a new kind of intelligence.
- **Growth is O(K).** Exactness is bought with ~400 B/fact RAM. 1M facts ≈
  410 MB. That's the trade — the proof's Pillar II shows every fixed-size
  alternative pays in silent forgetting instead.
- **Determinism means byte-exact replay,** not correctness: store a wrong
  fact and it will recall the wrong fact, exactly, forever (until you revise).
- **No commercial-value claim.** This repo claims exactly what its tests
  measure, nothing more.

## 8. License

MIT. One file. Take it, fuse it.
