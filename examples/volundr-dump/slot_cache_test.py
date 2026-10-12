#!/usr/bin/env python3
# Multi-turn prompt cache check for a volundr model in llama-server.
#
# Turn 1 is a long user message (a document of ~--hist tokens plus a question), turn 2 appends the assistant reply and a
# follow-up question. Turn 2 is sent twice with greedy sampling and top-k logprobs: warm (cache_prompt, reusing the
# slot state of turn 1) and cold (cache_prompt=false, full prefill). The script reports how many prompt tokens each run
# processed, the prompt time, the length of the identical greedy prefix and the max abs logprob difference over it.
#
# usage: slot_cache_test.py <port> <hist_tokens> [--n-predict N] [--think] [--docs DIR]
# start llama-server with -np 1 and -lv 4 to see the checkpoint log lines

import argparse
import glob
import json
import os
import sys
import time
import urllib.request


def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def make_doc(port, docs_dir, n_tok):
    text = ""
    for p in sorted(glob.glob(os.path.join(docs_dir, "**", "*.md"), recursive=True)):
        with open(p, encoding="utf-8", errors="ignore") as f:
            text += f.read() + "\n\n"
        if len(text) > 6*n_tok:
            break
    n_chars = 4*n_tok
    for _ in range(4):
        n = len(post(port, "/tokenize", {"content": text[:n_chars]})["tokens"])
        n_chars = int(n_chars*n_tok/max(n, 1))
    return text[:n_chars]


def chat(port, messages, n_predict, think, cache):
    body = {
        "messages": messages,
        "max_tokens": n_predict,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 10,
        "cache_prompt": cache,
        "id_slot": 0,
        "chat_template_kwargs": {"reasoning_effort": "low"} if think else {"enable_thinking": False},
    }
    t0 = time.time()
    r = post(port, "/v1/chat/completions", body)
    r["wall_s"] = time.time() - t0
    return r


def complete(port, prompt, n_predict, cache):
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "n_probs": 10, "cache_prompt": cache, "id_slot": 0}
    t0 = time.time()
    r = post(port, "/completion", body)
    r["wall_s"] = time.time() - t0
    r["usage"] = {"prompt_tokens": r["tokens_evaluated"]}
    return r


def toks(r):
    if "completion_probabilities" in r:
        return [(c["token"], c["logprob"], {t["token"]: t["logprob"] for t in c["top_logprobs"]}) for c in r["completion_probabilities"]]
    return [(c["token"], c["logprob"], {t["token"]: t["logprob"] for t in c["top_logprobs"]}) for c in r["choices"][0]["logprobs"]["content"]]


def compare(a, b):
    ta, tb = toks(a), toks(b)
    n = 0
    dtok = 0.0  # chosen token
    dtop = 0.0  # any token in both top-10 lists
    for (sa, la, ka), (sb, lb, kb) in zip(ta, tb):
        if sa != sb:
            break
        n += 1
        dtok = max(dtok, abs(la - lb))
        for t in set(ka) & set(kb):
            dtop = max(dtop, abs(ka[t] - kb[t]))
    return n, min(len(ta), len(tb)), dtok, dtop


def report_divergence(a, b, n, nmin):
    if n >= nmin:
        return
    ta, tb = toks(a), toks(b)
    print("a:", repr("".join(t[0] for t in ta[max(0, n-5):n+10])))
    print("b:", repr("".join(t[0] for t in tb[max(0, n-5):n+10])))
    for name, t in (("a", ta), ("b", tb)):
        top = sorted(t[n][2].items(), key=lambda kv: -kv[1])[:3]
        print(f"{name} top-3 at token {n}:", ", ".join(f"{k!r} {v:.4f}" for k, v in top))


def tim(r):
    t = r["timings"]
    return f"prompt_n={t['prompt_n']:6d} cache_n={t.get('cache_n', -1):6d} prompt_ms={t['prompt_ms']:9.1f} predicted_n={t['predicted_n']:4d} wall={r['wall_s']:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("hist", type=int)
    ap.add_argument("--n-predict", type=int, default=300)
    ap.add_argument("--n-turn1", type=int, default=200)
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--raw", action="store_true", help="plain text through /completion instead of chat: turn 2 is a longer prefix of the same text, so the restored checkpoint ends inside ordinary text")
    ap.add_argument("--interleave", action="store_true", help="run another conversation between turn 1 and turn 2 (with -np 1 this moves turn 1 to the host prompt cache)")
    ap.add_argument("--save", help="directory to write the turn-2 messages and the warm/cold responses to")
    ap.add_argument("--replay", help="directory written by --save: only run a cold turn 2 and compare it with the saved cold run (noise floor across server settings)")
    ap.add_argument("--docs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs"))
    args = ap.parse_args()

    if args.replay:
        with open(os.path.join(args.replay, "run.json")) as f:
            run = json.load(f)
        cold = chat(args.port, run["messages"], args.n_predict, args.think, False)
        print(f"turn 2 cold : {tim(cold)}")
        n, nmin, d, dt = compare(run["cold"], cold)
        print(f"RESULT replay cold vs saved cold: identical greedy tokens {n}/{nmin}, max abs logprob diff: chosen {d:.4f}, top-10 {dt:.4f}")
        report_divergence(run["cold"], cold, n, nmin)
        return 0

    doc = make_doc(args.port, args.docs, args.hist)

    if args.raw:
        doc2 = make_doc(args.port, args.docs, args.hist + 300)
        r1 = complete(args.port, doc, 1, True)
        print(f"turn 1      : {tim(r1)}")
        warm  = complete(args.port, doc2, args.n_predict, True)
        regen = complete(args.port, doc2, args.n_predict, True)
        cold  = complete(args.port, doc2, args.n_predict, False)
        cold2 = complete(args.port, doc2, args.n_predict, False)
        return report(args, warm, regen, cold, cold2, None)

    m1 = [{"role": "user", "content": "Here is a set of documents:\n\n" + doc + "\n\nSummarise what these documents are about in a few sentences."}]

    r1 = chat(args.port, m1, args.n_turn1, args.think, True)
    msg1 = r1["choices"][0]["message"]
    print(f"turn 1      : {tim(r1)}")

    m2 = m1 + [{"role": "assistant", "content": msg1.get("content") or ""}, {"role": "user", "content": "Now write a detailed, step by step explanation of how to build the project from source, citing the documents."}]

    if args.interleave:
        h = len(doc)//2
        mo = [{"role": "user", "content": "Here are some notes:\n\n" + doc[h:] + doc[:h] + "\n\nWhat is the main topic?"}]
        ro = chat(args.port, mo, 32, args.think, True)
        print(f"other conv  : {tim(ro)}")

    warm = chat(args.port, m2, args.n_predict, args.think, True)
    # same request again: restores the checkpoint taken 4 tokens before the end of the warm prefill and reprocesses
    # those tokens with the same split, so it must reproduce the warm run exactly
    regen = chat(args.port, m2, args.n_predict, args.think, True)
    cold = chat(args.port, m2, args.n_predict, args.think, False)
    cold2 = chat(args.port, m2, args.n_predict, args.think, False)

    return report(args, warm, regen, cold, cold2, m2)


def report(args, warm, regen, cold, cold2, m2):
    print(f"turn 2 warm : {tim(warm)}")
    print(f"turn 2 regen: {tim(regen)}")
    print(f"turn 2 cold : {tim(cold)}")
    print(f"turn 2 cold2: {tim(cold2)}")

    nr, nminr, dr, dtr = compare(regen, warm)
    print(f"regen vs warm: identical greedy tokens {nr}/{nminr}, max abs logprob diff over the matched prefix: chosen {dr:.4f}, top-10 {dtr:.4f}")

    n, nmin, d, dt = compare(warm, cold)
    n2, nmin2, d2, dt2 = compare(cold, cold2)
    print(f"warm vs cold : identical greedy tokens {n}/{nmin}, max abs logprob diff over the matched prefix: chosen {d:.4f}, top-10 {dt:.4f}")
    print(f"cold vs cold2: identical greedy tokens {n2}/{nmin2}, max abs logprob diff over the matched prefix: chosen {d2:.4f}, top-10 {dt2:.4f}")
    pw, pc = warm["timings"]["prompt_ms"], cold["timings"]["prompt_ms"]
    print(f"RESULT hist={args.hist} think={int(args.think)} prompt_tokens={warm['usage']['prompt_tokens']} warm_prompt_n={warm['timings']['prompt_n']} "
          f"ttft_warm_ms={pw:.0f} ttft_cold_ms={pc:.0f} speedup={pc/max(pw, 1e-3):.1f}x match={n}/{nmin} max_dlogprob_chosen={d:.4f} max_dlogprob_top10={dt:.4f} regen_prompt_n={regen['timings']['prompt_n']} regen_match={nr}/{nminr} regen_max_dlogprob_top10={dtr:.4f}")
    report_divergence(warm, cold, n, nmin)
    if args.save and m2:
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, "run.json"), "w") as f:
            json.dump({"messages": m2, "warm": warm, "cold": cold}, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
