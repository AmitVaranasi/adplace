#!/bin/zsh
# Run scripts/ingest.py on held-out clips (two at a time) and summarise: python summary at the end.
#   eval/run_testset.sh m43248 m42482 ...
cd "${0:A:h}/.."
eval "$(grep '^export MODEL_API_KEY=' ~/.zshrc)"; export MODEL_API_KEY
run() { .venv/bin/python scripts/ingest.py assets/clips/raw3/$1.mp4 --run-id r_eval_$1 > runs/eval_$1.log 2>&1 }
clips=("$@")
for ((i = 1; i <= ${#clips}; i += 2)); do
  run ${clips[i]} &
  [[ -n ${clips[i+1]} ]] && run ${clips[i+1]} &
  wait
done
.venv/bin/python - "$@" <<'PY'
import json, sys, pathlib
for c in sys.argv[1:]:
    d = pathlib.Path(f"runs/r_eval_{c}")
    r = d / "report.json"
    if not r.exists():
        last = (d / "progress.jsonl").read_text().splitlines()[-1] if (d / "progress.jsonl").exists() else "{}"
        print(f"{c:8s} NO REPORT  {json.loads(last).get('msg', '')[:120]}"); continue
    r = json.loads(r.read_text())
    print(f"{c:8s} {r['summary']}  {r['elapsed_s']} s  surfaces={r['scene'].get('surfaces')} safe={r['scene'].get('scene_safe')}")
    for p in r["placements"]:
        print(f"           {p['id']} {p['surface']:7s} {p['decision']:8s} {'; '.join(p['reasons'])[:150]}")
PY
