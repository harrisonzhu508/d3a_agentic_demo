"""Canned events for every hook, in each harness's native payload format (Claude Code, Codex, pi).

    uv run python hooks/tests/run_tests.py          # fast cases
    uv run python hooks/tests/run_tests.py --slow   # also run the post-edit smoke test on src/model.py (~20 s)

Runs in a throwaway experiment (results/_hooktest/, removed afterwards), so real results are never touched.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[2]
HOOKS = DEMO / "hooks"
EXP = "_hooktest"
ENV = {**os.environ, "D3A_EXPERIMENT": EXP, "D3A_MAX_EXPERIMENTS": "1"}

sys.path.insert(0, str(DEMO))
os.environ["D3A_EXPERIMENT"] = EXP
from checks.settings import RESULTS, settings  # noqa: E402

PREFIX = settings()["branch_prefix"]
BASE = settings()["base_branch"]
CURRENT = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=DEMO, capture_output=True,
                         text=True).stdout.strip()
ON_AGENT_BRANCH = CURRENT.startswith(PREFIX + "/")


def claude(tool: str, **ti) -> dict:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": ti, "session_id": "test"}


def bash(cmd: str) -> dict:
    return claude("Bash", command=cmd)


def codex(tool: str, command: str) -> dict:
    return {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": {"command": command},
            "session_id": "test", "turn_id": "test", "cwd": str(DEMO)}


def patch(*lines: str) -> dict:
    return codex("apply_patch", "*** Begin Patch\n" + "\n".join(lines) + "\n*** End Patch")


def pi(tool: str, **inp) -> list[str]:
    return ["--payload", json.dumps({"event": "PreToolUse", "tool": tool, "input": inp})]


R = f"results/{EXP}"
PRE = [
    # reads
    ("claude Read DATASET.md", claude("Read", file_path=str(DEMO.parent / "dataset" / "DATASET.md")), 2),
    ("claude Read ../slides", claude("Read", file_path="../slides/content.yaml"), 2),
    ("claude Read full data table", claude("Read", file_path="../dataset/cip_features.csv"), 2),
    ("claude Grep the repo root", claude("Grep", pattern="83", path=".."), 2),
    ("claude Glob outside demo", claude("Glob", pattern="**/*.md", path="/home"), 2),
    ("claude Grep demo", claude("Grep", pattern="beta", path="src"), 0),
    ("claude Read private test set", claude("Read", file_path="~/.d3a-demo-private/test.csv"), 2),
    ("claude Read data/dev.csv", claude("Read", file_path="data/dev.csv"), 0),
    ("claude Read a skill", claude("Read", file_path="skills/censored-mic-regression/SKILL.md"), 0),
    # writes
    ("claude Edit checks/evaluate.py", claude("Edit", file_path="checks/evaluate.py", old_string="a", new_string="b"), 2),
    ("claude Write TASK.md", claude("Write", file_path="TASK.md", content="x"), 2),
    ("claude Write config/autoresearch.toml", claude("Write", file_path="config/autoresearch.toml", content="x"), 2),
    ("claude Write notes.md (outside src)", claude("Write", file_path="notes.md", content="x"), 2),
    ("claude Edit src/model.py", claude("Edit", file_path="src/model.py", old_string="a", new_string="b"), 0),
    ("claude Write the report draft", claude("Write", file_path=f"{R}/x/report_draft.md", content="x"), 0),
    ("claude Write a final report by hand", claude("Write", file_path=f"{R}/x/report.md", content="x"), 2),
    ("claude Write into reports/", claude("Write", file_path=f"reports/{EXP}/x/report.md", content="x"), 2),
    ("claude Write metrics.json", claude("Write", file_path=f"{R}/x/metrics.json", content="{}"), 2),
    ("claude Write champion.json", claude("Write", file_path=f"{R}/champion.json", content="{}"), 2),
    # shell
    ("bash read full table", bash("head ../dataset/cip_features.csv"), 2),
    ("bash read DATASET.md by name", bash("cat /x/DATASET.md"), 2),
    ("bash look at slides", bash("ls ../slides/figures"), 2),
    ("bash run split.py", bash("uv run python checks/split.py"), 2),
    ("bash evaluate", bash('uv run python checks/evaluate.py --name x --hypothesis "h" --skill s'), 0),
    ("bash status", bash("uv run python checks/status.py"), 0),
    ("bash write into checks/", bash("echo 1 > checks/foo.txt"), 2),
    ("bash sed -i a hook", bash("sed -i 's/a/b/' hooks/stop.py"), 2),
    ("bash overwrite champion.json", bash(f"echo '{{}}' > {R}/champion.json"), 2),
    ("bash overwrite metrics.json", bash(f"cp /tmp/m.json {R}/x/metrics.json"), 2),
    ("bash harmless python", bash("uv run python -c 'print(1)'"), 0),
    ("bash lift the time limit", bash('D3A_BUDGET_SECONDS=0 uv run python checks/evaluate.py --name x'), 2),
    ("claude Read secrets.env", claude("Read", file_path="config/secrets.env"), 2),
    ("bash cat secrets", bash("cat config/secrets.env"), 2),
    ("bash env dump", bash("env | sort"), 2),
    ("bash bare env", bash("cd src && env"), 2),
    ("bash printenv", bash("printenv"), 2),
    ("bash echo a key", bash("echo $OPENAI_API_KEY"), 2),
    ("bash python environ", bash("uv run python -c 'import os; print(os.environ)'"), 2),
    ("bash env VAR=1 cmd", bash("env JAX_PLATFORMS=cpu uv run python train.py"), 0),
    ("bash export a var", bash("export JAX_PLATFORMS=cpu && uv run python train.py"), 0),
    ("bash stop the model server", bash("bash scripts/vllm.sh stop"), 2),
    ("bash pkill vllm", bash("pkill -f vllm"), 2),
    ("bash kill a pid", bash("sleep 1; kill 1234"), 2),
    ("bash read the skill (not kill)", bash("cat skills/mic-plots/SKILL.md"), 0),
    ("claude Write feedback.md", claude("Write", file_path=f"{R}/feedback.md", content="x"), 2),
    ("bash write feedback.md", bash(f"echo hi >> {R}/feedback.md"), 2),
    ("bash write a worker's session", bash(f"echo {{}} > {R}/session.w1.json"), 2),
    ("bash write a worker's feedback", bash(f"echo hi >> {R}/feedback.w2.md"), 2),
    ("claude Write wandb sync state", claude("Write", file_path=f"{R}/wandb_sync.json", content="{}"), 2),
    # git: commits and pushes only under the prefix
    ("git push an agent branch", bash(f"git push -u origin {PREFIX}/{EXP}/x"), 0),
    ("git push agent branch (HEAD:)", bash(f"git push origin HEAD:{PREFIX}/{EXP}/x"), 0),
    ("git push the experiment branch", bash(f"git push origin {PREFIX}/{EXP}/main"), 2),
    ("git branch -D experiment branch", bash(f"git branch -D {PREFIX}/{EXP}/main"), 2),
    ("git push the base branch", bash(f"git push origin {BASE}"), 2),
    ("git push HEAD:main", bash("git push origin HEAD:main"), 2),
    ("git push other prefix", bash("git push origin hyp/x"), 2),
    ("git push --force", bash(f"git push --force origin {PREFIX}/x"), 2),
    ("git push +refspec (force)", bash(f"git push origin +{PREFIX}/x"), 2),
    ("git push --all", bash("git push --all origin"), 2),
    ("git push after cd &&", bash(f"cd /tmp && git -C {DEMO} push origin {BASE}"), 2),
    ("git switch && commit", bash(f"git switch {BASE} && git commit -m x"), 2),
    ("git commit message with 'push'", bash('git log -1 --format="%s; git push origin main"'), 0),
    ("git branch -D base", bash(f"git branch -D {BASE}"), 2),
    ("git branch -D agent branch", bash(f"git branch -D {PREFIX}/{EXP}/x"), 0),
    (f"git commit on {CURRENT}", bash('git add src && git commit -m "hyp: x [skill: y]"'), 0 if ON_AGENT_BRANCH else 2),
    ("git commit -am", bash('git commit -am "x"'), 2),
    ("git commit --amend", bash("git commit --amend --no-edit"), 2),
    ("git reset --soft", bash("git reset --soft HEAD~1"), 2),
    ("git stash", bash("git stash"), 2),
    ("git switch", bash(f"git switch {BASE}"), 2),
    ("git checkout a file", bash("git checkout -- src/model.py"), 2),
    ("git grep", bash("git grep ELPD"), 2),
    ("git cat-file", bash("git cat-file -p 1234abcd"), 2),
    ("git add -A", bash("git add -A"), 2),
    ("git add src", bash("git add src"), 0),
    ("git add src/model.py", bash("git add src/model.py"), 0),
    ("git restore src", bash("git restore src"), 0),
    ("git restore .", bash("git restore ."), 2),
    ("git restore --staged :/", bash("git restore --staged :/"), 0),
    ("git status", bash("git status --short"), 0),
    ("git --no-pager log --oneline", bash("git --no-pager log --oneline -5"), 0),
    ("git log -p (whole history)", bash("git log -p"), 2),
    ("git log -p -- src", bash("git log -p -- src/model.py"), 0),
    ("git log -p -- :/slides", bash("git log -p -- :/slides"), 2),
    ("git show HEAD", bash("git show HEAD"), 2),
    ("git show --stat HEAD", bash("git show --stat HEAD"), 0),
    ("git show HEAD -- src", bash("git show HEAD -- src"), 0),
    ("git show HEAD:demo/src/model.py", bash("git show HEAD:demo/src/model.py"), 0),
    ("git show old:demo/DATASET.md", bash("git show 20c110b:demo/DATASET.md"), 2),
    ("git diff (whole repo)", bash("git diff"), 2),
    ("git diff -- src", bash("git diff -- src"), 0),
    ("git diff --stat", bash("git diff --stat HEAD~1"), 0),
    # Codex payloads: edits arrive as apply_patch patches
    ("codex patch src/model.py", patch("*** Update File: src/model.py", "@@", "-a", "+b"), 0),
    ("codex patch the report draft", patch(f"*** Add File: {R}/x/report_draft.md", "+x"), 0),
    ("codex patch checks/evaluate.py", patch("*** Update File: checks/evaluate.py", "@@", "-a", "+b"), 2),
    ("codex patch src and TASK.md", patch("*** Update File: src/model.py", "@@", "-a", "+b",
                                          "*** Delete File: TASK.md"), 2),
    ("codex patch adds notes.md", patch("*** Add File: notes.md", "+x"), 2),
    ("codex patch moves src into checks", patch("*** Update File: src/model.py", "*** Move to: checks/m.py"), 2),
    ("codex patch DATASET.md", patch("*** Update File: ../dataset/DATASET.md", "@@", "-a", "+b"), 2),
    ("codex patch .codex/hooks.json", patch("*** Update File: .codex/hooks.json", "@@", "-a", "+b"), 2),
    ("codex bash DATASET.md", codex("Bash", "cat ../dataset/DATASET.md"), 2),
    ("codex bash status", codex("Bash", "uv run python checks/status.py"), 0),
    # pi payloads
    ("pi read DATASET.md", pi("read", path="../dataset/DATASET.md"), 2),
    ("bash grep the parent dir", bash("grep -rn ELPD .."), 2),
    ("pi edit src/features.py", pi("edit", path="src/features.py", oldText="a", newText="b"), 0),
    ("pi write skills/", pi("write", path="skills/github-workflow/SKILL.md", content="x"), 2),
    ("pi bash slides", pi("bash", command="cat ../slides/content.yaml"), 2),
]


def run(hook: str, event) -> subprocess.CompletedProcess:
    args = [sys.executable, str(HOOKS / hook), "--harness", "test"]
    if isinstance(event, list):
        return subprocess.run(args + event, capture_output=True, text=True, env=ENV)
    return subprocess.run(args, input=json.dumps(event), capture_output=True, text=True, env=ENV)


def main() -> int:
    results = []

    def check(label: str, r: subprocess.CompletedProcess, want: int, hook: str = "pre_tool_use") -> None:
        ok = r.returncode == want
        results.append(ok)
        print(f"{'ok ' if ok else 'FAIL'} {hook:13s} {label:36s} -> {r.returncode} (want {want})"
              + ("" if ok else f"  {r.stderr.strip()[:160]}"))

    try:
        for label, event, want in PRE:
            check(label, run("pre_tool_use.py", event), want)

        exp = RESULTS / EXP
        check("session start", run("session_start.py", {"hook_event_name": "SessionStart", "source": "startup"}), 0,
              "session_start")
        check("blocks with no experiment yet", run("stop.py", {"hook_event_name": "Stop"}), 2, "stop")
        (exp / "t1").mkdir(exist_ok=True)
        (exp / "t1" / "metrics.json").write_text(json.dumps({"decision": "keep", "pr_url": None}))
        s = json.loads((exp / "session.json").read_text())
        (exp / "session.json").write_text(json.dumps({**s, "experiments": ["t1"]}))
        check("blocks a kept, unpublished result", run("stop.py", {"hook_event_name": "Stop"}), 2, "stop")
        (exp / "t1" / "metrics.json").write_text(json.dumps({"decision": "discard", "reverted": True}))
        clean = not subprocess.run(["git", "status", "--porcelain", "--", "src"], cwd=DEMO,
                                   capture_output=True, text=True).stdout.strip()
        if clean:
            check("allows when the budget is used", run("stop.py", {"hook_event_name": "Stop"}), 0, "stop")
            (exp / "session.json").write_text(json.dumps({**s, "experiments": [], "max_experiments": 0}))
            check("allows a zero budget (feedback --stop)", run("stop.py", {"hook_event_name": "Stop"}), 0, "stop")
        else:
            print("skip stop          allows-when-done case (src/ has uncommitted changes)")

        tmp = DEMO / "_hooktest_staged.txt"
        tmp.write_text("x")
        subprocess.run(["git", "add", "-f", str(tmp)], cwd=DEMO, check=True)
        try:
            sys.path.insert(0, str(HOOKS))
            import pre_tool_use
            outside = pre_tool_use.staged_outside_src()
            ok = any(p.endswith("_hooktest_staged.txt") for p in outside)
            results.append(ok)
            print(f"{'ok ' if ok else 'FAIL'} pre_tool_use  {'sees files staged outside src/':36s} -> {outside}")
        finally:
            subprocess.run(["git", "rm", "-q", "--cached", str(tmp)], cwd=DEMO)
            tmp.unlink()

        (exp / "feedback.md").write_text("Try the interaction next.\n")
        r = run("feedback.py", {"hook_event_name": "Feedback"})
        ok = r.returncode == 0 and "Try the interaction next." in r.stdout and not (exp / "feedback.md").exists() \
            and len(list((exp / "feedback").glob("*.md"))) == 1
        results.append(ok)
        print(f"{'ok ' if ok else 'FAIL'} feedback      {'delivers and archives feedback.md':36s} -> {r.returncode}")
        r = run("feedback.py", {"hook_event_name": "Feedback"})
        check("prints nothing when there is none", r, 0, "feedback")
        results[-1] = results[-1] and not r.stdout.strip()

        if "--slow" in sys.argv:
            check("smoke test on src/model.py",
                  run("post_tool_use.py", claude("Edit", file_path="src/model.py", old_string="a", new_string="b")),
                  0, "post_tool_use")
    finally:
        shutil.rmtree(RESULTS / EXP, ignore_errors=True)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
