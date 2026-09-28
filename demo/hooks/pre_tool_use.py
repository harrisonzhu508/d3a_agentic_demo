"""PreToolUse: block protected reads, writes outside src/, dangerous shell commands, and commits or pushes to any
branch outside the configured prefix (config/autoresearch.toml: git.branch_prefix)."""

import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DEMO, POLICY, READ_TOOLS, WRITE_TOOLS, allow, block, denied, experiment_branch,  # noqa: E402
                    is_agent_branch, read_event, resolve, settings, under)

EDITABLE = ["src", "results/*/*/report_draft.md"]   # the model code, and the text of a hypothesis report
MUTATING = r"(?:>>?|\btee\b|\bsed\s+-i|\b(?:cp|mv|rm|touch|truncate|ln|install|rsync)\b|\bgit\s+(?:checkout|restore|rm|mv)\b)"

# Git for agents is an allowlist. Branch switches, resets and discards go through the github-workflow scripts.
GIT_READ = {"status", "diff", "log", "show", "rev-parse", "ls-files", "describe", "shortlog", "show-branch", "branch"}
GIT_WRITE = {"add", "restore", "commit", "push"}
# history commands that print file contents need a pathspec inside the agent's files (older commits hold the slides)
HISTORY_PATHS = ["src", "checks", "skills", "train.py", "TASK.md", "program.md", "AGENTS.md", "results"]
QUIET = {"--stat", "--shortstat", "--numstat", "--name-only", "--name-status", "--no-patch", "-s", "--summary"}
COMMIT_FLAGS = r"-[A-Za-z]*[aio][A-Za-z]*|--(?:all|include|only|amend)"   # -a, -am, --amend, ...


def current_branch() -> str:
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=DEMO,
                          capture_output=True, text=True).stdout.strip()


def git_commands(cmd: str) -> list[list[str]]:
    """The arguments after `git` (and its global options) of every git command in a shell command line."""
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=True)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError:        # unbalanced quotes: fall back to a plain split
        tokens = cmd.split()
    out, cur = [], []
    for t in tokens + [";"]:
        if t and set(t) <= set("&|;()"):
            if "git" in cur:
                args = cur[cur.index("git") + 1:]
                while args and args[0].startswith("-"):          # global options: -C dir, -c k=v, --no-pager
                    args = args[2:] if args[0] in ("-C", "-c") else args[1:]
                out.append(args)
            cur = []
        else:
            cur.append(t)
    return out


def push_targets(args: list[str], current: str) -> list[str]:
    """Remote branches a `git push` would update (a leading + means a forced update)."""
    if any(a in ("--all", "--mirror", "--tags", "--prune") for a in args):
        return ["(all branches)"]
    pos = [a for a in args if not a.startswith("-")]
    refs = pos[1:] or [current]
    targets = []
    for ref in refs:
        dst = ref.split(":")[-1] if ":" in ref else ref
        dst = current if dst in ("HEAD", "") else dst
        targets.append(("+" if ref.startswith("+") else "") + dst.removeprefix("refs/heads/"))
    return targets


def pathspec(args: list[str]) -> list[str]:
    """Paths given to a git command: everything after `--`, else the non-option arguments."""
    return args[args.index("--") + 1:] if "--" in args else [a for a in args if not a.startswith("-")]


def inside(paths: list[str], allowed: list[str]) -> bool:
    return bool(paths) and all(any(under(resolve(p), e) for e in allowed) for p in paths)


def git_out(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=DEMO, capture_output=True, text=True).stdout


def staged_outside_src() -> list[str]:
    here = git_out("rev-parse", "--show-prefix").strip()          # e.g. "demo/"
    return [p for p in git_out("diff", "--cached", "--name-only").split() if not p.startswith(here + "src/")]


def prints_contents(sub: str, rest: list[str]) -> bool:
    """Would this log/show/diff print file contents from history (patches or blobs) outside the agent's files?"""
    if QUIET & set(rest) or (sub == "log" and not {"-p", "-u", "--patch"} & set(rest)):
        return False
    if "--" in rest:
        return not inside(rest[rest.index("--") + 1:], HISTORY_PATHS)
    here = git_out("rev-parse", "--show-prefix").strip()
    blobs = [a.split(":", 1)[1] for a in rest if not a.startswith("-") and ":" in a]
    if sub == "show" and blobs:                                    # git show <rev>:<path>
        paths = [b.removeprefix("./") if b.startswith("./") else b.removeprefix(here) for b in blobs]
        return len(blobs) < len([a for a in rest if not a.startswith("-")]) or not inside(paths, HISTORY_PATHS)
    return True


def check_git(ev: dict, cmd: str) -> None:
    prefix = settings()["branch_prefix"] + "/"
    current = current_branch()
    scripts = "skills/github-workflow/scripts/ (new_branch.py, publish.py, discard.py)"
    for args in git_commands(cmd):
        sub, rest = (args[0], args[1:]) if args else ("", [])
        if sub not in GIT_READ | GIT_WRITE:
            block(ev, f"Blocked: agents do not run `git {sub}`. Branches, resets and discards go through {scripts}; "
                      "if something is in a state they cannot handle, say so in your final message.")
        if sub in ("log", "show", "diff") and prints_contents(sub, rest):
            block(ev, f"Blocked: limit `git {sub}` to your files with a pathspec, e.g. `git {sub} ... -- src`.")
        if sub == "add" and not inside(pathspec(rest), ["src"]):
            block(ev, "Blocked: stage only src/ (git add src).")
        if sub == "restore" and not ("--staged" in rest and "--worktree" not in rest and "-W" not in rest) \
                and not inside(pathspec(rest), ["src"]):
            block(ev, "Blocked: agents restore only src/ (git restore src).")
        if sub == "commit":
            if current == experiment_branch():
                block(ev, f"Blocked: {current} is the experiment branch; it changes only through reviewed pull "
                          "requests. Start a hypothesis branch with skills/github-workflow/scripts/new_branch.py.")
            if not is_agent_branch(current):
                block(ev, f"Blocked: you are on '{current}'. Agents commit only to {prefix}... branches; start one "
                          "with uv run python skills/github-workflow/scripts/new_branch.py <slug>")
            if any(re.fullmatch(COMMIT_FLAGS, a) for a in rest) or "--" in rest:
                block(ev, "Blocked: commit what you staged with `git add src`, as a new commit (no -a, --amend, paths).")
            outside = staged_outside_src()
            if outside:
                block(ev, "Blocked: files outside src/ are staged: " + ", ".join(outside[:8]) + ". Unstage everything "
                          "with `git restore --staged :/` (the files stay as they are), then `git add src` and commit.")
        if sub == "push":
            for t in push_targets(rest, current):
                if t == experiment_branch():
                    block(ev, f"Blocked: {t} is pushed by skills/github-workflow/scripts/publish.py, not by hand.")
                if t.startswith("+") or not is_agent_branch(t):
                    block(ev, f"Blocked: agents may push only {prefix}... branches, without force (target: {t}). "
                              "Publish a kept result with skills/github-workflow/scripts/publish.py.")
        if sub == "branch" and [a for a in rest if not a.startswith("-")]:
            for b in (a for a in rest if not a.startswith("-")):
                if not is_agent_branch(b) or b == experiment_branch():
                    block(ev, f"Blocked: agents create, delete or rename only {prefix}... branches (not '{b}').")


def check_bash(ev: dict, cmd: str) -> None:
    if re.search(r"(?:^|[\s'\"=:(])\.\.(?:[/\s'\")]|$)", cmd):
        block(ev, "Blocked by the harness: stay inside demo/, your workspace (no `..` paths).")
    if re.search(r"(?:^|[;&|(]\s*)(?:env|set|export\s+-p|declare\s+-x|export)\s*(?:$|[|;&)>])", cmd):
        block(ev, "Blocked by the harness: the environment holds credentials; do not print it.")
    if re.search(r"(?:^|[;&|(]\s*)(?:sudo\s+)?(?:kill|pkill|killall)\b", cmd):
        block(ev, "Blocked by the harness: do not stop processes (the model server runs on this machine).")
    for tok in POLICY["bash_deny_tokens"]:
        if tok in cmd:
            block(ev, f"Blocked by the harness: commands containing '{tok}' are not allowed here. "
                      "See TASK.md (rules) and use the github-workflow skill for git.")
    for entry in POLICY["write_deny"]:
        target = re.escape(entry.rstrip("/")).replace(r"\*", r"[^/\s'\"]*")
        if re.search(MUTATING + r"[^|;&]*?(?:^|[\s'\"=/.])" + target + r"(?:[/\s'\"]|$)", cmd):
            block(ev, f"Blocked by the harness: '{entry}' is read-only; only src/ may be changed.")
    if re.search(r"\bgit\b", cmd):
        check_git(ev, cmd)


def main() -> None:
    ev = read_event()
    tool, paths = ev["tool"], ev["paths"]
    if tool in READ_TOOLS | WRITE_TOOLS:
        outside = [raw for raw in paths if not resolve(raw).is_relative_to(DEMO)]
        if outside:
            block(ev, f"Blocked by the harness: {outside[0]} is outside demo/, your workspace.")
        hit = denied(paths, POLICY["read_deny"])
        if hit:
            block(ev, f"Blocked by the harness: {hit} is off-limits to the agent. Work from data/dev.csv, "
                      "TASK.md and the skills.")
    if tool in WRITE_TOOLS:
        hit = denied(paths, POLICY["write_deny"])
        if hit:
            block(ev, f"Blocked by the harness: {hit} is read-only. You may only edit files in src/.")
        for raw in paths:
            if not any(under(resolve(raw), e) for e in EDITABLE):
                block(ev, f"Blocked by the harness: {raw} is outside src/. Change the model only through "
                          "src/features.py, src/model.py and src/sampler.py (and the report text in "
                          "results/<experiment>/<name>/report_draft.md).")
    if tool == "bash" and ev["command"]:
        check_bash(ev, ev["command"])
    allow(ev)


if __name__ == "__main__":
    main()
