"""SessionStart: open a new session budget and give the agent the current state of the research."""

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DEMO, log, read_event, session_file, settings  # noqa: E402
from checks.status import summary  # noqa: E402


def main() -> None:
    ev = read_event()
    cfg = settings()
    session = session_file()
    source = str(ev["raw"].get("source", "startup"))
    if source in ("startup", "clear") or not session.exists():
        session.write_text(json.dumps({"harness": ev["harness"], "experiment": cfg["experiment"], "worker": cfg["worker"], "cwd": str(DEMO),
                                       "started": dt.datetime.now().isoformat(timespec="seconds"),
                                       "max_experiments": cfg["max_experiments"], "experiments": []}, indent=2))
    s = json.loads(session.read_text())
    print(f"Autoresearch session ({ev['harness']}): budget {s['max_experiments']} experiments, "
          f"{len(s['experiments'])} done.\n"
          "Before changing anything, read AGENTS.md, TASK.md, program.md and skills/CATALOG.md, and read the SKILL.md "
          "for the change you plan to make.\n\n" + summary(8))
    log(ev, "context", f"session {source}, budget {s['max_experiments']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
