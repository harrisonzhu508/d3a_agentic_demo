"""Deliver human feedback to a running agent: text in results/<experiment>/feedback.md reaches the agent at its
next step (the pi extension calls this after every tool call), then moves to results/<experiment>/feedback/.

Write feedback with scripts/feedback.sh, or edit the file directly. Prints the message for the agent, or nothing.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import experiment_dir, feedback_file, log, read_event, settings  # noqa: E402

HEADER = ("Feedback from me, the human running this experiment (sent {time} from scripts/feedback.sh). Act on it "
          "now; where it conflicts with program.md, it wins. The harness rules (hooks, keep rule) still apply.\n\n")


def pending() -> str:
    """The pending feedback (archived as it is read), or ''."""
    f = feedback_file()
    if not f.exists() or not f.read_text().strip():
        return ""
    text = f.read_text().strip()
    now = dt.datetime.now()
    stamp = now.strftime("%Y%m%d-%H%M%S")
    archive = experiment_dir() / "feedback"
    archive.mkdir(exist_ok=True)
    w = settings()["worker"]
    f.rename(archive / (f"{stamp}-{w}.md" if w else f"{stamp}.md"))
    return HEADER.format(time=now.strftime("%H:%M")) + text


def main() -> None:
    ev = read_event()
    msg = pending()
    if msg:
        log(ev, "feedback", msg.split("\n\n", 1)[1][:300])
        print(msg)
    sys.exit(0)


if __name__ == "__main__":
    main()
