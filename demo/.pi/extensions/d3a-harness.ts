/**
 * d3a-harness: connects pi's lifecycle events to the demo's harness-agnostic hooks in ../../hooks/.
 *
 *   session_start        -> hooks/session_start.py (opens the session budget; context shown on the first prompt)
 *   tool_call            -> hooks/pre_tool_use.py  (exit 2 blocks the tool call, reason goes to the model)
 *   tool_result (edits)  -> hooks/post_tool_use.py (contract smoke test; failures are appended to the result)
 *   tool_result (all)    -> hooks/feedback.py      (human feedback from results/<experiment>/feedback.md, sent as
 *                                                   a user message, like typing in the TUI: steer or follow-up)
 *   agent_before_settle  -> hooks/stop.py          (exit 2 keeps the agent working: the autoresearch loop)
 *
 * Escape in the TUI aborts the current run; the stop hook only continues runs that completed, so it pauses.
 *
 * Loaded by scripts/run_pi.sh with `-e .pi/extensions/d3a-harness.ts`.
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DEMO = (() => {
  try {
    return resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
  } catch {
    return process.cwd();
  }
})();
const PYTHON = resolve(DEMO, ".venv", "bin", "python"); // the project environment (scripts/setup.sh)
const MAX_CONTINUES = 60; // hard cap on stop-hook continuations in one session (safety net)

export default function (pi: ExtensionAPI) {
  let continues = 0;
  let pendingContext = "";

  const run = (hook: string, payload: Record<string, unknown>) =>
    pi.exec(PYTHON, [resolve(DEMO, "hooks", hook), "--harness", "pi", "--payload", JSON.stringify(payload)], {
      cwd: DEMO,
      timeout: 240_000,
    });

  // human feedback (scripts/feedback.sh) goes in as a real user message, exactly as if typed in the TUI
  const deliverFeedback = async (deliverAs: "steer" | "followUp") => {
    const fb = (await run("feedback.py", { event: "Feedback" })).stdout.trim();
    if (fb) pi.sendUserMessage(fb, { deliverAs });
    return Boolean(fb);
  };

  pi.on("session_start", async (event) => {
    const fresh = event.reason === "startup" || event.reason === "new";
    const r = await run("session_start.py", { event: "SessionStart", source: fresh ? "startup" : "resume" });
    pendingContext = r.stdout.trim();
    if (fresh) continues = 0;
  });

  pi.on("before_agent_start", async () => {
    if (!pendingContext) return undefined;
    const content = pendingContext;
    pendingContext = "";
    return { message: { customType: "d3a-session", content, display: true } };
  });

  pi.on("tool_call", async (event) => {
    const r = await run("pre_tool_use.py", { event: "PreToolUse", tool: event.toolName, input: event.input });
    if (r.code === 2) return { block: true, reason: r.stderr.trim() || "Blocked by the harness." };
    return undefined;
  });

  pi.on("tool_result", async (event) => {
    const extra: { type: "text"; text: string }[] = [];
    let isError: boolean | undefined;
    if (event.toolName === "edit" || event.toolName === "write") {
      const r = await run("post_tool_use.py", { event: "PostToolUse", tool: event.toolName, input: event.input });
      if (r.code === 2) {
        extra.push({ type: "text", text: "\n[post-edit check] " + r.stderr.trim() });
        isError = true;
      }
    }
    await deliverFeedback("steer");
    if (!extra.length) return undefined;
    return { content: [...event.content, ...extra], ...(isError ? { isError } : {}) };
  });

  pi.on("agent_before_settle", async (event) => {
    if (event.outcome !== "completed" || continues >= MAX_CONTINUES) return undefined;
    // feedback that arrived after the last tool call starts the next run as a follow-up
    if (await deliverFeedback("followUp")) return undefined;
    const r = await run("stop.py", { event: "Stop" });
    if (r.code !== 2) return undefined;
    continues += 1;
    return {
      entries: [
        ...event.entries,
        { type: "custom_message" as const, customType: "d3a-stop", content: r.stderr.trim(), display: true },
      ],
      continue: true,
    };
  });
}
