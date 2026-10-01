# Working with AI agents on YARP

This file sets out how AI coding agents (Claude Code, Codex, and others) work
with YARP developers in this repo. It applies to every branch and PR.

## About YARP

YARP (Yet Another Reaction Program) is developed by the Savoie Research Group.
It has two main features:

1. **Product enumeration.** Break-n, form-m rules generate elementary reaction
   steps to explore chemical reaction space systematically.
2. **High-throughput reaction characterization.** Pipelines screen with
   low-level methods, then refine with high-level methods, to locate transition
   states for elementary steps reliably and efficiently.

The 3.0.X versions overhaul earlier code in two ways:

- The characterization pipeline runs in containers, so YARP moves more easily
  between computing systems and new methods are easier to add.
- Internal bookkeeping and reaction labeling are more robust, to support deep
  exploration and analysis of chemical reaction networks.

## Roles

- **The developer sets the direction.** They supply the ideas, choose the next
  step, and make the design decisions.
- **The agent carries the work out.** It edits code, writes and runs tests, and
  reports what it found.

## Workflow

1. **Ask before starting.** Ask clarifying questions until the task is
   unambiguous, then begin.
2. **One step at a time.** Do only the step the developer picked. Finish it,
   then stop for review. Don't start the next step unasked.
3. **Present design choices; don't make them.** When there is a design decision,
   measure first, then lay out the options with a recommendation. Never choose
   silently.
4. **Never commit or push.** Leave changes in the working tree. The developer
   reviews, approves, and commits them.

## Debug workspace

Each new PR or project gets its own folder under `debug/`:

    debug/YYMMDD_<branch>/
    ├── NOTES.md
    └── checks/
        ├── 01_<name>/
        │   ├── <script>.py
        │   └── <saved output>
        └── 02_<name>/
            └── ...

- **`YYMMDD` is the start date.** For example, `debug/261001_quick_irc`.
- **`debug/` is gitignored.** Its contents are for local review and are never
  committed.

### NOTES.md

- **Keep a running log.** Record the conversation's decisions and every trial,
  including failed ones and why they failed.
- **Fix stale text in place.** The file should never state something that is no
  longer true.
- **Read it before resuming.** When continuing a workstream, read its NOTES.md
  first. Trust its changelog and git history over its summary sections.

### Scripts

- **Every script lives in `debug/`.** Any script used to test an idea or draw a
  conclusion goes in its own `checks/NN_<name>/` folder, never only in a scratch
  or temp directory.
- **Save its output alongside it.** The developer should be able to read,
  review, and re-run every check.

## Evidence

- **Back every claim with a check** that can be re-run.
- **Say what was measured and what was read from the code.**
- **Show every regression test failing with its fix reverted.** Patch the name
  where the test actually looks it up; otherwise the revert check passes for the
  wrong reason.

## Repo rules

- **Never edit anything under `tutorials/`,** not even a one-character fix. The
  tutorials are snapshots of past versions. Report problems instead.
