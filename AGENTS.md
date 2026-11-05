# Agent notes for moebius-cpu

Rules for anyone (human or agent) working in this repo. Design and measurements are in `docs/PLAN.md`.
The project is finished (M0–M9), so any work from here is maintenance.

## Privacy

- Never send an email address or any other personal detail to an outside service: no `User-Agent`
  strings, URLs, headers or payloads. Use a generic client name. A site asking automated clients for
  contact details is not a reason to do it.

## Project rules

- Pure Rust, CPU only. No C libraries in the engine. Python is offline tooling only.
- Exact preset results must not change: `cargo test --release --test parity -- --ignored --test-threads=1`.
- Do not move a gate to pass it. Report the number, and let the maintainer decide.

## GIMP plugin

- GIMP's Python is MINGW and cannot load MSVC binaries: keep inference in `moebius-cli.exe`. Import
  only GIMP's own modules in `gimp-plugin/moebius-inpaint.py`.
- `moebius-cli --plugin` exits when stdin closes, so a killed plugin leaves no process holding the exe.
- Never host the models: `install.py` downloads them from pinned upstream revisions and
  `moebius-convert` checks their hashes. Release binaries come from CI, because a local build embeds
  local paths.
- `install.py` must stay standard library only, because it runs on GIMP's bundled Python.
- Judge removals with selections that cover the whole object. A selection that cuts through an
  object leaves part of it behind, and Exact does the same.

## Running benchmarks

- Long runs must be detached with output in `weights/logs/`. Session restarts kill attached shells.
- One heavy process at a time. Timing needs an idle system, and CPU power limits cycle, so
  compare configurations in interleaved rounds.
- Windows steers background console processes to the efficiency cores (~15x slower) unless a process
  opts out. `Engine::load` does. Anything benchmarking without `Engine` must call
  `engine::cpu::disable_power_throttling` itself.
