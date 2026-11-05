# Agent notes for moebius-cpu

Rules for anyone (human or agent) working in this repo. Design and measurements are in `docs/PLAN.md`.

## Privacy

- Never send the user's email address, or any other personal detail, to an outside service: no
  `User-Agent` strings, URLs, headers or payloads. Use a generic client name. A site asking automated
  clients for contact details is not a reason to do it.

## Project rules

- Pure Rust, CPU only. No C libraries in the engine. Python is offline tooling only.
- Exact preset results must not change: `cargo test --release --test parity -- --ignored --test-threads=1`.
- Do not move a gate to pass it. Report the number, and let the user decide.

## Running benchmarks

- Long runs must be detached with output in `weights/logs/`. Session restarts kill attached shells.
- One heavy process at a time. Timing needs an idle system, and CPU power limits cycle, so
  compare configurations in interleaved rounds.
- Windows steers background console processes to the efficiency cores (~15x slower) unless a process
  opts out. `Engine::load` does. Anything benchmarking without `Engine` must call
  `engine::cpu::disable_power_throttling` itself.
