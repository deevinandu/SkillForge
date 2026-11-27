# T3 · Agents: one skill library across two arms

**Goal:** the forge loop improves `ToppleChain` on the domino tasks from failure evidence, and
a skill accepted on one arm seeds the other: "skill transfer" measured, not claimed.

## Deliverables
1. **Evidence the improver can use.** For domino tasks the useful facts are in the
   events (`disturbance:*`, `toppled:*`, `release:*`) and the skill's own `log` (survey
   verdicts, repairs, called-off pushes). Put the skill log of failed seeds into the improver's
   evidence (via the sandbox worker), trimmed.
2. **Forge on dominoes.** `skillforge forge tasks/franka_topple_domino_chain_knock.json
   --seed-skill dominoes:ToppleChain` and the setup task; record what the agent changes and the paired gate result.
3. **Cross-arm library.** `SkillLibrary` keyed by (task family, robot). Transfer protocol:
   take the skill accepted on Franka, evaluate it as-is on the Kinova task (zero-shot), then
   let the improver adapt it with a fixed budget (rounds, $). Report zero-shot rate, adapted
   rate, rounds and cost, and the diff the agent made (which constants were arm-specific).
4. Do it both ways (Franka -> Kinova and Kinova -> Franka).

## Notes
- Keep a skill one file importing only `skillforge.*` and numpy, or the sandbox will refuse it.
- The paired gate needs the same seeds on both versions; disturbances are seeded, so it holds.
