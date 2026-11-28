# Roadmap

Living version with status: https://claude.ai/artifact/YWy3bV6TqXLWLTfGJrjUwj

| Phase | Tracks (parallel) | Gate to the next phase |
| --- | --- | --- |
| 0 Contracts | all, as stubs | stub pipeline runs, tests green ✅ |
| 1 First loop | T1, T2, T3, T4 | agents raise success under randomization on a Franka pick-and-place, no human edits |
| 2 Data engine | T5, T4, T8 prep | a learned policy matches or beats its scripted teacher under shifted conditions |
| 3 Real world | T6, T7 | a sim-hardened skill succeeds on a real arm, measured |
| 4 Scale | T8, T5, T3 | ~50 tasks, skill library, benchmark results worth a paper |
| 5 Launch | | paper, open-source release, DARWIN adapter if wanted, company decision |

Project focus (Team Echo): a scoped skill library across arms, see [`PROJECT.md`](PROJECT.md)
and [`tracks/scoped-library.md`](tracks/scoped-library.md).
