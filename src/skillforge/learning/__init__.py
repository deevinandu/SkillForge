"""T5: learning from the scripted teacher.

- `features`: Observation/Action <-> flat vectors, policy features (no torch).
- `dataset`: roll a teacher out (optionally with DART noise, or a learner under DAgger) and
  store episodes under runs/datasets/ (no torch).
- `policy`: behavior cloning (MLP over an observation history -> action chunk), needs torch.
- `skill`: `LearnedSkill`, a checkpoint behind `contracts.Skill`, needs torch.
- `experiment`: `python -m skillforge.learning {collect,train,dagger,eval}`.

torch is the optional `learning` extra; import `policy`/`skill` only where it is installed.
"""
