"""Scene-aware batch sampler.

Not implemented in Phase 1 (data pipeline). The training engine built in
Phase 5 needs a sampler that avoids mixing temporal queues from different
scenes within a batch in ways that break BEV temporal fusion; that logic
belongs here once the trainer exists to consume it.
"""

from __future__ import annotations

from torch.utils.data import Sampler


class SceneAwareSampler(Sampler):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("SceneAwareSampler is implemented in Phase 5 (training engine).")
