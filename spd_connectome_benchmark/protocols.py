"""Explicit scientific policies shared by benchmark entry points."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarmonizationPolicy:
    """How split-local harmonization treats one held-out fold."""

    fit_scope: str
    apply_to_test: bool
    test_target_used_by_preprocessor: bool
    label: str


def harmonization_policy(cv: str) -> HarmonizationPolicy:
    """Return the existing paper-aligned policy for a CV protocol.

    Grouped K-fold uses the known test dataset and true age covariate when
    applying the training-fitted transform. LODO is source-only: the unseen
    target dataset is not transformed. This function makes the distinction
    explicit; it does not alter the established numerical protocol.
    """
    protocol = str(cv).strip().lower()
    if protocol in {"gkf", "kfold"}:
        return HarmonizationPolicy(
            fit_scope="outer_train",
            apply_to_test=True,
            test_target_used_by_preprocessor=True,
            label="known_dataset_known_age_test",
        )
    if protocol == "lodo":
        return HarmonizationPolicy(
            fit_scope="outer_train",
            apply_to_test=False,
            test_target_used_by_preprocessor=False,
            label="source_only",
        )
    raise ValueError(f"Unsupported harmonization protocol: {cv!r}")
