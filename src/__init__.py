"""EEG anesthesia-depth estimation package.

Phase 1 provides VitalDB access. Phase 2 filters EEG, extracts spectral and
time-domain features, and aligns each window with the recorded BIS value.
Phase 3 trains regression models that predict BIS from those features.
"""
