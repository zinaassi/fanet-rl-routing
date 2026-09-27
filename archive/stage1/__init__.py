"""Stage 1: standalone classical routing baseline for the UAV routing project.

Static drones, full global information, one-shot routing computed at t=0,
then a queued packet simulation to measure delivery.

This package is intentionally self-contained: it imports nothing from the
rest of the repository and depends only on numpy, networkx and matplotlib
(plus pytest for the test suite).
"""
