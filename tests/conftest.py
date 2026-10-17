"""Pytest configuration (minimal).

The torch + qiskit-aer static-TLS conflict is handled by running the
circuit-level quantum tests in a dedicated subprocess (see
tests/_quantum_runner.py and tests/test_quantum.py); no global LD_PRELOAD
tricks are needed here.
"""
