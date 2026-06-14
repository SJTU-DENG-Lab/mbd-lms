"""Parallel plan for SDAR model (non-MoE)"""

# SDAR is not MoE, so no expert parallelism needed
# This file can be minimal or empty

def get_parallel_plan(config):
    """Return parallel plan for SDAR model"""
    # Non-MoE model, only TP/PP/DP needed
    return None
