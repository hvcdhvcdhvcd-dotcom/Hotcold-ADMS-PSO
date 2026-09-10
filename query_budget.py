# -*- coding: utf-8 -*-
"""Detector-query budget calculator for HOTCOLDBlock PSO variants.

Counts detector forward passes (queries) per batch / per epoch straight from
the algorithm logic, so old-vs-new comparisons can be aligned at equal query
budget instead of equal epochs.

New ADMS-PSO extra costs per run():
  - cooperative search:  floor(iterations / cooperative_interval) * swarm * (1 + num_patches)
  - stagnation recovery: at most floor(iterations / stagnation) events, each costing
                         ceil(levy_fraction*swarm) * num_patches * 2 + swarm
"""


def per_batch(swarm, iterations, cooperative_interval=5, num_patches=4,
              levy_fraction=0.30, stagnation_iterations=10,
              has_initial_eval=False):
    """has_initial_eval: new ADMS-PSO evaluates the whole swarm once before
    the iteration loop (old vanilla PSO did not)."""
    init = swarm if has_initial_eval else 0
    base = iterations * swarm
    coop_events = iterations // cooperative_interval
    cooperate = coop_events * swarm * (1 + num_patches)
    worst = max(1, round(levy_fraction * swarm))
    max_recover_events = max(0, iterations // stagnation_iterations)
    recover_max = max_recover_events * (worst * num_patches * 2 + swarm)
    lo = init + base + cooperate          # no stagnation recovery triggered
    hi = lo + recover_max                 # worst-case recovery count
    return lo, hi


def per_epoch(lo, hi, batches=37):
    return lo * batches, hi * batches


if __name__ == "__main__":
    print("=" * 78)
    print("HOTCOLDBlock detector-query budget (batch_size=32, 37 batches/epoch)")
    print("=" * 78)

    configs = [
        ("old  (swarm=100, iters=3 )", 100, 3, False),
        ("new  (swarm=20,  iters=20)", 20, 20, True),
        ("new  (swarm=30,  iters=100)", 30, 100, True),
        ("new  (swarm=16,  iters=10)", 16, 10, True),
    ]
    rows = []
    for label, swarm, iters, init in configs:
        lo, hi = per_batch(swarm, iters, has_initial_eval=init)
        elo, ehi = per_epoch(lo, hi)
        rows.append((label, lo, hi, elo, ehi))
        print(f"{label}: per-batch {lo:>6,} - {hi:>6,} | per-epoch {elo:>9,} - {ehi:>9,}")

    # key comparison: current new run vs previous old run
    old_lo, _ = per_batch(100, 3, has_initial_eval=False)
    new_lo, new_hi = per_batch(20, 20, has_initial_eval=True)
    print("-" * 78)
    print(f"current new/old ratio per epoch:  baseline {new_lo/old_lo:.2f}x, "
          f"max {new_hi/old_lo:.2f}x")
    print(f"old 10 epochs ~= {old_lo*37*10:,} queries")
    print(f"new 7 epochs  ~= {new_lo*37*7:,} - {new_hi*37*7:,} queries")
    print(f"new epochs to match old 10-epoch budget: "
          f"~{old_lo*37*10/(new_lo*37):.1f} (baseline) - "
          f"~{old_lo*37*10/(new_hi*37):.1f} (max)")
