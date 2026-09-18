"""Continue the Hanoi pilot's LR smoothly, then use ALOHA's later sharp decay."""


def aloha_training_schedule():
    """Fresh training: warm up, decay through 20k, then drop fivefold and hold."""
    return {
        "warm_up_steps": [2000, 0],
        "cycle_lengths": [20000, 10**9],
        "f_start": [1e-6, 0.06],
        "f_max": [1.0, 0.06],
        "f_min": [0.3, 0.06],
    }


def continuation_schedule(anchor_iteration, decay_iteration=20000):
    if not 100 < anchor_iteration < 10000 < decay_iteration:
        raise ValueError("Continuation must anchor inside the pilot's post-warmup schedule")
    anchor_factor = 0.1 + 0.9 * (10000 - anchor_iteration) / 9900
    return {
        "warm_up_steps": [100, 0, 0],
        "cycle_lengths": [anchor_iteration, decay_iteration - anchor_iteration, 10**9],
        "f_start": [1e-6, anchor_factor, 0.06],
        "f_max": [1.0, anchor_factor, 0.06],
        "f_min": [anchor_factor, 0.3, 0.06],
    }
