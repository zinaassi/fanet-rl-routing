"""Training examples, batching, and the torch boundary (Phase 1b)."""

from __future__ import annotations

import os
import sys

import pytest

from fanet_sim import config
from fanet_sim.envs.fanet_env import FANETEnv

from agents import make_router
from agents.tracker import Tracker


# ---------------------------------------------------------------------------
# The torch boundary
# ---------------------------------------------------------------------------

def test_the_simulator_imports_without_torch_or_agents() -> None:
    """fanet_sim/ must never reach for torch, nor for the agents package."""
    import subprocess

    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] in ('torch', 'agents'):\n"
        "            raise ImportError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import fanet_sim.envs.fanet_env, fanet_sim.utils.metrics, main\n"
        "assert 'torch' not in sys.modules\n"
        "print('clean')\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-c", probe], cwd=root,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_the_rate_rule_needs_no_torch() -> None:
    """"rate" is pure Python, so it must build with torch unavailable."""
    import subprocess

    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        if name.split('.')[0] == 'torch':\n"
        "            raise ImportError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from agents import make_router\n"
        "r = make_router('rate', run_seed=1)\n"
        "assert r.name == 'rate'\n"
        "assert 'torch' not in sys.modules\n"
        "print('clean')\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = subprocess.run([sys.executable, "-c", probe], cwd=root,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def test_targets_are_one_for_delivered_and_zero_for_lost() -> None:
    """Every decision on a packet's path gets the packet's own outcome."""
    tracker = Tracker()
    for packet_id, delivered in ((1, True), (2, False)):
        for drone_id, next_hop in [(0, 1), (1, 2), (2, "GS")]:
            tracker.record_decision(
                packet_id=packet_id, created_at=0, drone_id=drone_id,
                next_hop=next_hop, features=(0.1, 0.2, 0.3),
            )
        examples = tracker.resolve(packet_id, delivered=delivered)
        assert len(examples) == 3
        assert {target for _, target in examples} == {1.0 if delivered else 0.0}


def test_every_example_carries_the_inputs_its_decision_was_made_from() -> None:
    """The stored features travel with the decision, unchanged."""
    tracker = Tracker()
    rows = [(0.9, 0.1, 0.02), (0.4, 0.5, 0.30)]
    for index, row in enumerate(rows):
        tracker.record_decision(packet_id=1, created_at=0, drone_id=index,
                                next_hop=index + 1, features=row)
    examples = tracker.resolve(1, delivered=True)
    assert [features for features, _ in examples] == rows


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_a_gradient_step_waits_for_a_full_batch() -> None:
    """Fewer than RL_BATCH_SIZE examples accumulate; the batch then clears."""
    router = make_router("rl", run_seed=1, training=True, init_seed=0)
    batch = config.RL_BATCH_SIZE

    router._learn([((0.5, 0.0, 0.1), 1.0)] * (batch - 1))
    assert len(router.pending) == batch - 1
    assert router.batch_losses == []

    router._learn([((0.5, 0.0, 0.1), 1.0)])
    assert router.pending == []
    assert len(router.batch_losses) == 1


def test_evaluation_mode_never_learns() -> None:
    """With training off, examples resolve but no gradient step is taken."""
    router = make_router("rl", run_seed=1, training=False, init_seed=0)
    before = [p.detach().clone() for p in router.net.parameters()]

    router._learn([((0.5, 0.0, 0.1), 1.0)] * (config.RL_BATCH_SIZE * 3))

    assert router.batch_losses == []
    for was, now in zip(before, router.net.parameters()):
        assert (was == now).all(), "weights moved while evaluating"


def test_training_moves_the_weights_and_records_a_loss() -> None:
    """A full batch produces one Adam step and one recorded BCE."""
    router = make_router("rl", run_seed=1, training=True, init_seed=0)
    before = [p.detach().clone() for p in router.net.parameters()]

    router._learn([((0.9, 0.0, 0.02), 1.0)] * config.RL_BATCH_SIZE)

    assert len(router.batch_losses) == 1
    assert router.mean_batch_loss() == pytest.approx(router.batch_losses[0])
    assert any(not (was == now).all()
               for was, now in zip(before, router.net.parameters()))


def test_a_training_run_resolves_examples_and_takes_steps() -> None:
    """End to end: a short training run learns from real resolved packets."""
    router = make_router("rl", run_seed=9, training=True, init_seed=0)
    env = FANETEnv(routing="rl", log_path=os.devnull,
                   placement_seed=9, run_seed=9, router=router)
    env.reset()
    for _ in range(200):
        env.step()
    env.close_logger()

    assert router.batch_losses, "no gradient step was taken in 200 steps"
    assert router.mean_batch_loss() is not None
    assert 0.0 < router.mean_batch_loss() < 5.0


def test_reset_clears_pending_examples_but_keeps_the_network() -> None:
    """A new run starts with empty buffers and the weights it had."""
    router = make_router("rl", run_seed=1, training=True, init_seed=0)
    router._learn([((0.5, 0.0, 0.1), 1.0)] * (config.RL_BATCH_SIZE + 5))
    weights = [p.detach().clone() for p in router.net.parameters()]

    router.on_reset()

    assert router.pending == [] and router.batch_losses == []
    for was, now in zip(weights, router.net.parameters()):
        assert (was == now).all()


def test_parallel_evaluation_matches_a_sequential_one() -> None:
    """A worker pool must give exactly the sequential numbers.

    Each evaluation run is fully determined by its own seeds and a frozen
    network, so farming them out may not change a single digit.
    """
    from scripts.train_1b import evaluate

    seeds = (301, 302, 303, 304)
    sequential = evaluate("rate", seeds, workers=1)
    parallel = evaluate("rate", seeds, workers=4)

    assert sequential == parallel


def test_parallel_evaluation_matches_sequentially_for_the_network_too() -> None:
    """Same check with "rl", where frozen weights travel to each worker."""
    from agents.rl_router import LinkScoreNet
    from scripts.train_1b import evaluate

    import torch
    torch.manual_seed(0)
    state = LinkScoreNet().state_dict()

    seeds = (301, 302)
    sequential = evaluate("rl", seeds, state_dict=state, workers=1)
    parallel = evaluate("rl", seeds, state_dict=state, workers=2)

    assert sequential == parallel


# ---------------------------------------------------------------------------
# Evaluating a subset of the rules
# ---------------------------------------------------------------------------

def test_the_non_rl_rules_evaluate_with_no_model_present() -> None:
    """greedy, random and rate can be measured without a trained network.

    Only "rl" needs one. The evaluation scripts used to load the models
    before running anything, so deleting out/models/ blocked every rule,
    including the three that need no model at all.

    Checked against a directory that is guaranteed to hold no model, rather
    than against out/, so the test does not depend on whether a model
    happens to be lying around.
    """
    import tempfile

    from scripts import evaluate_1b_large as big

    rules = big.parse_rules(["greedy,random,rate"])
    assert not any(rule in big.MODEL_RULES for rule in rules)

    # Two layouts at one load: this test is about the model gate, not about
    # the numbers, and the full grid is 300 simulations.
    previous_seeds, previous_loads = big.TEST50_SEEDS, big.LOADS_MS
    big.TEST50_SEEDS, big.LOADS_MS = (1, 2), (200,)
    try:
        with tempfile.TemporaryDirectory() as empty:
            assert not os.path.exists(os.path.join(empty, "models"))
            rows = big.run_grid(empty, [0], workers=1, rules=rules)
    finally:
        big.TEST50_SEEDS, big.LOADS_MS = previous_seeds, previous_loads

    assert len(rows) == len(rules) * 2, rows
    assert {row["rule"] for row in rows} == set(rules)
    assert all(row["views_agree"] for row in rows)


def test_asking_for_rl_without_a_model_still_stops() -> None:
    """The guard is narrowed, not removed: rl without a model must fail."""
    import tempfile

    from scripts import evaluate_1b_large as big

    # It must stop before running anything, so no layouts are needed.
    with tempfile.TemporaryDirectory() as empty:
        with pytest.raises(SystemExit):
            big.run_grid(empty, [0], workers=1, rules=big.parse_rules(["rl"]))


def test_unknown_rule_names_are_rejected() -> None:
    """A typo in --rules fails loudly instead of silently running nothing."""
    from scripts.evaluate_1b_large import parse_rules

    with pytest.raises(SystemExit):
        parse_rules(["greedy,qroute"])
    with pytest.raises(SystemExit):
        parse_rules([","])
