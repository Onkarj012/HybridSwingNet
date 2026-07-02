"""
Fixed SCSO optimization with proper nested holdout validation.

This wrapper adds a held-out test set that is NEVER used during SCSO optimization,
preventing overfitting to the validation set.
"""

import logging
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any


def run_scso_with_holdout(
    cfg,
    device,
    run_name: str,
    train_samples: List,
    val_samples: List,
    test_samples: List,
    n_iterations: int = 30,
    population_size: int = 10,
) -> Tuple[Dict, List[Dict]]:
    """
    Run SCSO with proper holdout validation.

    The test set is used ONLY for final evaluation, not during optimization.
    This prevents selection bias from 1000 evaluations on the same validation set.

    Args:
        cfg: Configuration object
        device: torch device
        run_name: Name for this run
        train_samples: Training samples
        val_samples: Validation samples (used for SCSO optimization)
        test_samples: Test samples (held out, used only for final evaluation)
        n_iterations: Number of SCSO iterations
        population_size: Population size per iteration

    Returns:
        best_config: Best hyperparameter configuration
        history: Full optimization history with val AND test metrics
    """
    logger = logging.getLogger("stockxpert.scso")

    logger.info("Starting SCSO with nested holdout validation")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples (for SCSO): {len(val_samples)}")
    logger.info(f"Test samples (held out): {len(test_samples)}")

    # Split validation into inner val (for SCSO) and test (for final eval)
    # Actually, we already have test_samples, so use those as the true holdout

    history = []
    best_val_score = -np.inf
    best_config = None
    best_test_score = None

    # Initialize population with random hyperparameters
    population = _initialize_population(population_size, cfg)

    for iteration in range(n_iterations):
        logger.info(f"\nSCSO Iteration {iteration + 1}/{n_iterations}")

        iteration_results = []

        for i, hyperparams in enumerate(population):
            # Train with these hyperparameters
            config_name = f"{run_name}_scso_{iteration}_{i}"

            # Evaluate on validation set (used for optimization)
            val_metrics = _train_and_evaluate(
                cfg, device, train_samples, val_samples, hyperparams, config_name
            )

            # Evaluate on test set (held out, for tracking only)
            test_metrics = (
                _train_and_evaluate(
                    cfg,
                    device,
                    train_samples,
                    test_samples,
                    hyperparams,
                    config_name + "_test",
                    skip_training=True,  # Use same model, just eval
                )
                if test_samples
                else None
            )

            val_score = val_metrics.get("avg_dir_acc", 0)

            result = {
                "iteration": iteration,
                "individual": i,
                "hyperparams": hyperparams,
                "val_score": val_score,
                "test_score": test_metrics.get("avg_dir_acc", 0)
                if test_metrics
                else None,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }

            iteration_results.append(result)
            history.append(result)

            # Update best (based ONLY on validation score during optimization)
            if val_score > best_val_score:
                best_val_score = val_score
                best_config = hyperparams.copy()
                best_test_score = result["test_score"]
                logger.info(
                    f"  New best validation score: {val_score:.4f} "
                    f"(Test: {result['test_score']:.4f if result['test_score'] else 'N/A'})"
                )

        # Update population (simplified SCSO update)
        population = _update_population(population, iteration_results, population_size)

    # Final evaluation: Train best config and report both val and test scores
    logger.info("\n" + "=" * 70)
    logger.info("SCSO OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Validation Score: {best_val_score:.4f}")
    logger.info(f"Best Test Score: {best_test_score:.4f if best_test_score else 'N/A'}")
    logger.info(f"Generalization Gap: {best_val_score - (best_test_score or 0):.4f}")

    return best_config, history


def _initialize_population(n: int, cfg) -> List[Dict]:
    """Initialize random hyperparameter population."""
    import random

    population = []
    for _ in range(n):
        individual = {
            "lr": 10 ** random.uniform(-4, -2),
            "weight_decay": 10 ** random.uniform(-5, -3),
            "dropout": random.uniform(0.1, 0.3),
            "hidden_dim": random.choice([64, 96, 128]),
            "batch_size": random.choice([128, 256, 512]),
        }
        population.append(individual)
    return population


def _train_and_evaluate(
    cfg, device, train_samples, eval_samples, hyperparams, name, skip_training=False
):
    """Train model and evaluate on given samples."""
    # This is a placeholder - the actual implementation would call the training loop
    # For now, return dummy metrics
    return {
        "avg_dir_acc": np.random.uniform(0.5, 0.8),
        "h1_acc": np.random.uniform(0.5, 0.9),
    }


def _update_population(
    population: List[Dict], results: List[Dict], population_size: int
) -> List[Dict]:
    """Update population using SCSO strategy (simplified)."""
    # Sort by validation score
    sorted_results = sorted(results, key=lambda x: x["val_score"], reverse=True)

    # Keep top 50%
    n_keep = population_size // 2
    new_population = [r["hyperparams"] for r in sorted_results[:n_keep]]

    # Generate new individuals by mutating best
    import random
    import copy

    while len(new_population) < population_size:
        parent = copy.deepcopy(sorted_results[0]["hyperparams"])
        # Mutate
        if random.random() < 0.5:
            parent["lr"] *= random.uniform(0.8, 1.2)
        if random.random() < 0.5:
            parent["dropout"] = min(
                0.5, max(0.1, parent["dropout"] + random.uniform(-0.05, 0.05))
            )
        new_population.append(parent)

    return new_population
