"""
Training Monitoring Dashboard for GRPO/GiGPO.

Tracks metrics every N steps on a held-out eval set:
- Core metrics: reward, task completion, format compliance, KL divergence
- Behavioral metrics: tool calls, path diversity, answer length, pairwise BLEU
- Safety metrics: hallucination rate, uncertain match acceptance, T4 FP rate

See phase4_grpo.md Task 4.5 for monitoring requirements.
"""

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.training.grpo.environment import RolloutTrajectory, TaskMetadata
from src.training.grpo.reward import (
    RewardBreakdown,
    RewardHackingAlert,
    detect_reward_hacking,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Metrics computed on evaluation set."""

    step: int
    timestamp: float

    # Core metrics
    avg_reward: float = 0.0
    task_completion_rate: float = 0.0
    format_compliance: float = 0.0
    kl_divergence: float = 0.0

    # Behavioral metrics
    avg_tool_calls: float = 0.0
    tool_path_diversity: int = 0  # Unique tool paths
    avg_answer_length: float = 0.0
    pairwise_bleu: float = 0.0

    # Safety metrics
    hallucination_rate: float = 0.0
    uncertain_match_accept_rate: float = 0.0
    t4_false_positive_rate: float = 0.0

    # Per-tier breakdown
    t1_accuracy: float = 0.0
    t2_accuracy: float = 0.0
    t3_accuracy: float = 0.0
    t4_recall: float = 0.0

    # Training metrics
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "avg_reward": self.avg_reward,
            "task_completion_rate": self.task_completion_rate,
            "format_compliance": self.format_compliance,
            "kl_divergence": self.kl_divergence,
            "avg_tool_calls": self.avg_tool_calls,
            "tool_path_diversity": self.tool_path_diversity,
            "avg_answer_length": self.avg_answer_length,
            "pairwise_bleu": self.pairwise_bleu,
            "hallucination_rate": self.hallucination_rate,
            "uncertain_match_accept_rate": self.uncertain_match_accept_rate,
            "t4_false_positive_rate": self.t4_false_positive_rate,
            "t1_accuracy": self.t1_accuracy,
            "t2_accuracy": self.t2_accuracy,
            "t3_accuracy": self.t3_accuracy,
            "t4_recall": self.t4_recall,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
        }


@dataclass
class MonitorConfig:
    """Configuration for training monitor."""

    eval_interval: int = 200  # Evaluate every N steps
    log_interval: int = 10  # Log to console every N steps
    save_interval: int = 200  # Save checkpoint every N steps

    # Hacking detection thresholds
    kl_spike_threshold: float = 3.0  # Alert if KL > 3× recent avg
    tool_cliff_threshold: float = 0.7  # Alert if tool calls drop >30%
    bleu_threshold: float = 0.85  # Alert if pairwise BLEU > 0.85
    answer_inflation_threshold: float = 1.5  # Alert if length grows >50%

    # History buffer size
    history_size: int = 50

    # Output paths
    log_file: Optional[str] = None
    checkpoint_dir: Optional[str] = None


class TrainingMonitor:
    """
    Monitors GRPO/GiGPO training for reward hacking and quality.

    Usage:
        monitor = TrainingMonitor(config)
        for step in training_loop:
            # ... training step ...
            if step % config.eval_interval == 0:
                metrics = monitor.evaluate(model, eval_dataset)
                alerts = monitor.check_hacking(metrics)
                if alerts:
                    handle_alerts(alerts)
    """

    def __init__(
        self,
        config: MonitorConfig,
        reward_fn: Callable[[RolloutTrajectory, TaskMetadata], RewardBreakdown],
    ):
        """
        Initialize monitor.

        Args:
            config: Monitor configuration
            reward_fn: Reward function to use for evaluation
        """
        self.config = config
        self.reward_fn = reward_fn

        # Metrics history
        self.history: deque[EvalMetrics] = deque(maxlen=config.history_size)

        # Alerts history
        self.alerts_history: List[Tuple[int, List[RewardHackingAlert]]] = []

        # Setup logging
        if config.log_file:
            self._setup_file_logging(config.log_file)

    def _setup_file_logging(self, log_file: str) -> None:
        """Setup file logging for metrics."""
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_file)

    def evaluate(
        self,
        trajectories: List[RolloutTrajectory],
        task_metadatas: List[TaskMetadata],
        step: int,
        kl_divergence: float = 0.0,
        loss: float = 0.0,
        learning_rate: float = 0.0,
        gradient_norm: float = 0.0,
    ) -> EvalMetrics:
        """
        Evaluate model on a batch of trajectories.

        Args:
            trajectories: Completed rollout trajectories
            task_metadatas: Corresponding task metadata
            step: Current training step
            kl_divergence: KL divergence from reference model
            loss: Current training loss
            learning_rate: Current learning rate
            gradient_norm: Gradient norm

        Returns:
            Computed evaluation metrics
        """
        metrics = EvalMetrics(step=step, timestamp=time.time())
        metrics.kl_divergence = kl_divergence
        metrics.loss = loss
        metrics.learning_rate = learning_rate
        metrics.gradient_norm = gradient_norm

        if not trajectories:
            return metrics

        # Compute core metrics
        rewards = []
        format_valid_count = 0
        completed_count = 0
        tool_counts = []
        answer_lengths = []
        tool_paths = set()

        # Per-tier tracking
        tier_results = {"T1": [], "T2": [], "T3": [], "T4": []}

        for traj, meta in zip(trajectories, task_metadatas):
            # Compute reward
            reward_breakdown = self.reward_fn(traj, meta)
            rewards.append(reward_breakdown.total)

            # Format compliance
            if reward_breakdown.r_format >= 0.99:
                format_valid_count += 1

            # Task completion
            if traj.terminated and traj.termination_reason == "final_answer":
                completed_count += 1

            # Behavioral
            tool_counts.append(traj.total_tool_calls)
            if traj.final_answer:
                answer_lengths.append(len(traj.final_answer))
            tool_path = tuple(traj.get_tools_called())
            tool_paths.add(tool_path)

            # Per-tier accuracy
            tier = meta.tier
            if tier in tier_results:
                # Success = high reward
                success = reward_breakdown.total >= 0.7
                tier_results[tier].append(success)

        n = len(trajectories)
        metrics.avg_reward = np.mean(rewards)
        metrics.format_compliance = format_valid_count / n
        metrics.task_completion_rate = completed_count / n
        metrics.avg_tool_calls = np.mean(tool_counts) if tool_counts else 0
        metrics.tool_path_diversity = len(tool_paths)
        metrics.avg_answer_length = np.mean(answer_lengths) if answer_lengths else 0

        # Per-tier metrics
        for tier, results in tier_results.items():
            if results:
                accuracy = sum(results) / len(results)
                if tier == "T1":
                    metrics.t1_accuracy = accuracy
                elif tier == "T2":
                    metrics.t2_accuracy = accuracy
                elif tier == "T3":
                    metrics.t3_accuracy = accuracy
                elif tier == "T4":
                    metrics.t4_recall = accuracy

        # Compute pairwise BLEU (sample for efficiency)
        if len(answer_lengths) >= 2:
            metrics.pairwise_bleu = self._compute_pairwise_bleu(
                [t.final_answer or "" for t in trajectories[:20]]
            )

        # Store in history
        self.history.append(metrics)

        # Log to file
        self._log_metrics(metrics)

        return metrics

    def check_hacking(self, current_metrics: EvalMetrics) -> List[RewardHackingAlert]:
        """
        Check for reward hacking patterns.

        Args:
            current_metrics: Current evaluation metrics

        Returns:
            List of hacking alerts (empty if none detected)
        """
        # Convert history to dict format
        recent_metrics = [m.to_dict() for m in list(self.history)[-10:]]
        current_dict = current_metrics.to_dict()

        alerts = detect_reward_hacking(recent_metrics, current_dict)

        if alerts:
            self.alerts_history.append((current_metrics.step, alerts))
            for alert in alerts:
                logger.warning(
                    f"[Step {current_metrics.step}] REWARD HACKING ALERT: "
                    f"{alert.alert_type} - {alert.message} "
                    f"(current={alert.current_value:.3f}, threshold={alert.threshold:.3f})"
                )

        return alerts

    def should_stop_training(self) -> Tuple[bool, str]:
        """
        Check if training should be stopped due to critical alerts.

        Returns:
            (should_stop, reason)
        """
        if not self.alerts_history:
            return False, ""

        # Check recent critical alerts
        recent_alerts = [
            (step, alerts)
            for step, alerts in self.alerts_history[-3:]
        ]

        critical_count = sum(
            1 for _, alerts in recent_alerts
            for a in alerts if a.severity == "critical"
        )

        if critical_count >= 2:
            return True, "Multiple critical reward hacking alerts detected"

        return False, ""

    def get_rollback_checkpoint(self) -> Optional[int]:
        """
        Suggest checkpoint to rollback to if hacking detected.

        Returns step number of last "clean" checkpoint.
        """
        if not self.alerts_history:
            return None

        # Find step before first alert
        first_alert_step = self.alerts_history[0][0]
        clean_step = max(0, first_alert_step - self.config.save_interval)

        return clean_step

    def _compute_pairwise_bleu(self, texts: List[str], sample_size: int = 10) -> float:
        """
        Compute average pairwise BLEU score (simplified).

        High pairwise BLEU indicates mode collapse.
        """
        if len(texts) < 2:
            return 0.0

        # Simple word-overlap based similarity (not true BLEU, but fast)
        scores = []
        import random

        pairs = []
        for i in range(min(sample_size, len(texts))):
            for j in range(i + 1, min(sample_size, len(texts))):
                pairs.append((i, j))

        if len(pairs) > 20:
            pairs = random.sample(pairs, 20)

        for i, j in pairs:
            words_i = set(texts[i].lower().split())
            words_j = set(texts[j].lower().split())

            if not words_i or not words_j:
                continue

            overlap = len(words_i & words_j)
            similarity = 2 * overlap / (len(words_i) + len(words_j))
            scores.append(similarity)

        return np.mean(scores) if scores else 0.0

    def _log_metrics(self, metrics: EvalMetrics) -> None:
        """Log metrics to file and console."""
        # Console logging
        logger.info(
            f"[Step {metrics.step}] "
            f"reward={metrics.avg_reward:.3f} "
            f"completion={metrics.task_completion_rate:.1%} "
            f"format={metrics.format_compliance:.1%} "
            f"KL={metrics.kl_divergence:.4f} "
            f"T1={metrics.t1_accuracy:.1%} "
            f"T2={metrics.t2_accuracy:.1%} "
            f"T3={metrics.t3_accuracy:.1%}"
        )

        # File logging
        if hasattr(self, "log_path"):
            with open(self.log_path, "a") as f:
                f.write(json.dumps(metrics.to_dict()) + "\n")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.history:
            return {"status": "no_data"}

        recent = list(self.history)[-10:]

        return {
            "total_steps": self.history[-1].step,
            "total_alerts": len(self.alerts_history),
            "recent_avg_reward": np.mean([m.avg_reward for m in recent]),
            "recent_avg_completion": np.mean([m.task_completion_rate for m in recent]),
            "recent_avg_format": np.mean([m.format_compliance for m in recent]),
            "best_reward": max(m.avg_reward for m in self.history),
            "best_completion": max(m.task_completion_rate for m in self.history),
            "reward_trend": self._compute_trend([m.avg_reward for m in recent]),
            "completion_trend": self._compute_trend([m.task_completion_rate for m in recent]),
        }

    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction."""
        if len(values) < 3:
            return "insufficient_data"

        first_half = np.mean(values[: len(values) // 2])
        second_half = np.mean(values[len(values) // 2:])

        diff = second_half - first_half
        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        else:
            return "stable"


class WandbMonitor:
    """
    Weights & Biases integration for GRPO training monitoring.

    Logs metrics to W&B for visualization and experiment tracking.
    """

    def __init__(
        self,
        project: str = "nutrimind-grpo",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize W&B monitor.

        Args:
            project: W&B project name
            run_name: Run name (auto-generated if None)
            config: Training config to log
        """
        self.project = project
        self.run_name = run_name
        self.config = config
        self._initialized = False

    def init(self) -> None:
        """Initialize W&B run."""
        try:
            import wandb

            wandb.init(
                project=self.project,
                name=self.run_name,
                config=self.config,
            )
            self._initialized = True
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")

    def log(self, metrics: EvalMetrics) -> None:
        """Log metrics to W&B."""
        if not self._initialized:
            return

        try:
            import wandb

            wandb.log(metrics.to_dict(), step=metrics.step)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def log_alerts(self, step: int, alerts: List[RewardHackingAlert]) -> None:
        """Log hacking alerts to W&B."""
        if not self._initialized or not alerts:
            return

        try:
            import wandb

            for alert in alerts:
                wandb.log(
                    {
                        f"alert/{alert.alert_type}": alert.current_value,
                        "alert/severity": 1 if alert.severity == "critical" else 0,
                    },
                    step=step,
                )
        except Exception as e:
            logger.warning(f"Failed to log alerts to W&B: {e}")

    def finish(self) -> None:
        """Finish W&B run."""
        if not self._initialized:
            return

        try:
            import wandb

            wandb.finish()
        except Exception:
            pass
