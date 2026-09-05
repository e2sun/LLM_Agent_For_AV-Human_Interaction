"""
Mitigation policies for the strategic AV vs. Human repeated game
(Section 3 of LLM_Agent_For_AV_Human_Interaction.ipynb).

Baseline recap
---------------
The AV holds a Beta belief mu_AV over "the human will rush" and yields
whenever mu_AV crosses a fixed threshold mu_av_star. Because yielding is
always observed as evidence of a "soft" AV, humans learn to rush more,
mu_AV keeps climbing, and the AV converges to yielding ~100% of the time
("bullying"/exploitation).

This module adds three toggleable AV policies that intervene on that loop:

  Policy A  -- AVPolicyA   : eHMI + waiting-time trigger
  Policy B  -- AVPolicyB   : Intelligent Intersection (I2V / RSU) FCFS
  Policy C  -- AVPolicyC   : interaction-aware commitment prediction

All four policies (baseline included) share one `Human` population and one
`simulate_population(policy, ...)` entry point so they can be swapped with a
single string argument and compared on identical scenarios.
"""

from collections import defaultdict

import numpy as np

# ── Shared game parameters (matches Section 3 of the notebook) ──────────────
v = 5
c = 10
w = 3
gamma_av = 0.2
alpha = 0.90
q_star = 0.25

N_HUMANS = 500
N_ROUNDS = 30

# ── New parameters for the mitigation policies ───────────────────────────────
WAIT_THRESHOLD = 3          # Policy A: consecutive yields before AV asserts via eHMI
SAFETY_THRESHOLD = 0.5      # Policy C: predicted-commitment cutoff for yielding
SIGNAL_NOISE_STD = 0.25     # Policy C: sensor noise on the commitment signal
BLUFF_PROB_RANGE = (0.1, 0.6)  # per-human hidden trait: P(backs off if truly challenged)


def compute_threshold(alpha, v, c, w, q_star):
    """Human's belief threshold above which rushing dominates (unchanged from Sec. 3)."""
    human_branch = q_star * (-c) + (1 - q_star) * v
    eu_y = alpha * (-w) + (1 - alpha) * (q_star * (-w))
    numerator = eu_y - (1 - alpha) * human_branch + alpha * c
    return numerator / (alpha * (v + c))


mu_h_star = compute_threshold(alpha, v, c, w, q_star)
mu_av_star = v / (c + v - gamma_av * w)


# ── Human agent ───────────────────────────────────────────────────────────────
class Human:
    """Beta-belief human, plus a fixed hidden 'bluff_prob' trait used only for
    measuring ground-truth commitment (never read by any AV decision rule
    except Policy C's noisy sensor)."""

    def __init__(self, a0, b0, bluff_prob=None, rng=None):
        self.a = a0
        self.b = b0
        rng = rng or np.random.default_rng()
        self.bluff_prob = (
            bluff_prob if bluff_prob is not None else rng.uniform(*BLUFF_PROB_RANGE)
        )

    @property
    def mu(self):
        return self.a / (self.a + self.b)

    def decide(self):
        """Nominal incentive-based intent: rush if dominance gap > 0."""
        eu_r = alpha * (self.mu * v + (1 - self.mu) * (-c)) + (1 - alpha) * (
            q_star * (-c) + (1 - q_star) * v
        )
        eu_y = alpha * (-w) + (1 - alpha) * (q_star * (-w))
        return eu_r > eu_y

    def update_belief(self, av_yielded):
        if av_yielded:
            self.a += 1
        else:
            self.b += 1


# ── AV policies ───────────────────────────────────────────────────────────────
class AVBaseline:
    """Strategic AV from Section 3: yields whenever its belief that the human
    will rush exceeds the fixed threshold mu_av_star. No mitigation."""

    def __init__(self, a0, b0):
        self.a = a0
        self.b = b0

    @property
    def mu(self):
        return self.a / (self.a + self.b)

    def decide(self):
        return self.mu > mu_av_star

    def update_belief(self, human_rushed):
        if human_rushed:
            self.a += 1
        else:
            self.b += 1


class AVPolicyA(AVBaseline):
    """eHMI + waiting-time trigger.

    Decides exactly like the baseline, but tracks consecutive yields. Once
    the streak reaches `wait_threshold`, the AV asserts right-of-way via an
    explicit eHMI signal instead of yielding again, and resets the counter.
    A credible eHMI assertion is assumed to make a rational human back off
    for that round (handled in `simulate_population`).
    """

    def __init__(self, a0, b0, wait_threshold=WAIT_THRESHOLD):
        super().__init__(a0, b0)
        self.wait_threshold = wait_threshold
        self.consecutive_yields = 0
        self.signaling = False  # True on rounds where the AV asserts via eHMI

    def decide(self):
        would_yield = self.mu > mu_av_star
        if would_yield and self.consecutive_yields >= self.wait_threshold:
            self.signaling = True
            self.consecutive_yields = 0
            return False  # assert / reclaim right-of-way
        self.signaling = False
        self.consecutive_yields = self.consecutive_yields + 1 if would_yield else 0
        return would_yield


class IntersectionManager:
    """Roadside Unit (RSU): arbitrates strictly by first-come-first-served,
    non-negotiable arrival order. Neither agent's belief or incentives matter."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def av_has_priority(self):
        av_arrival = self.rng.random()
        human_arrival = self.rng.random()
        return av_arrival < human_arrival


class AVPolicyB(AVBaseline):
    """Defers all right-of-way decisions to the Intelligent Intersection.
    The AV (and, symmetrically, the human) simply complies with the
    infrastructure's FCFS assignment -- priority is non-negotiable."""

    def __init__(self, a0, b0, manager=None):
        super().__init__(a0, b0)
        self.manager = manager or IntersectionManager()

    def decide(self):
        return not self.manager.av_has_priority()  # yields iff it lacks priority


class AVPolicyC(AVBaseline):
    """Interaction-aware prediction & planning.

    Instead of reacting to the slow-moving population-level belief mu_AV,
    the AV reads a noisy, real-time behavioral cue (gap acceptance /
    deceleration / lateral drift, abstracted as `true_commit` + sensor
    noise) and only yields if the predicted commitment crosses a safety
    threshold. This lets it tell a genuinely committed human apart from one
    who is bluffing/exploiting the historical pattern.
    """

    def __init__(
        self,
        a0,
        b0,
        safety_threshold=SAFETY_THRESHOLD,
        noise_std=SIGNAL_NOISE_STD,
        rng=None,
    ):
        super().__init__(a0, b0)
        self.safety_threshold = safety_threshold
        self.noise_std = noise_std
        self.rng = rng or np.random.default_rng()
        self.last_predicted_commit = None

    def predict_commitment(self, true_commit):
        signal = float(true_commit) + self.rng.normal(0, self.noise_std)
        return float(np.clip(signal, 0.0, 1.0))

    def decide(self, true_commit):
        predicted = self.predict_commitment(true_commit)
        self.last_predicted_commit = predicted
        return predicted > self.safety_threshold


POLICY_CLASSES = {
    "baseline": AVBaseline,
    "A": AVPolicyA,
    "B": AVPolicyB,
    "C": AVPolicyC,
}


def _make_av(policy, av_a0, av_b0, rng, wait_threshold, safety_threshold, noise_std):
    if policy == "baseline":
        return AVBaseline(av_a0, av_b0)
    if policy == "A":
        return AVPolicyA(av_a0, av_b0, wait_threshold=wait_threshold)
    if policy == "B":
        return AVPolicyB(av_a0, av_b0, manager=IntersectionManager(rng))
    if policy == "C":
        return AVPolicyC(
            av_a0, av_b0, safety_threshold=safety_threshold, noise_std=noise_std, rng=rng
        )
    raise ValueError(f"Unknown policy: {policy!r}")


def resolve_outcome(av_yields, human_rushed, true_commit):
    """Real-world resolution of one AV/human interaction.

    Returns (outcome, final_human_rushed) where outcome is one of
    'success' (exactly one party proceeds), 'collision' (both proceed and
    the human was genuinely committed), or 'gridlock' (both yield).
    A human who nominally "rushes" but is not truly committed (a bluff)
    backs off for real once the AV holds firm.
    """
    if av_yields and not human_rushed:
        return "gridlock", False
    if (not av_yields) and human_rushed:
        if true_commit:
            return "collision", True
        return "success", False  # bluff called; human backs off
    return "success", human_rushed


def sample_mus(label, n=N_HUMANS, seed=42):
    rng = np.random.default_rng(seed)
    if label == "uniform":
        return rng.uniform(0.05, 0.95, n)
    if label == "normal":
        return np.clip(rng.normal(0.5, 0.15, n), 0.05, 0.95)
    if label == "skeptic":
        return np.clip(rng.normal(0.25, 0.12, n), 0.05, 0.95)
    if label == "trusting":
        return np.clip(rng.normal(0.75, 0.12, n), 0.05, 0.95)
    raise ValueError(f"Unknown scenario label: {label!r}")


SCENARIOS = [
    ("uniform", "Uniform", "#5F92BF"),
    ("normal", "Bell Curve", "#43AA8B"),
    ("skeptic", "Skeptic-Heavy", "#7D1D3F"),
    ("trusting", "Trusting-Heavy", "#EEAA00"),
]

# Fixed categorical order (never re-cycled) for the four policies.
POLICY_LABELS = {
    "baseline": "Baseline (no mitigation)",
    "A": "Policy A: eHMI + wait-time trigger",
    "B": "Policy B: Intelligent Intersection (I2V)",
    "C": "Policy C: Interaction-aware prediction",
}
POLICY_COLORS = {
    "baseline": "#2a78d6",  # blue
    "A": "#1baf7a",  # aqua
    "B": "#eda100",  # yellow
    "C": "#008300",  # green
}


def simulate_population(
    policy,
    human_mus_init,
    n_rounds=N_ROUNDS,
    av_a0=1,
    av_b0=3,
    wait_threshold=WAIT_THRESHOLD,
    safety_threshold=SAFETY_THRESHOLD,
    noise_std=SIGNAL_NOISE_STD,
    seed=42,
):
    """Run one policy over n_rounds x len(human_mus_init) interactions.

    Returns a dict of per-round arrays (length n_rounds) plus the final
    human/AV objects, suitable both for the Section-3-style belief plots and
    for the exploitation/throughput/delay metrics defined in
    `evaluate_policies.py`.
    """
    rng = np.random.default_rng(seed)
    humans = []
    for mu0 in human_mus_init:
        a0 = mu0 * 10
        b0 = (1.0 - mu0) * 10
        humans.append(Human(a0, b0, rng=rng))

    av = _make_av(policy, av_a0, av_b0, rng, wait_threshold, safety_threshold, noise_std)

    human_mu_snapshots = []
    av_mu_history = []
    human_rush_rate = []
    av_yield_rate = []
    exploitation_rate = []
    success_rate = []
    collision_rate = []
    gridlock_rate = []
    wait_streak_lengths = []  # completed wait-streak lengths, pooled across all rounds

    wait_streak = 0

    for _ in range(n_rounds):
        round_mus = []
        round_av_yielded = []
        round_human_rushed = []
        round_exploited = []
        round_outcomes = []

        for human in humans:
            nominal_rushed = human.decide()

            if policy == "C":
                true_commit_pre = nominal_rushed and (rng.random() > human.bluff_prob)
                av_yields = av.decide(true_commit_pre)
            else:
                av_yields = av.decide()

            if policy == "A" and getattr(av, "signaling", False):
                nominal_rushed_eff = False  # rational human yields to a credible eHMI assertion
            else:
                nominal_rushed_eff = nominal_rushed

            if policy == "C":
                true_commit = true_commit_pre
            else:
                true_commit = nominal_rushed_eff and (rng.random() > human.bluff_prob)

            if policy == "B":
                final_human_rushed = av_yields  # RSU-mandated compliance, no contest possible
                outcome = "success"
            else:
                outcome, final_human_rushed = resolve_outcome(
                    av_yields, nominal_rushed_eff, true_commit
                )

            human.update_belief(av_yielded=av_yields)
            av.update_belief(human_rushed=final_human_rushed)

            exploited = av_yields and not true_commit

            if av_yields:
                wait_streak += 1
            else:
                if wait_streak > 0:
                    wait_streak_lengths.append(wait_streak)
                wait_streak = 0

            round_mus.append(human.mu)
            round_av_yielded.append(av_yields)
            round_human_rushed.append(final_human_rushed)
            round_exploited.append(exploited)
            round_outcomes.append(outcome)

        round_outcomes = np.array(round_outcomes)
        human_mu_snapshots.append(np.array(round_mus))
        av_mu_history.append(av.mu)
        human_rush_rate.append(np.mean(round_human_rushed))
        av_yield_rate.append(np.mean(round_av_yielded))
        exploitation_rate.append(np.mean(round_exploited))
        success_rate.append(np.mean(round_outcomes == "success"))
        collision_rate.append(np.mean(round_outcomes == "collision"))
        gridlock_rate.append(np.mean(round_outcomes == "gridlock"))

    if wait_streak > 0:
        wait_streak_lengths.append(wait_streak)

    return {
        "policy": policy,
        "human_mu_snapshots": human_mu_snapshots,
        "av_mu_history": np.array(av_mu_history),
        "human_rush_rate": np.array(human_rush_rate),
        "av_yield_rate": np.array(av_yield_rate),
        "exploitation_rate": np.array(exploitation_rate),
        "success_rate": np.array(success_rate),
        "collision_rate": np.array(collision_rate),
        "gridlock_rate": np.array(gridlock_rate),
        "wait_streak_lengths": np.array(wait_streak_lengths) if wait_streak_lengths else np.array([0.0]),
        "humans": humans,
        "av": av,
    }


def summarize(result, tail=10):
    """Collapse a simulate_population() result into scalar headline metrics,
    averaged over the last `tail` rounds (steady-state behavior)."""
    return {
        "policy": result["policy"],
        "av_yield_rate": float(np.mean(result["av_yield_rate"][-tail:])),
        "exploitation_rate": float(np.mean(result["exploitation_rate"][-tail:])),
        "throughput_rate": float(np.mean(result["success_rate"][-tail:])),
        "collision_rate": float(np.mean(result["collision_rate"][-tail:])),
        "gridlock_rate": float(np.mean(result["gridlock_rate"][-tail:])),
        "avg_wait_streak": float(np.mean(result["wait_streak_lengths"])),
        "max_wait_streak": float(np.max(result["wait_streak_lengths"])),
    }
