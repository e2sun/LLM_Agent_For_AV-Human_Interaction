"""
Mitigation policies for the strategic AV vs. Human repeated game
(Section 3 of LLM_Agent_For_AV_Human_Interaction.ipynb).

Final Model (repeated incomplete-information game, mutual belief update)
------------------------------------------------------------------------
Human (opponent is an AV w.p. alpha, a human playing q* otherwise):
    EU_H(R) = alpha[mu_H v + (1-mu_H)(-c)] + (1-alpha)[q*(-c) + (1-q*)v]
    EU_H(Y) = alpha(-w) + (1-alpha)[q*(-w)]
    r_i = 1[EU_H(R) > EU_H(Y)] = 1[mu_H,i > mu_H*],   mu_H* ~= 0.452

AV (knows the opponent is human via V2V, so no alpha):
    EU_AV(R) = mu_AV(-c) + (1-mu_AV)v
    EU_AV(Y) = mu_AV(-gamma_AV w)
    yields iff mu_AV > mu_AV* = v / (c + v - gamma_AV w) ~= 0.347

Beta-Bernoulli beliefs:
    Human i: mu_H,i = P(AV yields),    a_i += y,      b_i += 1 - y
             prior Beta(10 mu_0, 10(1 - mu_0))
    AV:      mu_AV  = P(human rushes), a_AV += r_hat, b_AV += 1 - r_hat
             prior Beta(1, 3), so mu_AV,0 = 0.25

Standard template (shared by every policy)
------------------------------------------
Each interaction between the single AV and human i runs the same steps:

    1. intent       r_i = 1[mu_H,i > mu_H*]
    2. commitment   u ~ U(0,1), beta_i ~ U(0.1, 0.6), drawn before the AV decides
    3. AV action    (y, sigma) = pi(mu_AV, .)              <- the only per-policy part
    4. eHMI         r_i^eff = r_i (1 - sigma),   kappa_i = r_i^eff 1[u > beta_i]
    5. outcome map  (outcome, r_hat) = resolve(y, r_i^eff, kappa_i)
    6. Beta updates for the human (on y) and the AV (on r_hat)

The AV learns from r_hat, not r: a bluff that gets called counts as a yield.

  Baseline -- AVBaseline : y = 1[mu_AV > mu_AV*]
  Policy A -- AVPolicyA  : eHMI + waiting-time trigger (streak s, signal sigma)
  Policy B -- AVPolicyB  : Intelligent Intersection (I2V / RSU), FCFS
  Policy C -- AVPolicyC  : interaction-aware commitment prediction

All four policies share one `Human` population and one
`simulate_population(policy, ...)` entry point so they can be swapped with a
single string argument and compared on identical scenarios.
"""

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

HUMAN_PRIOR_STRENGTH = 10   # human prior: Beta(10 mu_0, 10(1 - mu_0))
AV_PRIOR = (1, 3)           # AV prior: Beta(1, 3), mu_AV,0 = 0.25

# ── New parameters for the mitigation policies ───────────────────────────────
WAIT_THRESHOLD = 3          # Policy A: W_th, consecutive yields before AV asserts via eHMI
SAFETY_THRESHOLD = 0.5      # Policy C: theta_s, predicted-commitment cutoff for yielding
SIGNAL_NOISE_STD = 0.25     # Policy C: sigma, sensor noise on the commitment signal
BLUFF_PROB_RANGE = (0.1, 0.6)  # beta_i ~ U(0.1, 0.6): P(human backs off if truly challenged)


# ── Payoffs and thresholds ───────────────────────────────────────────────────
def eu_human_rush(mu_h, alpha=alpha, v=v, c=c, q_star=q_star):
    """EU_H(R) = alpha[mu_H v + (1-mu_H)(-c)] + (1-alpha)[q*(-c) + (1-q*)v]"""
    av_branch = mu_h * v + (1 - mu_h) * (-c)
    human_branch = q_star * (-c) + (1 - q_star) * v
    return alpha * av_branch + (1 - alpha) * human_branch


def eu_human_yield(alpha=alpha, w=w, q_star=q_star):
    """EU_H(Y) = alpha(-w) + (1-alpha)[q*(-w)]"""
    return alpha * (-w) + (1 - alpha) * (q_star * (-w))


def eu_av_rush(mu_av, v=v, c=c):
    """EU_AV(R) = mu_AV(-c) + (1-mu_AV)v"""
    return mu_av * (-c) + (1 - mu_av) * v


def eu_av_yield(mu_av, w=w, gamma_av=gamma_av):
    """EU_AV(Y) = mu_AV(-gamma_AV w)"""
    return mu_av * (-gamma_av * w)


def compute_threshold(alpha, v, c, w, q_star):
    """mu_H* = [EU_H(Y) - (1-alpha)(q*(-c) + (1-q*)v) + alpha c] / [alpha(v+c)]"""
    human_branch = q_star * (-c) + (1 - q_star) * v
    numerator = eu_human_yield(alpha, w, q_star) - (1 - alpha) * human_branch + alpha * c
    return numerator / (alpha * (v + c))


def compute_av_threshold(v, c, w, gamma_av):
    """mu_AV* = v / (c + v - gamma_AV w)"""
    return v / (c + v - gamma_av * w)


mu_h_star = compute_threshold(alpha, v, c, w, q_star)
mu_av_star = compute_av_threshold(v, c, w, gamma_av)


# ── Beliefs and agents ───────────────────────────────────────────────────────
class BetaBelief:
    """Beta-Bernoulli belief: mu = a / (a + b); a success adds to a, else to b."""

    def __init__(self, a0, b0):
        self.a = a0
        self.b = b0

    @property
    def mu(self):
        return self.a / (self.a + self.b)

    def _observe(self, success):
        if success:
            self.a += 1
        else:
            self.b += 1


class Human(BetaBelief):
    """Human i with belief mu_H,i = P(AV yields), plus a fixed hidden trait
    beta_i = P(backs off if truly challenged). beta_i only sets ground-truth
    commitment kappa_i; no AV reads it except through Policy C's noisy sensor."""

    def __init__(self, a0, b0, bluff_prob=None, rng=None):
        super().__init__(a0, b0)
        rng = rng or np.random.default_rng()
        self.beta = bluff_prob if bluff_prob is not None else rng.uniform(*BLUFF_PROB_RANGE)

    def decide(self):
        """Intent r_i = 1[EU_H(R) > EU_H(Y)], i.e. rush iff mu_H,i > mu_H*."""
        return eu_human_rush(self.mu) > eu_human_yield()

    def update_belief(self, av_yielded):
        """a_i += y, b_i += 1 - y"""
        self._observe(av_yielded)


# ── AV policies ───────────────────────────────────────────────────────────────
class AVBaseline(BetaBelief):
    """Final Model AV: belief mu_AV = P(human rushes), y = 1[mu_AV > mu_AV*].

    Every policy follows this template and overrides only `decide`, the policy
    pi in y = pi(mu_AV, .). Policy B also overrides `resolve`, because the RSU
    assignment replaces the outcome map.
    """

    def __init__(self, a0, b0):
        super().__init__(a0, b0)

    def would_yield(self):
        """y_tilde = 1[EU_AV(Y) > EU_AV(R)] = 1[mu_AV > mu_AV*]"""
        return self.mu > mu_av_star

    def decide(self, kappa):
        """Return (y, sigma): AV yields?, eHMI signal shown?

        kappa is the human's real-time commitment, available to every policy
        but read only by Policy C."""
        return self.would_yield(), False

    def resolve(self, y, r_eff, kappa):
        return resolve_outcome(y, r_eff, kappa)

    def update_belief(self, human_rushed):
        """a_AV += r_hat, b_AV += 1 - r_hat"""
        self._observe(human_rushed)


class AVPolicyA(AVBaseline):
    """eHMI + waiting-time trigger.

        y_tilde = 1[mu_AV > mu_AV*]
        sigma   = 1[y_tilde = 1 and s >= W_th]
        y       = y_tilde (1 - sigma)
        s <- 0 if sigma = 1;  s + 1 if sigma = 0, y_tilde = 1;  0 if y_tilde = 0

    There is one AV, so the streak s carries over from one human to the next.
    A credible eHMI assertion makes the human yield (r_i^eff = 0), so a
    signal step resolves as success with r_hat = 0.
    """

    def __init__(self, a0, b0, wait_threshold=WAIT_THRESHOLD):
        super().__init__(a0, b0)
        self.wait_threshold = wait_threshold
        self.consecutive_yields = 0  # streak s

    def decide(self, kappa):
        would_yield = self.would_yield()
        sigma = would_yield and self.consecutive_yields >= self.wait_threshold
        if sigma:
            self.consecutive_yields = 0
        elif would_yield:
            self.consecutive_yields += 1
        else:
            self.consecutive_yields = 0
        return would_yield and not sigma, sigma


class IntersectionManager:
    """Roadside Unit (RSU): arbitrates strictly by first-come-first-served,
    non-negotiable arrival order. Neither agent's belief or incentives matter."""

    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def human_has_priority(self):
        """t_AV, t_H ~ U(0,1), drawn fresh each interaction; 1[t_H < t_AV]."""
        t_av = self.rng.random()
        t_h = self.rng.random()
        return t_h < t_av


class AVPolicyB(AVBaseline):
    """Intelligent Intersection (I2V / RSU), first-come-first-served.

        y = pi_B(t_AV, t_H) = 1[t_H < t_AV],   r_hat = y,   outcome = success

    Neither mu_AV nor mu_H enters the decision. Both beliefs are still updated
    (and drift toward 0.5) but are never used.
    """

    def __init__(self, a0, b0, manager=None):
        super().__init__(a0, b0)
        self.manager = manager or IntersectionManager()

    def decide(self, kappa):
        return self.manager.human_has_priority(), False

    def resolve(self, y, r_eff, kappa):
        # The human must follow the RSU's assignment, so exactly one party
        # proceeds: collision and gridlock can't happen.
        return "success", y


class AVPolicyC(AVBaseline):
    """Interaction-aware commitment prediction.

        p_hat = clip(kappa_i + eps, 0, 1),   eps ~ N(0, sigma^2)
        y     = pi_C(kappa_i, eps) = 1[p_hat > theta_s]

    The decision ignores mu_AV: the AV responds to the real-time commitment
    cue (gap acceptance / deceleration / lateral drift), not to its belief
    about the population. mu_AV is still updated.
        P(y=1 | kappa=1) = Phi(theta_s / sigma)     ~= 0.977
        P(y=1 | kappa=0) = 1 - Phi(theta_s / sigma) ~= 0.023
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

    def predict_commitment(self, kappa):
        signal = float(kappa) + self.rng.normal(0, self.noise_std)
        return float(np.clip(signal, 0.0, 1.0))

    def decide(self, kappa):
        predicted = self.predict_commitment(kappa)
        self.last_predicted_commit = predicted
        return predicted > self.safety_threshold, False


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
    """Outcome map shared by the Final Model and Policies A and C.

    Returns (outcome, r_hat), where r_hat is the human action the AV observes:
        y=1, r=0         -> gridlock,      r_hat = 0
        y=0, r=1, k=1    -> collision,     r_hat = 1
        y=0, r=1, k=0    -> bluff called,  r_hat = 0  (counts as success)
        otherwise        -> success,       r_hat = r
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
    av_a0=AV_PRIOR[0],
    av_b0=AV_PRIOR[1],
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
        a0 = mu0 * HUMAN_PRIOR_STRENGTH
        b0 = (1.0 - mu0) * HUMAN_PRIOR_STRENGTH
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
            # Steps 1-2: intent, and the commitment draw 1[u > beta_i], made
            # before the AV decides because Policy C senses it.
            rushed = human.decide()
            holds_firm = rng.random() > human.beta
            sensed_commit = rushed and holds_firm

            # Step 3: the only policy-specific step, (y, sigma) = pi(mu_AV, .)
            av_yields, signaling = av.decide(sensed_commit)

            # Step 4: a credible eHMI assertion makes the human yield.
            rushed_eff = rushed and not signaling
            true_commit = rushed_eff and holds_firm

            # Steps 5-6: outcome map, then Beta updates (the AV learns from r_hat).
            outcome, final_human_rushed = av.resolve(av_yields, rushed_eff, true_commit)
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
