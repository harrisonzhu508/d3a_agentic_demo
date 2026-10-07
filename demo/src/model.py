"""EVOLVABLE: the NumPyro model.

Contract (checks/contract.md):
- model(X, lo=None, hi=None): when lo/hi are given, add the likelihood of each isolate's log2-MIC interval and
  record it pointwise as numpyro.deterministic("log_lik", ...) with shape (n,).
- simulate(samples, X, key) -> array (draws, n): latent log2 MICs for the rows of X, from posterior samples.
- FEATURE_EFFECTS: name of the per-feature effect site (for the parsimony count), or None.

Baseline: log2 MIC ~ Normal(alpha + X beta, sigma), interval-censored, independent Normal(0, 2) priors on the
effects (skill: censored-mic-regression).
"""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import log_ndtr

FEATURE_EFFECTS = "beta"


def log_interval_prob(lo, hi, mu, sigma):
    """log P(lo < y <= hi) for y ~ Normal(mu, sigma); lo may be -inf and hi may be +inf.

    Every branch is computed on finite placeholder bounds so that jnp.where never multiplies an
    infinite gradient by zero (which gives NaN gradients and a stuck sampler).
    """
    lo_inf, hi_inf = jnp.isinf(lo), jnp.isinf(hi)
    lo_f = jnp.where(lo_inf, hi - 1.0, lo)
    hi_f = jnp.where(hi_inf, lo + 1.0, hi)
    a, b = (lo_f - mu) / sigma, (hi_f - mu) / sigma
    log_upper, log_lower = log_ndtr(b), log_ndtr(a)
    interval = log_upper + jnp.log1p(-jnp.exp(log_lower - log_upper))
    return jnp.where(hi_inf, log_ndtr(-a), jnp.where(lo_inf, log_upper, interval))


def logistic_log_interval_prob(lo, hi, mu, scale):
    """log P(lo < y <= hi) for y ~ Logistic(mu, scale); lo may be -inf and hi may be +inf.

    Heavier tails than the normal (log-survival ~ -|z| instead of -z^2/2), for isolates whose genotype
    does not explain an extreme MIC. log_sigmoid keeps each tail accurate; the middle bin is written
    as log_sigmoid(za) - log_sigmoid(zb) (both arguments positive, so no cancellation). Finite
    placeholders everywhere so jnp.where never lets a gradient multiply an inf by zero.
    """
    lo_f = jnp.where(jnp.isneginf(lo), hi - 1.0, lo)
    hi_f = jnp.where(jnp.isposinf(hi), lo_f + 1.0, hi)
    a, b = jnp.broadcast_arrays(lo_f, hi_f)
    za, zb = (mu - a) / scale, (b - mu) / scale
    log_upper = jax.nn.log_sigmoid(zb)                          # log P(y <= hi)
    log_surv = jax.nn.log_sigmoid(za)                           # log P(y > lo)
    interval = jnp.where(za > 0.0,                              # subtract in the more accurate tail
                         log_surv + jnp.log1p(-jnp.exp(jax.nn.log_sigmoid(-zb) - log_surv)),
                         log_upper + jnp.log1p(-jnp.exp(jax.nn.log_sigmoid(za) - log_upper)))
    return jnp.where(jnp.isposinf(hi), log_surv, jnp.where(jnp.isneginf(lo), log_upper, interval))


def model(X, lo=None, hi=None):
    p = X.shape[1]
    alpha = numpyro.sample("alpha", dist.Normal(-4.0, 3.0))
    # Logistic errors with the same variance as the baseline HalfNormal(2) errors: s = sigma sqrt(3/pi)
    # The error scale of the interval-censored likelihood is the parameter the censored rows identify least
    # well - a right-censored isolate is happy with almost any mu once mu is past the top of the plate, so what
    # pins s down is only the shape of the intervals, and its posterior is long-right-tailed (champion:
    # posterior mean 6.8 with a 95% interval of 5.2-8.7, which is the whole width of a ten-doubling grid). A
    # half-normal prior truncates that tail gently rather than hard: it is the weakly informative default for a
    # scale (Gelman et al., BDA3 ch. 21), and its log density is quadratic in s, so it leaves the bulk of the
    # posterior - where the 354 on-grid isolates do constrain s - untouched while removing the part of the tail
    # the censored rows cannot speak to. The champion's own error *distribution* change (logistic errors) was
    # worth +340 ELPD, so getting the residual right is where the loss still lives: coverage is 82.6% against a
    # nominal 90%.
    # Thirty-fifth experiment, continuing on the one parameter this session has found still worth ELPD. The error
    # scale has now been measured twice against the champion's HalfNormal(2), both times clean (0 divergences, R-hat
    # under 1.009, 13 s) and both times positive at the best keep-ratio this session has seen: a half-Cauchy at 4
    # doublings, half the width of the dilution series, gains +4.55 +/- 2.41; at the champion's own scale of 2 it
    # gains +4.06 +/- 2.28. That the scale matters is a structural fact about this likelihood, not a coincidence: a
    # right-censored row contributes 1 - Phi((hi - mu)/s), flat in mu once mu is past hi + 3s, so the 204 isolates
    # above the top of the plate - 56% of the cross-validated loss - carry information about s and almost none about
    # mu. That the *family* barely matters (4 vs 2 doublings of half-Cauchy differ by 0.5 ELPD out of a 2.4 SE, and
    # 4 vs 2 of half-normal differed by +0.60) says the gain is the Cauchy's tail rather than its location: the
    # posterior mean sits at 6.8 doublings with a 95% interval of 5.2-8.7, and a half-normal's log density is
    # quadratic in s, which truncates that interval from above. The remaining unknown is the one that decides
    # whether this axis has anything left: is the gain the heaviness of the tail or merely the absence of the
    # Gaussian penalty? Gamma(2, 0.5) has mean 4 - the centre of the measured posterior - a mode at 2, and a log
    # density that is linear-plus-log in s, so it decays like exp(-s/2): much *wider* than the half-normal at the
    # posterior mode (density ratio 2.7x at s = 6.8) but genuinely light above 12, where the half-Cauchy leaves the
    # door open. If the gain survives, what the censored rows wanted was room around the mode and the axis is done;
    # if it collapses, they wanted the upper tail, and a proper heavy-tailed family (a scale-mixture t, or an
    # inverse-gamma) is where the remaining density is.
    # Sixty-sixth experiment: the combination, with the prior settings the combination's own diagnostics asked for,
    # and nothing else. An InvGamma(2, 128) residual scale (mode 5.7 doublings, median 7.7, an s^-5 tail, written on
    # the reciprocal since numpyro has no inverse-gamma) with t(4, 0, 8) on the 14 rare columns produced the largest
    # gain of the session twice over - +8.28 at default settings (R-hat 1.0122, zero divergences, ESS 1351) and, when
    # I gave it the sampler that R-hat seemed to call for, +3.28 with R-hat 1.5541, ESS 18 and a 221 s run. Reading
    # those two runs together is the whole of this experiment. A pair of near-flat directions in the posterior is
    # either a ridge, where the chains diffuse and more steps help, or a funnel, where the density pinches and more
    # steps make it worse: the signature of the second is that a tighter step and a mass matrix that mixes all
    # parameters together turn a 1.012 into a 1.55, and that is what happened, so the obstruction is the geometry of
    # the priors meeting, not the number of steps. The two priors are not independent the way two hypotheses on
    # separate axes are. The wide rare-column prior supplies the fit with effects that can move an isolate's mu by
    # more than the plate; the inverse-gamma supplies it with a residual that makes such a move cheap; where both
    # slacken at once, the model has two ways of saying "this isolate is resistant" for the price of one, and the
    # posterior of each depends on how far the other has gone - which is a funnel in (scale, beta_rare), not a ridge.
    # So the fix is in the prior that has the adjustable part. The rare-column tail is cut at 16 doublings rather
    # than 12, which is where the same t(4, 0, 8) prior converged by itself at both residual scales this session
    # (uncut: 7 divergences at the narrow fit, and the cut at 12 costing a fifth of the gain at the wider one; cut at
    # 16 at the old champion: zero divergences, +3.37, R-hat 1.003 at 3000 draws), and the scale keeps the
    # distribution that has measured +6.1 to +6.6 five times with the tightest gates of any change on the table. If
    # the combination still will not converge at the settings the parts converge at, then the two slacks are one
    # degree of freedom expressed twice - which the 84% additivity of the last pair had already hinted at - and the
    # session's conclusion is the single InvGamma, +6.43 +/- 3.56, ratio 1.81, which is as close to the harness's rule
    # as anything has come and is where this loop should stop.
    # Ninety-fourth experiment: the last untried idea in this model, written as a matrix, after twenty experiments
    # on the prior for the columns it concerns. The champion splits the 77 determinants at a carrier count of 15 and
    # gives the 14 rare ones a t(4, 0, 8) with a soft knee at 16 doublings. That choice of cut has been defended as
    # the prevalence below which an allele carries too little information to be trusted, and every measurement of
    # the block since has been a measurement of how far its coefficients may run: knee at 12 costs 3.66 ELPD, at 14
    # it fails on one divergence for -0.42 +/- 0.44, at 16 it is main, at 20 it is main to a hundredth of an ELPD
    # with an SE of 0.38, at 24 it gains +0.49 +/- 0.65, and with no knee the fit reaches the best score the loop has
    # seen and is refused for a single divergence. Width has been tried at 3, 4, 6, 8, 12 and 16; the shape at t(4)
    # and Normal; the penalty as a softplus, a hinge and absent. What has never been varied is the thing the split
    # is actually for, which is not how far a rare coefficient may go but how much a rare column is allowed to say
    # at all. An allele carried by three isolates out of 558 appears in the training folds roughly twice; whatever
    # its fitted coefficient, the fold that holds all three carriers is scored on a prediction no training fold
    # contains any evidence for, and the cross-validated density the harness rewards is being bought with a
    # parameter that cannot generalise by construction. The mechanism reference makes the same point from the bench
    # side: a single isolate carrying qnrA1 or fosA is a sequencing and a phenotype as much as a genotype, and
    # AMRFinderPlus calls at that frequency are the calls a lab confirms before reporting. So: multiply every rare
    # column by the square root of its prevalence, sqrt(k/15), which is 0.45 for a carrier count of three and 0.82
    # at fourteen, and leave its wide prior exactly where the keep put it. The scaling is the standard variance-
    # stabilising choice for a coefficient estimated from k observations - it makes the prior's implied statement
    # about an isolate's MIC the same for a rare allele and a common one, so that t(4, 0, 8) means "eight doublings
    # for a determinant this dataset can resolve" rather than "eight doublings whatever the evidence" - and it is
    # applied to the design matrix alone, so the model, its priors and the likelihood are main's to the character.
    # Under this parameterisation the rare coefficient's fitted value must grow by a factor between 1.2 and 2.2 to
    # move an isolate as much as it does on main, which is the same loosening the knee at 24 bought in the prior,
    # bought instead in the design where the prevalence is a fact about the data rather than a number I chose.
    Xc = jnp.asarray(X, dtype=jnp.float32)
    k = Xc.sum(axis=0)
    rare = k <= 15.0
    Xd = jnp.where(rare, Xc * jnp.sqrt(jnp.clip(k, 1.0, 15.0) / 15.0), Xc)
    beta_common = numpyro.sample("beta_common", dist.StudentT(4.0, jnp.zeros(p), 2.0))
    beta_rare = numpyro.sample("beta_rare", dist.StudentT(4.0, jnp.zeros(p), 8.0))
    numpyro.factor("rare_tail", -jnp.sum(jax.nn.softplus(jnp.abs(beta_rare) - 16.0)))
    beta = numpyro.deterministic("beta", jnp.where(rare, beta_rare, beta_common))
    scale_raw = numpyro.sample("scale_raw", dist.Gamma(2.0, 128.0))
    scale = numpyro.deterministic("scale", 1.0 / scale_raw)
    mu = numpyro.deterministic("mu", alpha + Xd @ beta)
    if lo is not None:
        ll = logistic_log_interval_prob(lo, hi, mu, scale)
        numpyro.factor("censored_lik", ll.sum())
        numpyro.deterministic("log_lik", ll)


def simulate(samples, X, key):
    """Latent log2 MIC draws (draws, n) for the rows of X (logistic errors, matching the likelihood)."""
    beta = samples["beta"]                                    # (draws, p)
    mu = samples["alpha"][:, None] + beta @ jnp.asarray(X).T   # (draws, n)
    u = jax.random.uniform(key, mu.shape, minval=jnp.finfo(jnp.float32).tiny)
    return mu + samples["scale"][:, None] * jnp.log(u / (1.0 - u))
