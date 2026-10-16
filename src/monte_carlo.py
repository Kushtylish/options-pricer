import numpy as np


def mc_price(
    S,
    K,
    T,
    r,
    sigma,
    option_type="call",
    n_paths=100_000,
    n_steps=1,
    q=0.0,
    antithetic=True,
    control_variate=True,
    seed=None,
):
    """
    Monte Carlo price for a European option under geometric Brownian motion.

    Parameters
    ----------
    S, K, T, r, sigma : float
        Standard Black-Scholes inputs.
    option_type : 'call' or 'put'
    n_paths : int - number of simulated paths
    n_steps : int - time steps per path (1 = terminal only, exact for European)
    q : float - continuous dividend yield
    antithetic : bool - use antithetic variates
    control_variate : bool - use discounted S_T as control variate
    seed : int or None - for reproducibility

    Returns
    -------
    (price, std_error) : tuple of floats
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if T <= 0:
        price = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return price, 0.0

    dt = T / n_steps

    # Simulate terminal prices. For European options n_steps=1 is exact.
    if n_steps == 1:
        # Direct lognormal simulation (exact)
        Z = rng.standard_normal(n_paths)
        if antithetic:
            Z = np.concatenate([Z, -Z])
        ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    else:
        # Full path simulation (for generality / path-dependent later)
        Z = rng.standard_normal((n_paths, n_steps))
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)
        log_incr = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_incr, axis=1)
        ST = S * np.exp(log_paths[:, -1])

    # Payoff
    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        payoff = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    disc = np.exp(-r * T)

    if control_variate:
        # Control variate: discounted S_T has known mean S * exp(-qT)
        Y = disc * ST  # discounted terminal stock price
        EX = S * np.exp(-q * T)  # true mean of Y under risk-neutral
        cov = np.cov(disc * payoff, Y, ddof=1)
        beta = cov[0, 1] / cov[1, 1]
        adjusted = disc * payoff - beta * (Y - EX)
        price = adjusted.mean()
        std_error = adjusted.std(ddof=1) / np.sqrt(len(adjusted))
    else:
        price = (disc * payoff).mean()
        std_error = (disc * payoff).std(ddof=1) / np.sqrt(len(payoff))

    return price, std_error