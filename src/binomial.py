import numpy as np


def binomial_price(
    S,
    K,
    T,
    r,
    sigma,
    n_steps=100,
    option_type="call",
    american=False,
    q=0.0,
):
    """
    Cox-Ross-Rubinstein binomial tree price.

    Parameters
    ----------
    S : float - spot
    K : float - strike
    T : float - time to expiry (years)
    r : float - risk-free rate
    sigma : float - volatility
    n_steps : int - number of tree steps
    option_type : 'call' or 'put'
    american : bool - if True, allow early exercise
    q : float - continuous dividend yield

    Returns
    -------
    float - option price
    """
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Terminal asset prices
    j = np.arange(n_steps + 1)
    ST = S * (u ** j) * (d ** (n_steps - j))

    # Terminal payoffs
    if option_type == "call":
        values = np.maximum(ST - K, 0.0)
    elif option_type == "put":
        values = np.maximum(K - ST, 0.0)
    else:
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type}")

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(i + 1)
            Si = S * (u ** j) * (d ** (i - j))
            if option_type == "call":
                intrinsic = np.maximum(Si - K, 0.0)
            else:
                intrinsic = np.maximum(K - Si, 0.0)
            values = np.maximum(values, intrinsic)

    return values[0]