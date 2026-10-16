from src.black_scholes import bs_price, bs_greeks
from src.binomial import binomial_price
from src.monte_carlo import mc_price
from src.implied_vol import implied_vol


def main():
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

    print(f"Inputs: S={S}, K={K}, T={T}, r={r}, sigma={sigma}\n")

    bs = bs_price(S, K, T, r, sigma, "call")
    crr = binomial_price(S, K, T, r, sigma, n_steps=1000, option_type="call")
    mc, se = mc_price(S, K, T, r, sigma, "call", n_paths=200_000, seed=42)

    print(f"European call price")
    print(f"  Black-Scholes:  {bs:.6f}")
    print(f"  Binomial (1k):  {crr:.6f}")
    print(f"  Monte Carlo:    {mc:.6f}  (std err {se:.6f})")

    print(f"\nAmerican put price")
    print(f"  European:       {binomial_price(S, K, T, r, sigma, n_steps=500, option_type='put', american=False):.6f}")
    print(f"  American:       {binomial_price(S, K, T, r, sigma, n_steps=500, option_type='put', american=True):.6f}")

    print(f"\nGreeks (call)")
    for name, val in bs_greeks(S, K, T, r, sigma, "call").items():
        print(f"  {name:6s}: {val:+.6f}")

    print(f"\nImplied vol round-trip")
    price = bs_price(S, K, 0.5, r, 0.35, "call")
    iv = implied_vol(price, S, K, 0.5, r, "call")
    print(f"  True sigma:      0.350000")
    print(f"  Recovered sigma: {iv:.6f}")


if __name__ == "__main__":
    main()