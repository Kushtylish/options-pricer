\# Options Pricer



European and American option pricing via three independent methods —

Black-Scholes, CRR binomial tree, and Monte Carlo — with analytic and

finite-difference Greeks, implied volatility solving, and convergence analysis.



\## Contents



\- \*\*Black-Scholes\*\* — closed-form European pricing with continuous dividend yield

\- \*\*Binomial tree\*\* — Cox-Ross-Rubinstein lattice, supports American exercise

\- \*\*Monte Carlo\*\* — GBM simulation with antithetic variates and control variates

\- \*\*Greeks\*\* — analytic formulas and finite-difference validation

\- \*\*Implied volatility\*\* — Newton-Raphson with brentq fallback

\- \*\*Test suite\*\* — 22 unit tests covering prices, parity, Greeks, convergence



\## The core idea



A correct options pricer is one where three completely different

computational approaches agree. This project demonstrates that agreement

explicitly:



| Method | European call (S=K=100, T=1, r=5%, σ=20%) |

|---|---|

| Black-Scholes (closed form) | 10.4506 |

| Binomial tree (1000 steps) | 10.4505 |

| Monte Carlo (200k paths) | 10.4506 |



The analytic Greeks match finite-difference Greeks to \~1e-6. The implied

volatility solver recovers input σ to \~1e-10.



\## Notebook highlights



The full analysis is in \[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

Three plots tell most of the story:



\*\*Binomial tree converges to Black-Scholes\*\*



!\[Binomial convergence](results/binomial\_convergence.png)



Convergence is non-monotonic — a known CRR property, not a bug.



\*\*Monte Carlo converges as O(1/√N)\*\*



!\[MC convergence](results/mc\_convergence.png)



Error bars are ±2 standard errors. Control variates reduce the standard

error by \~10× at the same path count.



\*\*American puts carry an early-exercise premium\*\*



!\[American premium](results/american\_premium.png)



The premium is largest for in-the-money strikes and vanishes entirely for

American calls on non-dividend-paying stocks (as theory predicts).



\## Project structure



```

options-pricer/

├── src/

│   ├── black\_scholes.py    # Closed-form pricing and analytic Greeks

│   ├── binomial.py         # CRR tree, European and American

│   ├── monte\_carlo.py      # GBM simulation with variance reduction

│   ├── greeks.py           # Finite-difference Greeks + comparison

│   └── implied\_vol.py      # Newton-Raphson + brentq solver

├── tests/                  # 22 unit tests (pytest)

├── notebooks/analysis.ipynb

└── results/                # Generated plots

```



\## Install and run



```bash

pip install -r requirements.txt

pytest -q                    # 22 tests should pass

python main.py               # Quick pricing demo

```



\## Key findings



1\. \*\*Three methods agree to \~1e-4\*\* at practical resolution. This is the

&#x20;  fundamental correctness check — independent implementations converging

&#x20;  to the same number.

2\. \*\*Binomial convergence oscillates\*\* but is O(1/N). By n=1000 steps,

&#x20;  the tree price is within 1e-4 of closed form.

3\. \*\*Control variates are powerful here.\*\* Using the discounted terminal

&#x20;  stock price as a control cuts MC standard error by \~10× at no cost.

4\. \*\*American puts have positive early-exercise premium\*\* for all ITM

&#x20;  strikes; American calls on non-dividend stocks have none.

5\. \*\*Analytic and finite-difference Greeks agree to 1e-6\*\* across moneyness

&#x20;  and maturity. This validates both the closed-form formulas and the

&#x20;  numerical implementation.



\## Limitations and next steps



\- Binomial tree is CRR only; no trinomial or Leisen-Reimer for faster

&#x20; convergence.

\- Monte Carlo handles only European options; path-dependent payoffs

&#x20; (Asian, barrier, lookback) would need path simulation and possibly

&#x20; Brownian bridge techniques.

\- Implied vol solver assumes European options; American IV requires

&#x20; inverting a lattice, which is slower.

\- No calibration to a real option chain (would need market data).



Natural extensions:



\- Local volatility and Dupire's formula

\- Heston stochastic volatility via Fourier inversion (Carr-Madan)

\- Longstaff-Schwartz for American Monte Carlo

\- Calibration to a SPY option chain and volatility-surface construction



\## References



\- Black, F. and Scholes, M. (1973). \*The Pricing of Options and Corporate Liabilities.\*

\- Cox, J., Ross, S., and Rubinstein, M. (1979). \*Option Pricing: A Simplified Approach.\*

\- Glasserman, P. (2003). \*Monte Carlo Methods in Financial Engineering.\*

