"""a6: response-surface fits of PRIVATE score from leaderboard points (SE ~2.0 each). Advisory; tiny n.
Old family (11 hedge rows, mean mode) in u=1-s, v=g-1:  y = b0 + b1 u + b2 v + b3 v^2 + b4 u v  (5 params, 5 points -> exact, 0 dof)."""
import numpy as np
pts = [  # (s, g, private)  -- 11-row mean family + undamped gain-only points (s=1 is the same family's u=0 edge)
    (1.0, 1.0, 18.6), (1.0, 1.25, 19.0), (1.0, 1.667, 17.2), (0.6, 1.4, 28.2), (0.6, 1.25, 26.0)]
X = np.array([[1, 1 - s, g - 1, (g - 1) ** 2, (1 - s) * (g - 1)] for s, g, _ in pts]); y = np.array([p[2] for p in pts])
b = np.linalg.solve(X, y); print('exact fit b0,b1(u),b2(v),b3(v^2),b4(uv) =', np.round(b, 2))
# parameter SE from point SE 2.0 (exact system): cov = 4 * (X^T X)^-1
se = np.sqrt(np.diag(4 * np.linalg.inv(X.T @ X))); print('SE of coefficients (point SE 2.0):', np.round(se, 2))
f = lambda s, g: np.array([1, 1 - s, g - 1, (g - 1) ** 2, (1 - s) * (g - 1)]) @ b
fse = lambda s, g: np.sqrt(4 * np.array([1, 1 - s, g - 1, (g - 1) ** 2, (1 - s) * (g - 1)]) @ np.linalg.inv(X.T @ X) @ np.array([1, 1 - s, g - 1, (g - 1) ** 2, (1 - s) * (g - 1)]))
for s, g in [(0.6, 1.3), (0.6, 1.5), (0.6, 1.75), (0.5, 1.4), (0.45, 1.5), (0.4, 1.0), (0.4, 1.4)]:
    print(f'pred s={s} g={g}: {f(s, g):.1f} +- {fse(s, g):.1f}')
# without the v^2 term (4 params, 1 dof) as a robustness check
X4 = X[:, [0, 1, 2, 4]]; b4, res, *_ = np.linalg.lstsq(X4, y, rcond=None); print('4-param (no v^2) b =', np.round(b4, 2), 'resid SS', np.round(res, 2))
