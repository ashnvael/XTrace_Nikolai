import numpy as np
from scipy.linalg import qr, cholesky, svd, orth
from numpy.linalg import norm, inv
from scipy.special import gammainc

# ─── Helper functions ─────────────────────────────────────────────────────────

def process_matrix(A, *args):
    """
    If A is an ndarray, returns (matvec, n, adjvec, m).
    If A is a callable, user must supply n (int) and optionally adjvec.
    """
    if isinstance(A, np.ndarray):
        matvec = lambda X: A @ X
        adjvec = lambda X: A.conj().T @ X
        n, m = A.shape
        return matvec, n, adjvec, m

    if callable(A):
        matvec = A
        n = None
        adjvec = None
        m = None
        for arg in args:
            if isinstance(arg, int):
                if n is None:
                    n = arg
                elif m is None:
                    m = arg
            elif callable(arg) and adjvec is None:
                adjvec = arg
        if n is None:
            raise ValueError("Must specify n when A is a function.")
        if adjvec is None:
            adjvec = matvec
        return matvec, n, adjvec, m

    raise ValueError("Input must be a NumPy array or a callable.")


def generate_test_matrix(n, m, default='signs', *args):
    """
    Return (Om, improved, type) of shape (n, m) according to `type`:
    'improved','cimproved','rademacher'/'signs','steinhaus','phases',
    'gaussian','cgaussian','unif'/'sphere','cunif'/'csphere','orth','corth'.
    """
    typ = default
    for v in args:
        if isinstance(v, str):
            typ = v
            break

    def _normc(X):
        norms = norm(X, axis=0)
        return X / norms

    if typ == 'improved':
        X = np.random.randn(n, m)
        return np.sqrt(n) * _normc(X), True, typ
    if typ == 'cimproved':
        X = np.random.randn(n, m) + 1j * np.random.randn(n, m)
        return np.sqrt(n) * _normc(X), True, typ
    if typ in ('rademacher', 'signs'):
        Om = -3 + 2 * np.random.randint(1, 3, size=(n, m))
        return Om, False, typ
    if typ in ('steinhaus', 'csigns', 'phases'):
        Om = np.exp(2j * np.pi * np.random.rand(n, m))
        return Om, False, typ
    if typ == 'gaussian':
        Om = np.random.randn(n, m)
        return Om, False, typ
    if typ == 'cgaussian':
        Om = (1/np.sqrt(2)) * (np.random.randn(n, m) + 1j*np.random.randn(n, m))
        return Om, False, typ
    if typ in ('unif', 'sphere'):
        Om = np.sqrt(n) * _normc(np.random.randn(n, m))
        return Om, False, typ
    if typ in ('cunif', 'csphere'):
        X = np.random.randn(n, m) + 1j*np.random.randn(n, m)
        Om = np.sqrt(n) * _normc(X)
        return Om, False, typ
    if typ == 'orth':
        Om = np.sqrt(n) * orth(np.random.randn(n, m))
        return Om, False, typ
    if typ == 'corth':
        Om = np.sqrt(n) * orth(np.random.randn(n, m) + 1j*np.random.randn(n, m))
        return Om, False, typ

    raise ValueError(f"'{typ}' not recognized as matrix type")


def supfind(i, d):
    """
    Find largest alpha in [0,1] such that gammainc(i/2, alpha*i/2) <= d.
    """
    f = lambda a: gammainc(i/2, a*i/2)
    alphas = np.flip(np.arange(0, 1.01, 0.01))
    for a in alphas:
        if f(a) <= d:
            return float(a)
    return 0.0


# ─── Hutchinson ───────────────────────────────────────────────────────────────

def hutch(A, m, *args):
    matvec, n, adjvec, _ = process_matrix(A, *args)
    Om, _, _ = generate_test_matrix(n, m, 'signs', *args)
    Y = matvec(Om)
    t = np.trace(Om.conj().T @ Y) / m
    return t, None


# ─── Hutch++ ─────────────────────────────────────────────────────────────────

def hutch_plusplus(A, m, *args):
    matvec, n, adjvec, _ = process_matrix(A, *args)
    s = int(np.ceil(m/3))
    g = int(np.floor(m/3))
    S, _, _ = generate_test_matrix(n, s, 'signs', *args)
    G, _, _ = generate_test_matrix(n, g, 'signs', *args)
    Q, _ = qr(matvec(S), mode='economic')
    G = G - Q @ (Q.conj().T @ G)
    t = ( np.trace(Q.conj().T @ matvec(Q))
        + np.trace(G.conj().T @ matvec(G)) / G.shape[1] )
    return t, None


# ─── LRA ──────────────────────────────────────────────────────────────────────

def lra(A, m, *args):
    matvec, n, adjvec, _ = process_matrix(A, *args)
    s = int(np.floor(m/2))
    S, _, _ = generate_test_matrix(n, s, 'signs', *args)
    Q, _ = qr(matvec(S), mode='economic')
    t = np.trace(Q.conj().T @ matvec(Q))
    return t, None


# ─── Nyström++ ────────────────────────────────────────────────────────────────

def product_trace(A, B):
    # Trace(A B) without forming a full matrix product
    return np.sum(A * B.conj().T, axis=None)

def nystrompp(A, m, *args):
    matvec, n, adjvec, _ = process_matrix(A, *args)
    Omega, _, _ = generate_test_matrix(n, int(np.ceil(m/2)), 'signs', *args)
    Psi, _, _   = generate_test_matrix(n, int(np.floor(m/2)), 'signs', *args)
    Y = matvec(Omega)
    Z = matvec(Psi)
    nu = np.sqrt(n) * np.finfo(float).eps * norm(Y)
    Y_nu = Y + nu * Omega
    C = cholesky(Omega.conj().T @ Y_nu, lower=False)
    Bm = Y_nu @ inv(C)
    U, Svals, _ = svd(Bm, full_matrices=False)
    Lambda = np.maximum(0, Svals**2 - nu)
    tr1 = np.sum(Lambda)
    tr2 = 2 * (
        product_trace(Psi.conj().T, Z)
      - product_trace(Psi.conj().T, U @ (np.diag(Lambda) @ (U.conj().T @ Psi)))
    ) / m
    return tr1 + tr2, None


# ─── BKS ──────────────────────────────────────────────────────────────────────

def bks(A, m, *args):
    matvec, n, adjvec, _ = process_matrix(A, *args)
    Om, _, _ = generate_test_matrix(n, m, 'signs', *args)
    Y = matvec(Om)
    d = np.diag(Om.conj().T @ Y) / m
    return d


# ─── Diagonal++ ───────────────────────────────────────────────────────────────

def diag_plusplus(A, m, *args):
    matvec, n, adjvec, _ = process_matrix(A, *args)
    s = int(np.ceil(m/3))
    g = int(np.floor(m/3))
    S, _, _ = generate_test_matrix(n, s, 'signs', *args)
    G, _, _ = generate_test_matrix(n, g, 'signs', *args)
    Q, _ = qr(matvec(S), mode='economic')
    B = matvec(G - Q @ (Q.conj().T @ G))
    term1 = np.diag(Q.conj().T @ matvec(Q))
    term2 = np.sum(G * (B - Q @ (Q.conj().T @ B)), axis=0) / G.shape[1]
    return term1 + term2


# ─── Adaptive HPP ──────────────────────────────────────────────────────────────

def adap_hpp(matrix_size, Afun, epsilon, delta):
    C = 4 * np.log(2/delta) / epsilon**2

    # Low‐rank phase
    y = Afun(np.random.randn(matrix_size, 1))
    q = y / norm(y)
    Qmat = q
    x = Afun(q)
    t = float(q.conj().T @ x)
    c = t**2
    trest1 = t
    iteration = 1
    b = norm(x)**2
    fnc = [2*iteration + C*(c - 2*b)]

    while True:
        y = Afun(np.random.randn(matrix_size, 1))
        qt = y - Qmat @ (Qmat.conj().T @ y)
        if norm(qt) < 1e-10:
            lowrank_matvecs = 2 * iteration
            return trest1, lowrank_matvecs, 0, 0
        qt = qt / norm(qt)
        q = qt - Qmat @ (Qmat.conj().T @ qt)
        q = q / norm(q)
        Qmat = np.hstack([Qmat, q])
        x = Afun(q)
        b += norm(x)**2
        t = float(q.conj().T @ x)
        trest1 += t
        c += 2 * norm(Qmat[:, :iteration].conj().T @ x)**2 + t**2
        iteration += 1
        fnc.append(2*iteration + C*(c - 2*b))
        if iteration > 2 and fnc[-3] < fnc[-2] < fnc[-1]:
            break

    lowrank_matvecs = 2 * iteration

    # Hutchinson phase
    trest2_vals = []
    t2 = 0
    iteration2 = 0

    while True:
        psi = np.random.randn(matrix_size, 1)
        y = psi - Qmat @ (Qmat.conj().T @ psi)
        y = Afun(y)
        y = y - Qmat @ (Qmat.conj().T @ y)
        trest2_vals.append(float(psi.conj().T @ y))
        t2 += float(y.conj().T @ y)
        estFrob = t2 / (iteration2 + 1)
        alpha = supfind(iteration2 + 1, delta)
        M = int(np.ceil(C * estFrob / alpha))
        iteration2 += 1
        if iteration2 > M:
            break

    trest2 = np.mean(trest2_vals)
    total_matvecs = lowrank_matvecs + iteration2
    return trest1 + trest2, total_matvecs, lowrank_matvecs, iteration2