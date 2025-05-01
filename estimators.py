import numpy as np
from scipy.linalg import qr, inv, cholesky

# ——— Helpers ——————————————————————————————————————————————

def process_matrix(A, *args):
    """
    If A is an ndarray, returns
      matvec(x)=A@x, adjvec(x)=A.T@x, n=A.shape[0], m=A.shape[1].
    If A is a callable, you must pass n (int) and optionally adjvec (callable) in args.
    """
    if isinstance(A, np.ndarray):
        matvec = lambda x: A @ x
        adjvec = lambda x: A.T @ x
        n, m = A.shape
        return matvec, n, adjvec, m
    
    if isinstance(A, tuple) and len(A) == 2 and callable(A[0]) and isinstance(A[1], int):
        matvec = A[0]
        n      = A[1]
        adjvec = None
        return matvec, n, adjvec, A
    
    if isinstance(A, tuple) and len(A) == 3 \
       and callable(A[0]) and isinstance(A[1], int) and callable(A[2]):
        matvec, n, adjvec = A
        return matvec, n, adjvec, A

    if callable(A):
        matvec, n, adjvec, m = A, None, None, None
        for arg in args:
            if isinstance(arg, int) and n is None:
                n = arg
            elif callable(arg) and adjvec is None:
                adjvec = arg
        if n is None:
            raise ValueError("Must specify n when A is a callable.")
        if adjvec is None:
            adjvec = matvec
        return matvec, n, adjvec, m

    raise ValueError("A must be a NumPy array or a callable.")


def cnormc(M: np.ndarray) -> np.ndarray:
    """
    Column-normalize M so each column has unit 2-norm.
    Port of cnormc.m.
    """
    norms = np.linalg.norm(M, axis=0)
    return M / norms


def generate_test_matrix(n, m, default='improved', *args):
    """
    Return (Om, improved, type) of shape (n, m) according to `type`.
    Supports 'improved','cimproved','signs'/'rademacher','steinhaus','phases',
             'gaussian','cgaussian','unif'/'sphere','cunif'/'csphere',
             'orth','corth'.
    """
    typ = default
    for v in args:
        if isinstance(v, str):
            typ = v
            break

    def _normc(X):
        return X / np.linalg.norm(X, axis=0)

    if typ == 'improved':
        X = np.random.randn(n, m)
        return np.sqrt(n) * _normc(X), True, typ

    if typ == 'cimproved':
        X = np.random.randn(n, m) + 1j*np.random.randn(n, m)
        return np.sqrt(n) * _normc(X), True, typ

    # ←–– Add this branch to catch "signs" too
    if typ in ('rademacher', 'signs'):
        # ±1 with equal probability
        Om = np.random.choice([-1.0, 1.0], size=(n, m))
        return Om, False, typ

    if typ in ('steinhaus', 'phases'):
        Om = np.exp(2j * np.pi * np.random.rand(n, m))
        return Om, False, typ

    if typ == 'gaussian':
        return np.random.randn(n, m), False, typ

    if typ == 'cgaussian':
        Om = (1/np.sqrt(2)) * (np.random.randn(n, m) + 1j*np.random.randn(n, m))
        return Om, False, typ

    if typ in ('unif', 'sphere'):
        X = np.random.randn(n, m)
        return np.sqrt(n) * _normc(X), False, typ

    if typ in ('cunif', 'csphere'):
        X = np.random.randn(n, m) + 1j*np.random.randn(n, m)
        return np.sqrt(n) * _normc(X), False, typ

    if typ == 'orth':
        Q, _ = qr(np.random.randn(n, m), mode='economic')
        return np.sqrt(n) * Q, False, typ

    if typ == 'corth':
        Q, _ = qr(np.random.randn(n, m) + 1j*np.random.randn(n, m), mode='economic')
        return np.sqrt(n) * Q, False, typ

    raise ValueError(f'"{typ}" not recognized as matrix type')



def diag_prod(A, B):
    """
    diag_prod(A, B) = sum(conj(A)*B, axis=0),
    i.e. diag(A^* B) returned as 1-D array.
    """
    return np.sum(np.conjugate(A) * B, axis=0)


# ——— XTrace (basic) ——————————————————————————————————————

def xtrace_helper(Om, Z, Q, R, improved):
    """
    Core algebra for XTrace-style estimators.
    Robust to rank-deficient R: uses pinv when inv fails.
    """
    n, m = Om.shape

    W = Q.conj().T @ Om

    # --- robust inverse: fall back to pseudoinverse on LinAlgError -------
    try:
        RinvT = np.linalg.inv(R).T
    except np.linalg.LinAlgError:
        RinvT = np.linalg.pinv(R).T          # safe for singular R

    S = cnormc(RinvT)                        # column-normalised

    # --- optional “improved” scaling -------------------------------------
    if improved:
        normW2 = np.linalg.norm(W, axis=0) ** 2
        normS  = np.linalg.norm(S, axis=0)
        scale  = (n - m + 1) / (n - normW2 + np.abs(diag_prod(S, W) * normS) ** 2)
    else:
        scale = np.ones(m)

    # --- bulk algebra -----------------------------------------------------
    H      = Q.conj().T @ Z
    HW     = H @ W
    T      = Z.conj().T @ Om
    dSW    = diag_prod(S, W)
    dSHS   = diag_prod(S, H @ S)
    dTW    = diag_prod(T, W)
    dWHW   = diag_prod(W, HW)
    dSRmHW = diag_prod(S, R - HW)
    dTmHRS = diag_prod(T - H.conj().T @ W, S)

    ests = (
        np.trace(H) * np.ones(m)
        - dSHS
        + ( -dTW + dWHW
            + np.conj(dSW) * dSRmHW
            + np.abs(dSW) ** 2 * dSHS
            + dTmHRS * dSW
          ) * scale
    )

    t   = np.mean(ests)
    err = np.std(ests) / np.sqrt(m)
    return t, err



def xtrace_basic(A, m, *args):
    """
    Unmodified XTrace estimator from Meyer–Musco–Musco–Woodruff (2021).

    Parameters
    ----------
    A :  (n × n) linear operator or callable
    m :  total number of matrix–vector products allowed
    *args :  extra flags forwarded to `process_matrix` / helpers.

    Returns
    -------
    t_hat  :  stochastic estimate of trace(A)
    err    :  diagnostic variance bound returned by `xtrace_helper`
    """
    # ----------------------------------------------------------------
    # 1) bind the mat-vec oracle and basic dimensions
    # ----------------------------------------------------------------
    matvec, n, _, _ = process_matrix(A, *args)

    # ----------------------------------------------------------------
    # 2) build Ω  (m/2 random vectors) and the first sketch  Y = A Ω
    # ----------------------------------------------------------------
    cols                 = max(1, m // 2)        # at least one probe
    Ω, improved, _unused = generate_test_matrix(n, cols, 'improved', *args)
    Y                    = matvec(Ω)             # one mat-vec pass

    # ----------------------------------------------------------------
    # 3) orthonormalise  Q = orth(Y)  via thin QR and do final pass
    # ----------------------------------------------------------------
    Q, R = qr(Y, mode='economic')                # economical QR
    Z    = matvec(Q)                             # second mat-vec pass

    # ----------------------------------------------------------------
    # 4) plug everything into the shared algebraic core
    # ----------------------------------------------------------------
    return xtrace_helper(Ω, Z, Q, R, improved)


def _rand_flip(X: np.ndarray) -> np.ndarray:
    """Multiply each column of X by an independent Rademacher sign (±1)."""
    return X * np.random.choice((-1, 1), size=(1, X.shape[1]))

# ---- main routine ---------------------------------------------------------

def xtrace_krylov(A, m, *args):
    """
    XTrace + short Krylov chain with                   ┌───── matvec budget ─────┐
      • random sign flips for exchangeability          │ 6 · blk  ≈  m           │
      • *independent* Ω in each level                  └─────────────────────────┘

    Budget per blk-column
        Ω0                 0
        Y0  =   A Ω0       1
        AY0 = A² Ω0        1
        Ω1                 0
        Y1  =   A Ω1       1
        Z   =   A Q        3
                         ───
                         6
    """
    matvec, n, adjvec, _ = process_matrix(A, *args)

    blk = max(1, m // 6)                  # 6·blk ≤ m

    # --- independent probe blocks -----------------------------------------
    Ω0, improved0, _ = generate_test_matrix(n, blk, 'improved', *args)
    Ω1, improved1, _ = generate_test_matrix(n, blk, 'improved', *args)

    # --- Krylov construction ----------------------------------------------
    Y0  = matvec(Ω0)                      #  A Ω0
    AY0 = matvec(Y0)                      #  A² Ω0
    Y1  = matvec(Ω1)                      #  A Ω1   (independent)

    # random sign flips  → restores (approx.) exchangeability
    Y0  = _rand_flip(Y0)
    AY0 = _rand_flip(AY0)
    Y1  = _rand_flip(Y1)

    K = np.hstack([Y0, AY0, Y1])          #  (n × 3·blk)

    Q, R_big = qr(K, mode='economic')     #  orthonormal basis
    Z        = matvec(Q)                  #  final pass  (3·blk matvecs)

    # raw test matrix columns seen by the algorithm
    Ω_full = np.hstack([Ω0, AY0, Ω1])

    improved = improved0 | improved1
    return xtrace_helper(Ω_full, Z, Q, R_big, improved)



def xtrace_tol(A, abstol, reltol, *args):
    """
    [t, err, m] = xtrace_tol(A, abstol, reltol, *args)
    Adaptive-tolerance loop wrapping xtrace.
    """
    matvec, n, adjvec, _ = process_matrix(A, *args)
    Y  = np.zeros((n, 0))
    Om = np.zeros((n, 0))
    err = np.inf
    t   = np.inf if reltol != 0 else 0

    while err >= abstol + reltol * abs(t):
        NewOm, improved, _ = generate_test_matrix(n, max(2, Y.shape[1]), 'improved', *args)
        Y  = np.hstack([Y,  matvec(NewOm)])
        Om = np.hstack([Om, NewOm])
        Q, R = qr(Y, mode='economic')
        # build Z incrementally
        nz = Om.shape[1] - NewOm.shape[1]
        Z  = matvec(Q[:, nz:]) if nz > 0 else matvec(Q)
        t, err = xtrace_helper(Om, Z, Q, R, improved)

    m_out = 2 * (Z.shape[1])
    return t, err, m_out


# ——— XDiag ——————————————————————————————————————————

def xdiag(A, m, *args):
    """
    d = xdiag(A, m, *args)
    Diagonal estimator via randomized sketches.
    """
    matvec, n, adjvec, _ = process_matrix(A, *args)
    m2 = m // 2

    Om, _, type_ = generate_test_matrix(n, m2, 'signs', *args)
    if type_ != 'signs':
        raise ValueError("XDiag only implemented with random signs")

    Y  = matvec(Om)
    Q, R = qr(Y, mode='economic')
    Z  = adjvec(Q)
    T  = Z.conj().T @ Om
    S  = cnormc(inv(R).T)

    dQZ     = diag_prod(Q.conj().T, Z.conj().T)
    dQSSZ   = diag_prod((Q @ S).conj().T, (Z @ S).conj().T)
    dOmQT   = diag_prod(Om.conj().T, (Q @ T).conj().T)
    dOmY    = diag_prod(Om.conj().T, Y.conj().T)
    inner   = np.diag(diag_prod(S, T))
    dOmQSST = diag_prod(Om.conj().T, (Q @ S @ inner).conj().T)

    d = dQZ + (-dQSSZ + dOmY - dOmQT + dOmQSST) / m2
    return d


# ——— XNyström Trace ——————————————————————————————————————

def xnystrace_helper(Y, Om, improved):
    n, m = Y.shape

    nu = np.finfo(float).eps * np.linalg.norm(Y, 'fro') / np.sqrt(n)
    Y  = Y + nu * Om

    Q, R = qr(Y, mode='economic')
    H    = Om.conj().T @ Y
    C    = cholesky((H + H.conj().T)/2, lower=False)
    B    = R @ inv(C)

    if improved:
        QQ, RR = qr(Om, mode='economic')
        WW     = QQ.conj().T @ Om
        SS     = cnormc(inv(RR).T)
        normWW2 = np.linalg.norm(WW, axis=0)**2
        normSS  = np.linalg.norm(SS, axis=0)
        scale = (n - m + 1) / (n - normWW2 + np.abs(diag_prod(SS, WW)*normSS)**2)
    else:
        scale = np.ones(m)

    W     = Q.conj().T @ Om
    invH  = inv(H)
    diagH = np.diag(invH)**(-0.5)
    S     = (B / C.T) * diagH[np.newaxis, :]

    dSW = diag_prod(S, W)

    ests = (
      np.linalg.norm(B, 'fro')**2
    - np.linalg.norm(S, axis=0)**2
    + np.abs(dSW)**2 * scale
    - nu * n
    )

    t   = np.mean(ests)
    err = np.std(ests) / np.sqrt(m)
    return t, err


def xnystrace(A, m, *args):
    """
    [t, err] = xnystrace(A, m, *args)
    Nyström-based trace estimator.
    """
    matvec, n, adjvec, _ = process_matrix(A, *args)
    Om, improved, _     = generate_test_matrix(n, m, 'improved', *args)
    Y = matvec(Om)
    return xnystrace_helper(Y, Om, improved)


def xnystrace_tol(A, abstol, reltol, *args):
    """
    [t, err, m] = xnystrace_tol(A, abstol, reltol, *args)
    Adaptive-tolerance Nyström loop.
    """
    matvec, n, adjvec, _ = process_matrix(A, *args)
    Y, Om = np.zeros((n, 0)), np.zeros((n, 0))
    err   = np.inf
    t     = np.inf if reltol != 0 else 0

    while err >= abstol + reltol * abs(t):
        NewOm, improved, _ = generate_test_matrix(n, max(2, Y.shape[1]), 'improved', *args)
        Y  = np.hstack([Y,  matvec(NewOm)])
        Om = np.hstack([Om, NewOm])
        t, err = xnystrace_helper(Y, Om, improved)

    m_out = Y.shape[1]
    return t, err, m_out


def _krylov3(matvec, Om):
    """
    Return K = [Y, AY, A²Y] together with the first two blocks Y, AY.
    """
    Y   = matvec(Om)         #  A Ω
    AY  = matvec(Y)          #  A² Ω
    A2Y = matvec(AY)         #  A³ Ω   (second extra pass)
    return np.hstack([Y, AY, A2Y]), Y, AY


def _estimate_diag(matvec_or_A, n, r):
    """
    Return (d_hat, matvecs_used).

    Uses XDiag (2·⌈r/2⌉ matvecs) for a low-variance diagonal estimate.
    """
    m_diag = 2 * ((r + 1) // 2)           # even number ≥ r

    # --- make a 3-tuple so process_matrix knows n and adjoint ----------
    A_tuple = (matvec_or_A, n, matvec_or_A)

    d_hat  = xdiag(A_tuple, m_diag, 'signs')
    return d_hat, m_diag


def xtrace_diagcv(A, m, *args, diag_frac=0.25):
    """
    Trace estimator   tr(A)  ≈  tr(D̂) + XTrace(A − D̂)

    Parameters
    ----------
    A         : matrix / linear operator / (matvec,n) tuple understood
                by `process_matrix`
    m         : total matvec budget
    diag_frac : fraction of that budget to spend on diagonal probing

    Returns
    -------
    t_hat   :  stochastic estimate of tr(A)
    err     :  diagnostic tuple from the inner XTrace call
    """
    # --------  bind oracle and basic dimensions  --------------------
    matvec_A, n, _, _ = process_matrix(A, *args)

    # -------- 1) estimate the diagonal  ----------------------------
    r_diag      = max(1, int(round(m * diag_frac)))          # ≥ 1 probe
    d_hat, _    = _estimate_diag(matvec_A, n, r_diag)
    trace_D_hat = d_hat.sum()

    # -------- 2) form matvec for  Ã = A − diag(d̂)  ---------------
    def matvec_res(X):
        return matvec_A(X) - d_hat[:, None] * X               # no extra cost

    # -------- 3) spend the *remaining* budget on XTrace  ----------
    m_rem   = max(2, m - r_diag)      # XTrace needs ≥2 matvecs
    t_resid, err = xtrace_basic((matvec_res, n), m_rem, *args)

    return trace_D_hat + t_resid, err


def xtrace_two_stage(A, m, *args):
    """
    Hybrid (two-stage) XTrace estimator.

    Stage 1: run a half-budget vanilla sketch with k = ⌊m/5⌋ probe columns
             → uses 3 k matvecs   [A Ω,  A Q₀]

    Stage 2: add a short power chain  [A Q₀,  A² Q₀]
             → uses 2 k matvecs

    Final orthogonalisation + one pass  A Q  (≤  k + k  matvecs)
    keeps the grand total ≤ m.
    """
    matvec, n, adjvec, _ = process_matrix(A, *args)

    k = max(1, m // 5)                              # probe-block width
    # ----------------  Stage 1  ------------------------------------------------
    Ω, improved, _ = generate_test_matrix(n, k, 'improved', *args)
    Y      = matvec(Ω)                              #  A Ω          (k)
    K1     = np.hstack([Ω, Y])                      #  (n × 2k)
    Q0, _  = qr(K1, mode='economic')                #  orthonormal
    Z0     = matvec(Q0)                             #  A Q₀         (k)

    # ----------------  Stage 2  ------------------------------------------------
    A2Q0   = matvec(Z0)                             #  A² Q₀        (k)
    K2     = np.hstack([K1, A2Q0])                  #  (n × 3k)

    # full basis after enrichment
    Q, R_big = qr(K2, mode='economic')

    # We already know A Q for the first 2 k cols (stored in Z0);
    # compute it for the new A²Q₀ block only, then concatenate.
    ncols_old = Z0.shape[1]
    Z_new     = matvec(Q[:, ncols_old:])            # matvecs ≤ k
    Z         = np.hstack([Z0, Z_new])              # (n × Q.shape[1])

    # --------------  Trace estimate & error bar  -----------------------------
    return xtrace_helper(K2, Z, Q, R_big, improved)


def _sample_omega(n, k, probs):
    """n×k probe matrix whose columns are ±e_i drawn with Pr(i)=probs[i]."""
    idxs  = np.random.choice(n, size=k, p=probs, replace=True)
    signs = np.random.choice((-1.0, 1.0), size=k)
    Om    = np.zeros((n, k))
    Om[idxs, np.arange(k)] = signs
    return Om


def xtrace_leverage(A, m, *args):
    """
    Leverage-score version of XTrace.

    * spend m_diag = max(2, m//3) matvecs on an XDiag warm-up
      → leverage probabilities
    * use remaining budget on a *standard* XTrace sketch whose
      probes are ±e_i drawn according to those probabilities
    """
    matvec, n, _, _ = process_matrix(A, *args)

    # ---------- 1)  diagonal warm-up  -------------------------------------
    m_diag = max(2, m // 3)                        # ≤ m/3, at least 2
    d_hat, _ = _estimate_diag(matvec, n, m_diag)   # uses exactly m_diag matvecs
    probs = d_hat / d_hat.sum()

    # ---------- 2)  main sketch  ------------------------------------------
    k   = max(1, (m - m_diag) // 2)                # 2k matvecs must fit
    Om  = _sample_omega(n, k, probs)
    Y   = matvec(Om)                               #  A Ω      (k matvecs)
    Q, R_big = qr(Y, mode='economic')              #  |Q| = k cols
    Z   = matvec(Q)                                #  A Q      (k matvecs)

    improved = False                               # Om not “improved” type
    return xtrace_helper(Om, Z, Q, R_big, improved)

def _lambda_max_est(matvec, n):
    v  = np.random.randn(n, 1)
    Av = matvec(v)
    return float(np.sqrt((v.T @ Av) / (v.T @ v)))

# helper: degree-p Chebyshev filter  (cost: p matvecs)
def _chebyshev_filter(matvec_scaled, Om, p):
    if p == 0:
        return Om
    Tm1 = Om                          # T₀
    Tm  = matvec_scaled(Om)           # T₁
    for _ in range(2, p + 1):
        Tn = 2 * matvec_scaled(Tm) - Tm1
        Tm1, Tm = Tm, Tn
    return Tm                          # T_p Ω

# ------------------------------------------------------------
def xtrace_chebyshev(A, m, *args, p=3):
    """
    Chebyshev-filtered XTrace (degree p, default 3, PSD matrices).

    Budget:   1  +  (p + 2)·k   ≤   m
              • 1  matvec  for λ̂
              • p·k          for filter
              • 2·k          for standard XTrace sketch
    """
    matvec, n, _, _ = process_matrix(A, *args)

    # -- (0) quick λ_max estimate -----------------------------------------
    lam_max = _lambda_max_est(matvec, n)
    lam_max = lam_max if lam_max > 0 else 1.0

    # scaled operator  Â = 2A/λ_max − I   → spectrum in [-1,1]
    scale            = 2.0 / lam_max
    matvec_scaled    = lambda X: scale * matvec(X) - X

    # -- choose probe width so budget holds --------------------------------
    k = max(1, (m - 1) // (p + 2))        # ≥1

    # -- build Ω and apply Chebyshev filter --------------------------------
    Ω, _imp, _ = generate_test_matrix(n, k, 'signs', *args)
    Y          = _chebyshev_filter(matvec_scaled, Ω, p)   # p matvecs

    # -- standard two-block XTrace:  Q from Y only  ------------------------
    Q, R_big = qr(Y, mode='economic')     # Q has k columns  (matches Ω)
    Z        = matvec(Q)                  # k matvecs

    return xtrace_helper(Ω, Z, Q, R_big, improved=False)


def _hierarchical_probes(n, k):
    idx    = np.arange(n, dtype=int)[:, None]              # (n,1)
    powers = 1 << np.arange(k, dtype=int)                  # 2^j
    cols   = ((idx // powers) & 1) * 2 - 1                 # ±1 pattern
    flips  = np.random.choice((-1, 1), size=(1, k))
    return cols.astype(float) * flips                      # (n×k)

# ---------------- robust XTrace variant -----------------------------------
def xtrace_hierarchical(A, m, *args):
    """
    Hierarchical-probe XTrace with full-rank safeguard.
    Uses  k = ⌊m/2⌋  probes  →  exactly 2k matvecs ≤ m.
    """
    matvec, n, _, _ = process_matrix(A, *args)
    k = max(1, m // 2)

    Ω = _hierarchical_probes(n, k)
    Y = matvec(Ω)                                          # A Ω

    # ----- insure R is invertible (tiny jitter) ----------------------------
    nu = np.finfo(float).eps * np.linalg.norm(Y, 'fro') / np.sqrt(n)
    Y  = Y + nu * Ω

    Q, R_big = qr(Y, mode='economic')                      # Q has k cols
    Z = matvec(Q)                                          # A Q

    return xtrace_helper(Ω, Z, Q, R_big, improved=False)
