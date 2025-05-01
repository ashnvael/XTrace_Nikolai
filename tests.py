import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import qr
from tqdm import tqdm
from existing_estimators import generate_test_matrix

from existing_estimators import (
    hutch,
    lra,
    hutch_plusplus,
    nystrompp,
    adap_hpp,
    bks,
    diag_plusplus,
)

from estimators import (
    xtrace_basic,
    xtrace_krylov,
    xnystrace,
    xnystrace_tol,
    xdiag,
    xtrace_diagcv,
    xtrace_two_stage,
    xtrace_leverage,
    xtrace_chebyshev,
    xtrace_hierarchical
)


def rand_with_evals(evals, complex=False):
    """
    Port of rand_with_evals.m:
    Generates a random symmetric (or Hermitian) matrix with given eigenvalues.
    """
    n = len(evals)
    X = np.random.randn(n, n)
    if complex:
        X = X + 1j*np.random.randn(n, n)
    Q, _ = np.linalg.qr(X)
    A = Q @ np.diag(evals) @ Q.conj().T
    # symmetrize
    return (A + A.conj().T) / 2

def tfim(n, h):
    """
    Port of tfim.m: constructs the transverse-field Ising Hamiltonian (2^n x 2^n sparse).
    """
    N = 1 << n
    rows, cols, data = [], [], []
    def bitget(i, k):
        return (i >> (k-1)) & 1

    for i in range(N):
        # off-diagonals: spin flips
        for j in range(n):
            mask = 1 << j
            rows.append(i)
            cols.append(i ^ mask)
            data.append(-h)
        # diagonal: sum of neighbor interactions + periodic term
        val = bitget(i,1) ^ bitget(i,n)
        for j in range(1, n):
            val += bitget(i, j) ^ bitget(i, j+1)
        rows.append(i)
        cols.append(i)
        data.append(2*val - n)

    return csr_matrix((data, (rows, cols)), shape=(N, N))

def tfim_eigs(n, h):
    """
    Port of tfim_eigs.m: analytic spectrum of the TFIM.
    Returns array of length 2^n of all eigenvalues.
    """
    if n % 2 != 0:
        raise ValueError("tfim_eigs not implemented for odd n")
    # first sector (even parity)
    ks1 = np.arange(-n+1, n, 2) * np.pi / n
    e1 = 2 * np.sqrt(1 + h**2 + 2*h*np.cos(ks1))
    ground1 = -0.5 * e1.sum()
    N = 1 << n
    # build spin‐occupation matrix
    spins = ((np.arange(N)[:,None] & (1 << np.arange(n))) > 0).astype(int)
    parity = spins.sum(axis=1) % 2
    d_even = spins[parity==0] @ e1 + ground1

    # second sector (odd parity)
    ks2 = np.arange(-n//2, n//2) * 2*np.pi / n
    e2 = 2 * np.sqrt(1 + h**2 + 2*h*np.cos(ks2))
    # adjust special modes
    e2[0] = -2*(1+h)
    e2[n//2] = 2*(1-h)
    ground2 = -0.5 * e2.sum()
    d_odd = spins[parity==1] @ e2 + ground2

    return np.concatenate([d_even, d_odd])


# ─── basic_tests ────────────────────────────────────────────────────────────────

def basic_tests():
    np.random.seed(42)
    print("Running basic_tests...")
    n = 1000
    As = [
        rand_with_evals(np.linspace(1,3,n)),
        rand_with_evals((np.arange(1,n+1,dtype=float))**(-2)),
        rand_with_evals(0.9**np.arange(n)),
        rand_with_evals(0.7**np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-3*np.ones(n - n//20)])),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-8*np.ones(n - n//20)]))
    ]
    names = ['flat','poly','slowexp','fastexp','smallstep','bigstep']
    methods = [hutch, lra, hutch_plusplus, nystrompp, xtrace_basic, xnystrace]
    method_names = ['Hutch','LRA','Hutch++','Nyström++','xtrace_basic','NysTrace']
    colors = ["#0072BD","#D95319","#EDB120","#4DBEEE","#7E2F8E","#77AC30"]
    markers = ['o','s','*','<','x','^']

    ms = np.arange(20, 301, 20)
    # num_trials = 1000
    num_trials = 100

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8,6))
        for method, mname, c, mk in zip(methods, method_names, colors, markers):
            errors = []
            for m in tqdm(ms):
                errs = []
                for _ in range(num_trials):
                    t, _ = method(A, m, 'signs')
                    errs.append(abs(t - trace_A))
                errors.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, errors, label=mname, color=c, marker=mk, lw=1.5)
        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'basic_tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/basic_{name}.png')
        plt.close()


# ─── compare_adaptive_hutchpp ────────────────────────────────────────────────────

def compare_adaptive_hutchpp():
    np.random.seed(42)
    print("Running compare_adaptive_hutchpp...")
    n = 1000
    As = [
        rand_with_evals(np.linspace(1,3,n)),
        rand_with_evals((np.arange(1,n+1))**(-2)),
        rand_with_evals(0.7**np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-3*np.ones(n - n//20)]))
    ]
    names = ['flat','poly','fastexp','smallstep']
    ranges = [
        (1e-2,1e0),
        (1e-5,1e0),
        (1e-10,1e0),
        (2e-4,1e0)
    ]

    num_trials = 1000
    num_range = 20

    for A, name, (rmin, rmax) in zip(As, names, ranges):
        trace_A = np.trace(A)
        betas = np.logspace(np.log10(rmin), np.log10(rmax), num_range)
        ahpp_errs = []
        ahpp_ms   = []

        for b in betas:
            errs = []
            ms_run = []
            for _ in range(num_trials):
                t, m_used = adap_hpp(n, lambda x: A @ x, b*trace_A, 0.1)
                errs.append(abs(trace_A - t) / trace_A)
                ms_run.append(m_used)
            ahpp_errs.append(np.mean(errs))
            ahpp_ms.append(np.mean(ms_run))

        # xtrace_basic comparison
        ms_x = np.round(np.linspace(min(ahpp_ms), min(max(ahpp_ms),1000), num_range)).astype(int)
        xt_errs = []
        for m in tqdm(ms_x):
            errs = []
            for _ in range(num_trials):
                t = xtrace_basic(A, m, 'signs')
                errs.append(abs(trace_A - t) / trace_A)
            xt_errs.append(np.mean(errs))

        plt.figure(figsize=(8,6))
        plt.semilogy(ahpp_ms, ahpp_errs, 'd-', label='Adaptive HPP', color="#A2142F")
        plt.semilogy(ahpp_ms, betas, 'k--', label='tolerance')
        plt.semilogy(ms_x, xt_errs, 'x-', label='xtrace_basic', color="#7E2F8E")
        plt.xlabel('m')
        plt.ylabel('Average relative error')
        plt.title(f'compare_adaptive_hutchpp: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/compare_adapt_{name}.png')
        plt.close()


# ─── test_vectors ────────────────────────────────────────────────────────────────

def test_vectors():
    np.random.seed(42)
    print("Running test_vectors...")
    n = 1000
    As = [
        rand_with_evals(np.linspace(1,3,n)),
        rand_with_evals((np.arange(1,n+1,dtype=float))**(-2)),
        rand_with_evals(0.9**np.arange(n,dtype=float)),
        rand_with_evals(0.7**np.arange(n,dtype=float)),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-3*np.ones(n - n//20)])),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-8*np.ones(n - n//20)]))
    ]
    names = ['flat','poly','slowexp','fastexp','smallstep','bigstep']

    method = xtrace_basic
    test_types = ['rademacher','gaussian','unif','improved']
    markers = ['x','v','>','p']
    styles = [':','-','--','-.']
    colors = ["#0072BD","#D95319","#EDB120","#7E2F8E"]
    mvec = np.arange(24, 301, 24)
    num_trials = 1000

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8,6))
        for tt, mk, st, c in zip(test_types, markers, styles, colors):
            errors = []
            for m in tqdm(mvec):
                errs = [abs(method(A, m, tt) - trace_A) for _ in range(num_trials)]
                errors.append(np.mean(errs) / trace_A)
            plt.semilogy(mvec, errors, linestyle=st, marker=mk, color=c, label=tt)
        plt.xlabel('Matrix--vector products m')
        plt.ylabel('Average relative error')
        plt.title(f'test_vectors: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/test_vectors_{name}.png')
        plt.close()


# ─── tfim_energy ────────────────────────────────────────────────────────────────

def tfim_energy():
    np.random.seed(42)
    print("Running tfim_energy...")
    hs = np.logspace(-1, 1, 10)
    betas = np.logspace(-1, 1, 10)
    n = 8    # use n=8 or smaller for tractability in Python
    tol = 1e-4

    Zs = np.zeros((len(hs), len(betas)))
    Z_ests = np.zeros_like(Zs)
    Z_ms   = np.zeros_like(Zs)
    Es = np.zeros_like(Zs)

    for i, h in enumerate(hs):
        H = tfim(n, h)
        N = 1 << n
        for j, beta in enumerate(betas):
            print(f"hs idx {i}, beta idx {j}")
            b = -(1 + h) * n
            H_shift = H - b*eye(N, format='csr')

            # Z trace
            fun = lambda x: expm_multiply(-beta*H_shift, x)
            t, err, m_used = xnystrace_tol(fun, 0, tol, N, 'signs')
            Zs[i,j], Z_ests[i,j], Z_ms[i,j] = t, err, m_used

            # EZ trace
            fun2 = lambda x: expm_multiply(-beta*H_shift, H_shift @ x)
            EZ, err2, m2 = xnystrace_tol(fun2, 0, tol, N, 'signs')
            Es[i,j] = EZ / Zs[i,j] + b

    Hs, Bs = np.meshgrid(betas, hs)
    plt.figure(figsize=(6,5))
    cf = plt.contourf(Hs, Bs, Es/n, levels=np.logspace(-1.2, 1, 12), norm=plt.LogNorm())
    plt.xscale('log')
    plt.yscale('log')
    plt.colorbar(cf)
    plt.xlabel('beta')
    plt.ylabel('h')
    plt.title('TFIM Energy per spin')
    plt.tight_layout()
    plt.savefig('figs/tfim_energy.png')
    plt.close()


# ─── tfim_partition ─────────────────────────────────────────────────────────────

def tfim_partition():
    np.random.seed(42)
    print("Running tfim_partition...")
    n, h, beta = 8, 1.0, 3.0  # smaller n for Python
    H = tfim(n, h)
    d = tfim_eigs(n, h)
    b = -(1 + h) * n
    N = 1 << n
    H_shift = H - b*eye(N, format='csr')

    exact = np.sum(np.exp(-beta*(d - b)))
    ms = np.round(10 * 2**np.arange(0, 4+1/3, 1/3)).astype(int)
    num_trials = 100

    h_errs = []
    xny_errs, xny_ests = [], []
    xt_errs, xt_ests = [], []
    hp_errs = []

    for m in ms:
        errs_h, errs_xny, ests_xny = [], [], []
        errs_xt, ests_xt = [], []
        errs_hp = []
        for _ in range(num_trials):
            # Hutch
            t_h = hutch(lambda x: expm_multiply(-beta*H_shift, x), m, N)
            errs_h.append(abs(t_h - exact)/exact)
            # XNyström
            t_ny, s_ny = xnystrace(lambda x: expm_multiply(-beta*H_shift, x), m, N, 'signs')
            errs_xny.append(abs(t_ny - exact)/exact)
            ests_xny.append(s_ny/exact)
            # xtrace_basic
            t_xt, s_xt = xtrace_basic(lambda x: expm_multiply(-beta*H_shift, x), m, N, 'signs')
            errs_xt.append(abs(t_xt - exact)/exact)
            ests_xt.append(s_xt/exact)
            # Hutch++
            t_hp = hutch_plusplus(lambda x: expm_multiply(-beta*H_shift, x), m, N, 'signs')
            errs_hp.append(abs(t_hp - exact)/exact)

        xny_errs.append(np.mean(errs_xny))
        xny_ests.append(np.mean(ests_xny))
        xt_errs.append(np.mean(errs_xt))
        xt_ests.append(np.mean(ests_xt))
        h_errs.append(np.mean(errs_h))
        hp_errs.append(np.mean(errs_hp))

    # Plot
    plt.figure(figsize=(8,6))
    plt.loglog(ms, h_errs,    'o-', label='Hutch')
    plt.loglog(ms, hp_errs,   '*-', label='Hutch++')
    plt.loglog(ms, xt_errs,   'x-', label='xtrace_basic')
    plt.loglog(ms, xny_errs,  '^-', label='XNysTrace')
    plt.xlabel('m')
    plt.ylabel('Mean relative error')
    plt.legend()
    plt.title('TFIM Partition Errors')
    plt.tight_layout()
    plt.savefig('figs/tfim_partition.png')
    plt.close()


# ─── networks ──────────────────────────────────────────────────────────────────

def networks_test():
    import scipy.io
    print("Running networks_test...")
    # load yeast adjacency matrix (expects yeast.mat in cwd)
    mat = scipy.io.loadmat('yeast.mat')
    A = csr_matrix(mat['Problem']['A'][0,0])
    d = np.linalg.eigvals(A.toarray())
    trials = 1000

    methods = [bks, diag_plusplus, xdiag]
    names = ['BKS','Diag++','XDiag']
    colors = ["#0072BD","#EDB120","#7E2F8E"]
    markers = ['o','*','x']

    ms = np.arange(10, 201, 10)

    for i, (target_func, vec_func, algs) in enumerate([
        (lambda A: np.diag(expm_multiply(A, np.eye(A.shape[0]))),
         lambda A,x: expm_multiply(A, x), [0,1,2]),
        (lambda A: np.diag((A@A@A)/2), 
         lambda A,x: A@(A@(A@x)) / 2,        [0,2])
    ]):
        target = target_func(A)
        plt.figure(figsize=(8,6))
        for idx in algs:
            method = methods[idx]
            errs = []
            for m in tqdm(ms):
                err_trials = []
                for _ in range(trials):
                    d_est = method(lambda x: vec_func(A,x), m, A.shape[0], 'signs')
                    err_trials.append(np.linalg.norm(d_est - target, np.inf))
                errs.append(np.mean(err_trials)/np.linalg.norm(target, np.inf))
            plt.semilogy(ms, errs, marker=markers[idx], color=colors[idx],
                         label=names[idx], lw=1.5)
        plt.xlabel('m')
        plt.ylabel('Avg relative inf-norm error')
        plt.title(f'networks: case {i+1}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/networks_case{i+1}.png')
        plt.close()



def compare_basic_krylov():
    np.random.seed(42)
    print("Running basic vs krylov xtrace...")
    n = 1000
    As = [
        rand_with_evals(np.linspace(1,3,n)),
        rand_with_evals((np.arange(1,n+1,dtype=float))**(-2)),
        rand_with_evals(0.9**np.arange(n)),
        rand_with_evals(0.7**np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-3*np.ones(n - n//20)])),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-8*np.ones(n - n//20)]))
    ]
    names = ['flat','poly','slowexp','fastexp','smallstep','bigstep']
    methods = [xtrace_krylov, xtrace_basic]
    method_names = ['xtrace_krylov', 'xtrace_basic']
    colors = ["#7E2F8E","#77AC30"]
    markers = ['o','x']

    ms = np.arange(20, 301, 20)
    # num_trials = 1000
    num_trials = 100

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8,6))
        for method, mname, c, mk in zip(methods, method_names, colors, markers):
            errors = []
            for m in tqdm(ms):
                errs = []
                for _ in range(num_trials):
                    t, _ = method(A, m, 'signs')
                    errs.append(abs(t - trace_A))
                errors.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, errors, label=mname, color=c, marker=mk, lw=1.5)
        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'basic_tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/basic_{name}.png')
        plt.close()



def compare_basic_diagcv():
    np.random.seed(42)
    print("Running basic vs diag-cv xtrace...")
    n = 1000
    As = [
        rand_with_evals(np.linspace(1,3,n)),
        rand_with_evals((np.arange(1,n+1,dtype=float))**(-2)),
        rand_with_evals(0.9**np.arange(n)),
        rand_with_evals(0.7**np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-3*np.ones(n - n//20)])),
        rand_with_evals(np.concatenate([np.ones(n//20),1e-8*np.ones(n - n//20)]))
    ]
    names = ['flat','poly','slowexp','fastexp','smallstep','bigstep']
    methods       = [xtrace_diagcv, xtrace_basic]
    method_names  = ['xtrace_diagcv', 'xtrace_basic']
    colors, marks = ["#0072BD","#77AC30"], ['o','x']

    ms          = np.arange(20, 301, 20)
    num_trials  = 100           # or 1000 if you have patience

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8,6))
        for meth, mname, c, mk in zip(methods, method_names, colors, marks):
            mean_errs = []
            for m in tqdm(ms):
                errs = []
                for _ in range(num_trials):
                    t_hat, _ = meth(A, m, 'signs')
                    errs.append(abs(t_hat - trace_A))
                mean_errs.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, mean_errs, label=mname,
                         color=c, marker=mk, lw=1.5)
        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'diagcv_tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/diagcv_{name}.png')
        plt.close()


def compare_two_stage():
    """
    Benchmark xtrace_two_stage vs. vanilla xtrace_basic on six spectra.

    Creates PNG figures in ./figs/ named twostage_<matrix_name>.png
    showing the average relative error as a function of the matvec
    budget m (20 … 300).  Uses 100 Monte-Carlo trials per point.
    """
    np.random.seed(42)

    print("Running two-stage vs. basic XTrace …")

    n = 1_000
    As = [
        rand_with_evals(np.linspace(1, 3, n)),                      # flat
        rand_with_evals((np.arange(1, n + 1, dtype=float)) ** -2),  # polynomial
        rand_with_evals(0.9 ** np.arange(n)),                       # slow exp.
        rand_with_evals(0.7 ** np.arange(n)),                       # fast exp.
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-3 * np.ones(n - n // 20)])),  # small step
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-8 * np.ones(n - n // 20)])),  # big step
    ]
    names = ['flat', 'poly', 'slowexp', 'fastexp', 'smallstep', 'bigstep']

    methods       = [xtrace_two_stage, xtrace_basic]
    method_names  = ['xtrace_two_stage', 'xtrace_basic']
    colors, marks = ["#D95319", "#0072BD"], ['o', 'x']

    ms         = np.arange(20, 301, 20)   # matvec budgets
    num_trials = 100                      # raise to 1 000 for smoother curves

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8, 6))

        for meth, mname, c, mk in zip(methods, method_names, colors, marks):
            mean_errs = []
            for m in tqdm(ms, desc=f"{name} – {mname}"):
                errs = []
                for _ in range(num_trials):
                    t_hat, _ = meth(A, m, 'signs')
                    errs.append(abs(t_hat - trace_A))
                mean_errs.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, mean_errs, label=mname,
                         color=c, marker=mk, lw=1.5)

        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'two-stage tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/twostage_{name}.png')
        plt.close()


def compare_leverage():
    np.random.seed(42)
    print("Running leverage-score vs. basic XTrace …")

    n = 1_000
    As = [
        rand_with_evals(np.linspace(1, 3, n)),
        rand_with_evals((np.arange(1, n + 1, dtype=float)) ** -2),
        rand_with_evals(0.9 ** np.arange(n)),
        rand_with_evals(0.7 ** np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-3 * np.ones(n - n // 20)])),
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-8 * np.ones(n - n // 20)])),
    ]
    names = ['flat', 'poly', 'slowexp', 'fastexp', 'smallstep', 'bigstep']

    methods       = [xtrace_leverage, xtrace_basic]
    method_names  = ['xtrace_leverage', 'xtrace_basic']
    colors, marks = ["#EDB120", "#0072BD"], ['o', 'x']

    ms         = np.arange(20, 301, 20)
    num_trials = 100

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8, 6))

        for meth, mname, c, mk in zip(methods, method_names, colors, marks):
            mean_errs = []
            for m in tqdm(ms, desc=f"{name} – {mname}"):
                errs = []
                for _ in range(num_trials):
                    t_hat, _ = meth(A, m, 'signs')
                    errs.append(abs(t_hat - trace_A))
                mean_errs.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, mean_errs, label=mname,
                         color=c, marker=mk, lw=1.5)

        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'leverage-score tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/leverage_{name}.png')
        plt.close()

def compare_chebyshev():
    np.random.seed(42)
    print("Running Chebyshev-filtered vs. basic XTrace …")

    n = 1_000
    As = [
        rand_with_evals(np.linspace(1, 3, n)),
        rand_with_evals((np.arange(1, n + 1, dtype=float)) ** -2),
        rand_with_evals(0.9 ** np.arange(n)),
        rand_with_evals(0.7 ** np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-3 * np.ones(n - n // 20)])),
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-8 * np.ones(n - n // 20)])),
    ]
    names = ['flat', 'poly', 'slowexp', 'fastexp', 'smallstep', 'bigstep']

    methods       = [xtrace_chebyshev, xtrace_basic]
    method_names  = ['xtrace_chebyshev', 'xtrace_basic']
    colors, marks = ["#A2142F", "#0072BD"], ['o', 'x']

    ms         = np.arange(20, 301, 20)
    num_trials = 100

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8, 6))

        for meth, mname, c, mk in zip(methods, method_names, colors, marks):
            mean_errs = []
            for m in tqdm(ms, desc=f"{name} – {mname}"):
                errs = []
                for _ in range(num_trials):
                    t_hat, _ = meth(A, m, 'signs')   # PSD so 'signs' fine
                    errs.append(abs(t_hat - trace_A))
                mean_errs.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, mean_errs, label=mname,
                         color=c, marker=mk, lw=1.5)

        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'Chebyshev-filter tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/cheby_{name}.png')
        plt.close()


def compare_hierarchical():
    np.random.seed(42)
    print("Running hierarchical-probe vs. basic XTrace …")

    n = 1_000
    As = [
        # rand_with_evals(np.linspace(1, 3, n)),
        rand_with_evals((np.arange(1, n + 1, dtype=float)) ** -2),
        rand_with_evals(0.9 ** np.arange(n)),
        rand_with_evals(0.7 ** np.arange(n)),
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-3 * np.ones(n - n // 20)])),
        rand_with_evals(np.concatenate([np.ones(n // 20),
                                        1e-8 * np.ones(n - n // 20)])),
    ]
    names = [
        # 'flat', 
        'poly', 'slowexp', 'fastexp', 'smallstep', 'bigstep']

    methods       = [xtrace_hierarchical, xtrace_basic]
    method_names  = ['xtrace_hierarchical', 'xtrace_basic']
    colors, marks = ["#4DBEEE", "#0072BD"], ['o', 'x']

    ms         = np.arange(20, 301, 20)
    num_trials = 100

    for A, name in zip(As, names):
        trace_A = np.trace(A)
        plt.figure(figsize=(8, 6))

        for meth, mname, c, mk in zip(methods, method_names, colors, marks):
            mean_errs = []
            for m in tqdm(ms, desc=f"{name} – {mname}"):
                errs = []
                for _ in range(num_trials):
                    t_hat, _ = meth(A, m)
                    errs.append(abs(t_hat - trace_A))
                mean_errs.append(np.mean(errs) / trace_A)
            plt.semilogy(ms, mean_errs, label=mname,
                         color=c, marker=mk, lw=1.5)

        plt.xlabel('m (matvecs)')
        plt.ylabel('Average relative error')
        plt.title(f'Hierarchical-probe tests: {name}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'figs/hier_{name}.png')
        plt.close()


if __name__ == '__main__':
    import os
    os.makedirs('figs', exist_ok=True)

    """
    New attempted estimators built upon XTrace
    """
    # compare_basic_krylov()
    # compare_basic_diagcv()
    # compare_two_stage()
    compare_chebyshev()
    compare_hierarchical()
    # compare_leverage()

    """
    Basic tests implemented by XTrace authors
    """
    # basic_tests()
    # compare_adaptive_hutchpp()
    # test_vectors()
    # tfim_energy()
    # tfim_partition()
    # networks_test()

