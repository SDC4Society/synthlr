"""Generate the LaTeX experiment tables for the JPC logistic paper, for all
eps in {0.5, 1.0, 1.1, 4}.

scipy-free reimplementation of the synthlr formulas; validated against the known
eps=1.1 values. For QMS the dummy size gamma is the EXACT solution of the
privacy inequality when eps<=1 and the closed-form APPROXIMATION when eps>1,
matching the paper. Both the mean-error subtables and the condition tables
auto-highlight cells whose experimental mean error is < 0.1 (lightgray) and
< 0.01 (gray), so the generated tables reproduce the paper including colors.
"""
import math, os
import numpy as np

J = 1024
C = 5 * math.sqrt(J)
nums_records = [1000, 10000, 100000, 1000000]
n_str = ["$10^3$", "$10^4$", "$10^5$", "$10^6$"]
sizes_m_str = ['J/10', 'J', '10J', '100J']
sizes_m = [J//10, J, 10*J, 100*J]
# condition tables follow the notebook convention sizes_m=[J/10, J, 10J, 100J] (J/10=102.4, float)
sizes_m_cond = [J/10, J, 10*J, 100*J]
ARR = "journal/arrays/experiment1"
EPS_TAG = [(0.5, "05"), (1.0, "10"), (1.1, "11"), (4, "4")]

# ---- scipy-free formulas (replicating synthlr) ----
def gamma_min_DMS(eps, m): return m/(math.exp(eps)-1)

def gamma_min_QMS_approx(eps, m):
    if eps > 1:   return 1/(math.exp(eps-1)-1)
    elif eps==1:  return math.sqrt(m)
    else:         return (1/eps - 1)*m

def gamma_min_QMS_exact(eps, m):
    def f(g): return math.log1p(1.0/g) + (m-1)*math.log1p(1.0/(g+m)) - eps
    g = gamma_min_QMS_approx(eps, m)
    hi = g
    while f(hi) > 0: hi *= 2.0
    lo = hi/2.0
    while f(lo) < 0:
        lo /= 2.0
        if lo < 1e-300: break
    # bisection (replaces scipy brentq)
    for _ in range(200):
        mid = 0.5*(lo+hi)
        if f(mid) > 0: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

def gamma_min_QMS(eps, m):
    """Paper convention: exact gamma for eps<=1, approximation for eps>1."""
    return gamma_min_QMS_exact(eps, m) if eps <= 1 else gamma_min_QMS_approx(eps, m)

def phi_QMS(m, npg):
    i = np.arange(0, m-1)              # 0..m-2
    lg = np.frompyfunc(math.lgamma, 1, 1)
    log_coe = math.lgamma(m) + math.log(npg) - (m-1)*math.log(npg+m)
    log_terms = (i-1)*np.log(npg+i) + (m-i-1)*np.log(m-i) \
                - np.array(lg(i+1), dtype=float) - np.array(lg(m-i-1), dtype=float)
    return 1 + np.exp(log_coe + log_terms.astype(float)).sum()

def phi_DMS(m, npg): return (npg + m)/(npg + 1)   # (n+gamma+m)/(n+gamma+1)

def e2(x):
    s = f"{x:.2e}"; return f"${s}$"

# ---- validation against known eps=1.1 values (QMS uses APPROX gamma, as in paper) ----
def validate():
    eps = 1.1
    checks = {  # (m_index, n_index): (DMS phi/m C^2, QMS gamma/(n+g) C)
        (0,0): (2.50e+02, 1.45e+02),
        (1,3): (2.50e+01, 1.54e+00),
        (3,2): (2.50e-01, None),
    }
    ok = True
    for (mi,ni),(dms_phi, qms_bias) in checks.items():
        m, n = sizes_m[mi], nums_records[ni]
        gd = gamma_min_DMS(eps,m)*J
        val = phi_DMS(m,n+gd)/m*C**2
        if abs(val-dms_phi)/dms_phi > 0.02: ok=False; print(f"  MISMATCH DMS phi/mC2 m={m},n={n}: {val:.3e} vs {dms_phi:.3e}")
        if qms_bias is not None:
            gq = gamma_min_QMS_approx(eps,m)*J
            valb = gq/(n+gq)*C
            if abs(valb-qms_bias)/qms_bias > 0.02: ok=False; print(f"  MISMATCH QMS bias m={m},n={n}: {valb:.3e} vs {qms_bias:.3e}")
    # also check QMS phi/m C^2 at J/10,1e3 = 2.71e+02 (cond tables use J/10=102.4 float)
    mc = sizes_m_cond[0]
    gq = gamma_min_QMS_approx(eps, mc)*J
    valp = phi_QMS(mc, nums_records[0]+gq)/mc*C**2
    if abs(valp-2.71e2)/2.71e2 > 0.03: ok=False; print(f"  MISMATCH QMS phi/mC2: {valp:.3e} vs 2.71e+02")
    print("VALIDATION:", "PASS" if ok else "FAIL", f"(C={C:.1f})")
    return ok

# ---- shared threshold-highlight helper (experimental mean error) ----
def cond_highlights(est, tag, dms_cols, qms_cols):
    """tblr cell-coloring lines for a condition table. Data rows are 3..18
    (row = 3 + 4*m_index + n_index); whole synthesizer block colored per (m,n)."""
    dms = np.load(f"{ARR}/res_{est}_DMS_eps{tag}.npy").mean(axis=2)
    qms = np.load(f"{ARR}/res_{est}_QMS_eps{tag}.npy").mean(axis=2)
    spec = ""
    for mi in range(4):
        for ni in range(4):
            r = 3 + 4*mi + ni
            for block, cols in [(dms, dms_cols), (qms, qms_cols)]:
                v = block[mi, ni]
                if v < 0.01:
                    spec += "        cell{%d}{%s} = {bg=gray, fg=white},\n" % (r, cols)
                elif v < 0.1:
                    spec += "        cell{%d}{%s} = {bg=lightgray},\n" % (r, cols)
    return spec

# ---- mean-error subtable (tblr) ----
def mean_table(est, eps, tag):
    dms = np.load(f"{ARR}/res_{est}_DMS_eps{tag}.npy").mean(axis=2)  # [m,n]
    qms = np.load(f"{ARR}/res_{est}_QMS_eps{tag}.npy").mean(axis=2)
    estlab = "Plg" if est=="plugin" else "Adj"
    # color specs: data rows are tblr rows 4..7 (m), cols 2..5 DMS, 6..9 QMS
    light, dark = [], []
    for mi in range(4):
        for ni in range(4):
            for block,base in [(dms,2),(qms,6)]:
                v = block[mi,ni]; r = 4+mi; c = base+ni
                if v < 0.01: dark.append((r,c))
                elif v < 0.1: light.append((r,c))
    spec = ""
    for (r,c) in light: spec += f"        cell{{{r}}}{{{c}}} = {{bg=lightgray}},\n"
    for (r,c) in dark:  spec += f"        cell{{{r}}}{{{c}}} = {{bg=gray, fg=white}},\n"
    def fmt(v): return f"${v:.3g}$"
    rows = ""
    for mi in range(4):
        cells = " & ".join(fmt(dms[mi,ni]) for ni in range(4)) + " & " + " & ".join(fmt(qms[mi,ni]) for ni in range(4))
        rows += f"            ${sizes_m_str[mi]}$ & {cells}\\\\\n"
    return f"""    \\begin{{subtable}}[h]{{\\textwidth}}
        \\subcaption{{{estlab}, $\\epsilon={eps}$}}
        \\begin{{tblr}}{{
            width = \\linewidth,
            colspec = {{c || X[l] X[l] X[l] X[l]|X[l] X[l] X[l] X[l]}},
            cell{{2}}{{2}} = {{r=1, c=8}}{{halign=c}},
            cell{{1}}{{2}} = {{c=4}}{{halign=c}},
            cell{{1}}{{6}} = {{c=4}}{{halign=c}},
{spec}        }}
            & DMS & & & & QMS\\\\\\hline
            & $n$\\\\\\hline
            $m$ & $10^3$ & $10^4$ & $10^5$ & $10^6$ & $10^3$ & $10^4$ & $10^5$ & $10^6$ \\\\\\hline
{rows}        \\end{{tblr}}
    \\end{{subtable}}
"""

# ---- condition tables (QMS gamma: exact for eps<=1, approx for eps>1) ----
def cond_table_plg(eps, tag, label):
    rows = ""
    for mi, m in enumerate(sizes_m_cond):
        rows += f"        ${sizes_m_str[mi]}$\n"
        for ni, n in enumerate(nums_records):
            gd = gamma_min_DMS(eps,m)*J;  pd = phi_DMS(m,n+gd)
            gq = gamma_min_QMS(eps,m)*J; pq = phi_QMS(m,n+gq)
            rows += (f"        & {n_str[ni]} & {e2(pd/m*C**2)} & {e2(n/(n+gd)**2*C**2)} & {e2(gd/(n+gd)*C)} "
                     f"& {e2(pq/m*C**2)} & {e2(n/(n+gq)**2*C**2)} & {e2(gq/(n+gq)*C)} \\\\\n")
        rows += "        \\hline\n"
    cspec = cond_highlights("plugin", tag, "3-5", "6-8")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Plg with DMS and QMS for $\\epsilon={eps}$}}
    \\label{{{label}}}
    \\begin{{tblr}}{{
        width = \\linewidth,
        colspec = {{c|c||X[c]|X[c]|X[c]||X[c]|X[c]|X[c]}},
        cell{{1}}{{1}} = {{c=2}}{{}},
        cell{{1}}{{3}} = {{c=3}}{{halign=c}},
        cell{{1}}{{6}} = {{c=3}}{{halign=c}},
        cell{{3}}{{1}} = {{r=4}}{{valign=m}},
        cell{{7}}{{1}} = {{r=4}}{{valign=m}},
        cell{{11}}{{1}} = {{r=4}}{{valign=m}},
        cell{{15}}{{1}} = {{r=4}}{{valign=m}},
{cspec}    }}
         &  & DMS & & & QMS\\\\\\hline
        $m$ & $n$ & $\\frac{{\\phi}}{{m}}C^2$ & $\\frac{{n}}{{(n+\\gamma)^2}}C^2$ & $\\frac{{\\gamma}}{{n+\\gamma}} C$ & $\\frac{{\\phi}}{{m}}C^2$ & $\\frac{{n}}{{(n+\\gamma)^2}}C^2$ & $\\frac{{\\gamma}}{{n+\\gamma}} C$\\\\\\hline\\hline
{rows}    \\end{{tblr}}
\\end{{table}}
"""

def cond_table_unb(eps, tag, label):
    rows = ""
    for mi, m in enumerate(sizes_m_cond):
        rows += f"        ${sizes_m_str[mi]}$\n"
        for ni, n in enumerate(nums_records):
            gd = gamma_min_DMS(eps,m)*J;  pd = phi_DMS(m,n+gd)
            gq = gamma_min_QMS(eps,m)*J; pq = phi_QMS(m,n+gq)
            rows += (f"        & {n_str[ni]} & {e2(((n+gd)/n)**2*pd/m*C**2)} & {e2(C**2/n)} "
                     f"& {e2(((n+gq)/n)**2*pq/m*C**2)} & {e2(C**2/n)} \\\\\n")
        rows += "        \\hline\n"
    cspec = cond_highlights("unbiased", tag, "3-4", "5-6")
    return f"""\\begin{{table}}[htbp]
    \\centering
    \\caption{{Adj with DMS and QMS for $\\epsilon={eps}$}}
    \\label{{{label}}}
    \\begin{{tblr}}{{
        width = \\linewidth,
        colspec = {{c|c||X[c]|X[c]||X[c]|X[c]}},
        cell{{1}}{{1}} = {{c=2}}{{}},
        cell{{1}}{{3,5}} = {{c=2}}{{halign=c}},
        cell{{3, 7, 11, 15}}{{1}} = {{r=4}}{{valign=m}},
{cspec}    }}
         &  & DMS & & QMS\\\\\\hline
        $m$ & $n$ & $\\left(\\frac{{n+\\gamma}}{{n}}\\right)^2\\frac{{\\phi}}{{m}}C^2$ & $\\frac{{1}}{{n}}C^2$ & $\\left(\\frac{{n+\\gamma}}{{n}}\\right)^2\\frac{{\\phi}}{{m}}C^2$ & $\\frac{{1}}{{n}}C^2$ \\\\\\hline\\hline
{rows}    \\end{{tblr}}
\\end{{table}}
"""

if __name__ == "__main__":
    out = "tables"
    os.makedirs(out, exist_ok=True)
    if not validate():
        raise SystemExit("validation failed; aborting")
    with open(f"{out}/mean_subtables.tex","w") as f:
        for eps, tag in EPS_TAG:
            for est in ["plugin","unbiased"]:
                f.write(mean_table(est, eps, tag)); f.write("\n")
    with open(f"{out}/cond_tables.tex","w") as f:
        for eps, tag in EPS_TAG:
            f.write(cond_table_plg(eps, tag, f"tab:evaluating_conditions_plg_eps{tag}")); f.write("\n")
            f.write(cond_table_unb(eps, tag, f"tab:evaluating_conditions_unb_eps{tag}")); f.write("\n")
    print("wrote mean_subtables.tex and cond_tables.tex (eps in {0.5,1.0,1.1,4})")
    # quick numeric peek: highlighted-cell counts per setting
    for eps, tag in EPS_TAG:
        for est in ["plugin","unbiased"]:
            for syn in ["DMS","QMS"]:
                a=np.load(f"{ARR}/res_{est}_{syn}_eps{tag}.npy").mean(axis=2)
                print(f"eps{tag} {est} {syn}: min={a.min():.4f} (<0.1 cells={int((a<0.1).sum())}, <0.01 cells={int((a<0.01).sum())})")
