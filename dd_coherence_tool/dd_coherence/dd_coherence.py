
"""
DD Cohérence IA — outil strictement descriptif de diagnostic de transitions de phase.

Implémente la Formule "D.D Cohérence IA" (docs/DD_Coherence_Formule_Final.pdf).
Séparation stricte : ce module n'implémente ni DD-R ni Équilibre E.

Sorties : instants {t_k*}, séries Φ_t et F_t, métriques locales associées, paramètres exacts, hash code + données.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Sequence, Dict, Any, Tuple, List
import hashlib
import json
import math
import os
import csv
import gzip
from pathlib import Path
import numpy as np
import pandas as pd


EPS_DEFAULT = 1e-12


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_dir_py(root: str | os.PathLike) -> str:
    """
    Hash stable : concatène les sha256 de tous les .py (ordre lexical) puis re-hash.
    """
    rootp = Path(root)
    py_files = sorted([p for p in rootp.rglob("*.py") if p.is_file()])
    h = hashlib.sha256()
    for p in py_files:
        h.update(p.as_posix().encode("utf-8"))
        h.update(b"\n")
        h.update(sha256_file(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _mad(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Median Absolute Deviation (MAD), non-normalisé.
    """
    med = np.nanmedian(x, axis=axis)
    return np.nanmedian(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)


def _ensure_2d_numeric(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = df.loc[:, list(cols)].to_numpy(dtype=float)
    if X.ndim != 2 or X.shape[1] < 1:
        raise ValueError("X doit être une matrice (N,d) avec d>=1.")
    return X


def _fill_nan(df: pd.DataFrame, cols: Sequence[str], nan_policy: str) -> pd.DataFrame:
    out = df.copy()
    if nan_policy == "zero":
        out.loc[:, list(cols)] = out.loc[:, list(cols)].fillna(0.0)
    elif nan_policy == "ffill":
        out.loc[:, list(cols)] = out.loc[:, list(cols)].ffill().bfill()
    elif nan_policy == "drop":
        out = out.dropna(subset=list(cols))
    elif nan_policy == "none":
        if out.loc[:, list(cols)].isna().any().any():
            raise ValueError("NaN détectés et nan_policy='none'.")
    else:
        raise ValueError("nan_policy doit être parmi: zero, ffill, drop, none.")
    return out


def _quantile_edges(x_ref: np.ndarray, K: int) -> np.ndarray:
    q = np.linspace(0.0, 1.0, K + 1)
    edges = np.quantile(x_ref, q)
    # élargit très légèrement les bords
    edges = edges.astype(float)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    # si non strictement croissant, fallback sur min-max
    if np.unique(edges).size < edges.size:
        mn = float(np.min(x_ref))
        mx = float(np.max(x_ref))
        if mx == mn:
            mx = mn + 1e-9
        edges = np.linspace(mn - 1e-12, mx + 1e-12, K + 1)
    return edges


def _rolling_var(x: np.ndarray, m: int) -> np.ndarray:
    N = x.size
    out = np.full(N, np.nan, dtype=float)
    cs = np.cumsum(np.insert(x, 0, 0.0))
    cs2 = np.cumsum(np.insert(x * x, 0, 0.0))
    for t in range(m - 1, N):
        s = cs[t + 1] - cs[t + 1 - m]
        s2 = cs2[t + 1] - cs2[t + 1 - m]
        mu = s / m
        var = s2 / m - mu * mu
        out[t] = max(var, 0.0)
    return out


def _rolling_autocorr_lag1(x: np.ndarray, m: int, eps: float) -> np.ndarray:
    """
    Corr(x_tau, x_{tau-1}) sur la fenêtre W_t\{t-m+1}, longueur m-1, alignée sur t.
    """
    N = x.size
    out = np.full(N, np.nan, dtype=float)
    if m < 3:
        return out
    lag = x[:-1]
    lead = x[1:]
    cross = lag * lead
    mp = m - 1  # longueur pairs
    # prefix sums
    pre_lag = np.cumsum(np.insert(lag, 0, 0.0))
    pre_lead = np.cumsum(np.insert(lead, 0, 0.0))
    pre_lag2 = np.cumsum(np.insert(lag * lag, 0, 0.0))
    pre_lead2 = np.cumsum(np.insert(lead * lead, 0, 0.0))
    pre_cross = np.cumsum(np.insert(cross, 0, 0.0))
    # pour t>=m-1, window sur indices [t-m+1 .. t-1] dans lag/lead/cross
    for t in range(m - 1, N):
        a = t - m + 1
        b = t - 1
        # fenêtre dans arrays de longueur N-1
        if b >= (N - 1) or a < 0:
            continue
        sum_lag = pre_lag[b + 1] - pre_lag[a]
        sum_lead = pre_lead[b + 1] - pre_lead[a]
        sum_lag2 = pre_lag2[b + 1] - pre_lag2[a]
        sum_lead2 = pre_lead2[b + 1] - pre_lead2[a]
        sum_cross = pre_cross[b + 1] - pre_cross[a]

        mu_lag = sum_lag / mp
        mu_lead = sum_lead / mp
        var_lag = max(sum_lag2 / mp - mu_lag * mu_lag, 0.0)
        var_lead = max(sum_lead2 / mp - mu_lead * mu_lead, 0.0)
        cov = sum_cross / mp - mu_lag * mu_lead

        denom = math.sqrt(var_lag * var_lead) + eps
        out[t] = cov / denom if denom > 0 else 0.0
    return out


def _rolling_susceptibility(x: np.ndarray, u: np.ndarray, m: int, eps: float) -> np.ndarray:
    """
    | Cov(x,u) / (Var(u)+eps) | sur W_t
    """
    N = x.size
    out = np.full(N, np.nan, dtype=float)

    cs_x = np.cumsum(np.insert(x, 0, 0.0))
    cs_u = np.cumsum(np.insert(u, 0, 0.0))
    cs_xu = np.cumsum(np.insert(x * u, 0, 0.0))
    cs_u2 = np.cumsum(np.insert(u * u, 0, 0.0))

    for t in range(m - 1, N):
        s_x = cs_x[t + 1] - cs_x[t + 1 - m]
        s_u = cs_u[t + 1] - cs_u[t + 1 - m]
        s_xu = cs_xu[t + 1] - cs_xu[t + 1 - m]
        s_u2 = cs_u2[t + 1] - cs_u2[t + 1 - m]

        mu_x = s_x / m
        mu_u = s_u / m
        cov = s_xu / m - mu_x * mu_u
        var_u = s_u2 / m - mu_u * mu_u

        out[t] = abs(cov / (var_u + eps))
    return out


def _rolling_entropy_discrete(bin_idx: np.ndarray, m: int, K: int, eps: float) -> np.ndarray:
    """
    Entropie discrète sur histogramme de classes (K), fenêtre W_t.
    Implémentation incrémentale (O(N)).
    """
    N = bin_idx.size
    out = np.full(N, np.nan, dtype=float)
    if N < m:
        return out

    counts = np.bincount(bin_idx[:m], minlength=K).astype(float)

    def ent_from_counts(c: np.ndarray) -> float:
        p = c / (np.sum(c) + eps)
        return float(-(p * np.log(p + eps)).sum())

    out[m - 1] = ent_from_counts(counts)
    for t in range(m, N):
        out_bin = int(bin_idx[t - m])
        in_bin = int(bin_idx[t])
        counts[out_bin] -= 1.0
        counts[in_bin] += 1.0
        out[t] = ent_from_counts(counts)
    return out


@dataclass(frozen=True)
class DDParams:
    # Fenêtres
    m: int = 30                  # fenêtre W_t pour mesures critiques
    k: int = 10                  # taille de blocs pré/post pour rupture globale
    r: int = 5                   # persistance sur Φ (r instants)
    cooldown: int = 20           # c
    # Entropie
    K: int = 20                  # classes pour entropie
    # Normalisation
    N0: int = 200                # fin référence initiale T0^(0) = [m-1, N0]
    Lref: int = 200              # longueur référence roulante
    a: int = 0                   # délai avant activation référence roulante après transition
    zmax: float = 6.0            # clip z-scores
    eps: float = EPS_DEFAULT
    # Pondérations
    alpha_V: float = 1.0
    alpha_A: float = 1.0
    alpha_chi: float = 1.0
    alpha_H: float = 1.0
    # Agrégation / verrous
    q: float = 0.2               # quantile bas Q_q
    theta: float = 0.75          # seuil sur Φ et sur C pour F_t
    p: float = 0.3               # proportion lock sur F_t
    p_C: float = 0.3             # proportion lock sur Δ_{t,j}
    # seuils de rupture
    kappa: float = 0.10          # μ_post - μ_pre
    kappa_C: float = 0.10        # Δ_{t,j}


@dataclass
class DDRunMeta:
    tool: str
    tool_version: str
    run_utc: str
    timezone: str
    dataset_rows: int
    dataset_cols: int


def compute_dd_measures(
    X: np.ndarray,
    u: Optional[np.ndarray],
    params: DDParams,
    ref_slice_for_bins: slice,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Calcule V, A, chi, H pour toutes les composantes.
    Retourne measures dict et artefacts (ex : edges entropie).
    """
    N, d = X.shape
    m = params.m
    eps = params.eps

    V = np.full((N, d), np.nan, dtype=float)
    A = np.full((N, d), np.nan, dtype=float)
    CHI = np.full((N, d), np.nan, dtype=float)
    H = np.full((N, d), np.nan, dtype=float)

    entropy_edges = {}

    for j in range(d):
        xj = X[:, j]
        V[:, j] = _rolling_var(xj, m)
        A[:, j] = _rolling_autocorr_lag1(xj, m, eps)

        if u is None:
            CHI[:, j] = 0.0
        else:
            CHI[:, j] = _rolling_susceptibility(xj, u, m, eps)

        # Entropie: bins dérivées de la référence initiale fixée ex ante
        ref_vals = xj[ref_slice_for_bins]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if ref_vals.size == 0:
            ref_vals = xj[np.isfinite(xj)]
        edges = _quantile_edges(ref_vals, params.K)
        entropy_edges[f"col_{j}"] = edges.tolist()

        # discretisation
        # digitize sur edges internes -> classes 0..K-1
        bin_idx = np.digitize(xj, edges[1:-1], right=False)
        bin_idx = np.clip(bin_idx, 0, params.K - 1).astype(int)
        H[:, j] = _rolling_entropy_discrete(bin_idx, m, params.K, eps)

    measures = {"V": V, "A": A, "CHI": CHI, "H": H}
    artefacts = {"entropy_edges": entropy_edges}
    return measures, artefacts


def _robust_zscore(x: np.ndarray, med: np.ndarray, mad: np.ndarray, eps: float, zmax: float) -> np.ndarray:
    z = (x - med) / (mad + eps)
    return np.clip(z, -zmax, zmax)


def run_dd_coherence(
    df: pd.DataFrame,
    cols: Sequence[str],
    params: DDParams,
    time_col: Optional[str] = None,
    u_col: Optional[str] = None,
    nan_policy: str = "zero",
) -> Dict[str, Any]:
    """
    Exécute DD Cohérence sur un DataFrame.
    """
    df2 = _fill_nan(df, cols, nan_policy=nan_policy)
    if u_col is not None:
        df2 = _fill_nan(df2, [u_col], nan_policy=nan_policy)

    X = _ensure_2d_numeric(df2, cols)
    N, d = X.shape

    if time_col is None:
        t_values = np.arange(N)
        t_label = "t"
    else:
        t_values = pd.to_datetime(df2[time_col], errors="coerce")
        if t_values.isna().any():
            # fallback sur index si timestamps invalides
            t_values = np.arange(N)
            t_label = "t"
        else:
            t_values = t_values.dt.strftime("%Y-%m-%dT%H:%M:%S").to_numpy()
            t_label = time_col

    u = None
    if u_col is not None:
        u = df2[u_col].to_numpy(dtype=float)

    m = params.m
    if N < max(m, 2 * params.k + params.r + 1):
        raise ValueError("Série trop courte pour les paramètres (m, k, r).")

    # Référence initiale indices t in [m-1, N0]
    N0 = min(params.N0, N - 1)
    if N0 < m - 1:
        N0 = m - 1
    ref_initial = slice(m - 1, N0 + 1)

    measures, artefacts = compute_dd_measures(X, u, params, ref_slice_for_bins=ref_initial)
    V, A, CHI, H = measures["V"], measures["A"], measures["CHI"], measures["H"]

    # Pré-calc médiane + MAD sur référence initiale (par composante)
    med0 = {}
    mad0 = {}
    for key, M in measures.items():
        med0[key] = np.nanmedian(M[ref_initial, :], axis=0)
        mad0[key] = _mad(M[ref_initial, :], axis=0)

    # Séries de sortie
    Phi = np.full(N, np.nan, dtype=float)
    Ft = np.full(N, np.nan, dtype=float)
    Cand = np.zeros(N, dtype=bool)

    # matrices utiles
    C = np.full((N, d), np.nan, dtype=float)

    # pour μ_pre/μ_post et Δ : prefix sums
    # on calculera C au fil de l'eau, puis on mettra à jour prefix sums
    prefix_Phi = np.zeros(N + 1, dtype=float)
    prefix_C = np.zeros((N + 1, d), dtype=float)  # somme cumulée par composante
    prefix_valid_C = np.zeros((N + 1, d), dtype=float)  # compte (pour gérer NaN éventuel)

    transitions: List[int] = []
    last_transition: Optional[int] = None

    # helper : référence pour z à l'instant t
    def get_ref_stats(key: str, t: int) -> Tuple[np.ndarray, np.ndarray]:
        if last_transition is None or t < (last_transition + params.a):
            return med0[key], mad0[key]
        # référence roulante
        start = max(m - 1, t - params.Lref + 1)
        win = measures[key][start:t + 1, :]
        med = np.nanmedian(win, axis=0)
        mad = _mad(win, axis=0)
        return med, mad

    # Boucle principale
    for t in range(m - 1, N):
        # z-scores
        medV, madV = get_ref_stats("V", t)
        medA, madA = get_ref_stats("A", t)
        medC, madC = get_ref_stats("CHI", t)
        medH, madH = get_ref_stats("H", t)

        zV = _robust_zscore(V[t, :], medV, madV, params.eps, params.zmax)
        zA = _robust_zscore(A[t, :], medA, madA, params.eps, params.zmax)
        zChi = _robust_zscore(CHI[t, :], medC, madC, params.eps, params.zmax)
        zH = _robust_zscore(H[t, :], medH, madH, params.eps, params.zmax)

        z = (
            params.alpha_V * zV
            + params.alpha_A * zA
            + params.alpha_chi * zChi
            + params.alpha_H * zH
        )

        Ct = _sigmoid(z)
        C[t, :] = Ct

        # Φ_t = quantile bas
        Phi[t] = float(np.nanquantile(Ct, params.q))
        # verrou de proportion
        Ft[t] = float(np.mean(Ct >= params.theta))

        # prefix sums
        prefix_Phi[t + 1] = prefix_Phi[t] + (0.0 if np.isnan(Phi[t]) else Phi[t])

        prefix_C[t + 1, :] = prefix_C[t, :] + np.where(np.isnan(Ct), 0.0, Ct)
        prefix_valid_C[t + 1, :] = prefix_valid_C[t, :] + np.where(np.isnan(Ct), 0.0, 1.0)

        # Cand(t) évaluable seulement si assez d'historique pour r + 2k
        k = params.k
        if t < (m - 1 + 2 * k - 1) or t < (m - 1 + params.r - 1):
            continue

        # condition cooldown
        if last_transition is not None and t < last_transition + params.cooldown:
            continue

        # persistance Φ_s >= theta sur [t-r+1, t]
        r0 = params.r
        if np.any(Phi[t - r0 + 1:t + 1] < params.theta):
            continue
        if Phi[t] < params.theta:
            continue

        # μ_pre / μ_post sur Φ
        # pre = [t-2k+1, t-k] ; post = [t-k+1, t]
        pre_a = t - 2 * k + 1
        pre_b = t - k
        post_a = t - k + 1
        post_b = t

        mu_pre = (prefix_Phi[pre_b + 1] - prefix_Phi[pre_a]) / k
        mu_post = (prefix_Phi[post_b + 1] - prefix_Phi[post_a]) / k
        if (mu_post - mu_pre) < params.kappa:
            continue

        # F_t >= p
        if Ft[t] < params.p:
            continue

        # Δ_{t,j} sur C
        # mean post - mean pre par composante
        sum_pre = prefix_C[pre_b + 1, :] - prefix_C[pre_a, :]
        sum_post = prefix_C[post_b + 1, :] - prefix_C[post_a, :]

        cnt_pre = prefix_valid_C[pre_b + 1, :] - prefix_valid_C[pre_a, :]
        cnt_post = prefix_valid_C[post_b + 1, :] - prefix_valid_C[post_a, :]

        mean_pre = np.divide(sum_pre, np.maximum(cnt_pre, 1.0))
        mean_post = np.divide(sum_post, np.maximum(cnt_post, 1.0))
        delta = mean_post - mean_pre

        if float(np.mean(delta >= params.kappa_C)) < params.p_C:
            continue

        # Cand validé
        Cand[t] = True
        transitions.append(t)
        last_transition = t

    # Snapshots de métriques locales aux transitions
    snapshots = []
    for t in transitions:
        k = params.k
        pre_a = t - 2 * k + 1
        pre_b = t - k
        post_a = t - k + 1
        post_b = t

        mu_pre = float((prefix_Phi[pre_b + 1] - prefix_Phi[pre_a]) / k)
        mu_post = float((prefix_Phi[post_b + 1] - prefix_Phi[post_a]) / k)

        # deltas composantes
        sum_pre = prefix_C[pre_b + 1, :] - prefix_C[pre_a, :]
        sum_post = prefix_C[post_b + 1, :] - prefix_C[post_a, :]
        cnt_pre = prefix_valid_C[pre_b + 1, :] - prefix_valid_C[pre_a, :]
        cnt_post = prefix_valid_C[post_b + 1, :] - prefix_valid_C[post_a, :]
        mean_pre = np.divide(sum_pre, np.maximum(cnt_pre, 1.0))
        mean_post = np.divide(sum_post, np.maximum(cnt_post, 1.0))
        delta = mean_post - mean_pre

        snapshots.append({
            "t_index": int(t),
            "t_value": str(t_values[t]),
            "Phi_t": float(Phi[t]),
            "F_t": float(Ft[t]),
            "mu_pre": mu_pre,
            "mu_post": mu_post,
            "delta_mu": float(mu_post - mu_pre),
            "component": {
                cols[j]: {
                    "C_tj": float(C[t, j]),
                    "delta_tj": float(delta[j]),
                    "V_tj": float(V[t, j]),
                    "A_tj": float(A[t, j]),
                    "CHI_tj": float(CHI[t, j]),
                    "H_tj": float(H[t, j]),
                }
                for j in range(d)
            }
        })

    result = {
        "meta": asdict(DDRunMeta(
            tool="dd_coherence",
            tool_version="0.1.0",
            run_utc=pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            timezone="UTC",
            dataset_rows=int(N),
            dataset_cols=int(d),
        )),
        "input": {
            "time_col": time_col,
            "u_col": u_col,
            "cols": list(cols),
            "nan_policy": nan_policy,
            "t_label": t_label,
        },
        "params": asdict(params),
        "artefacts": artefacts,
        "outputs": {
            "transitions_t_index": transitions,
            "transitions_t_value": [str(t_values[t]) for t in transitions],
            "series": {
                "Phi": Phi.tolist(),
                "F": Ft.tolist(),
                "Cand": Cand.astype(int).tolist(),
                "t": [str(v) for v in t_values],
            },
            "snapshots": snapshots,
        },
    }
    return result


def write_outputs(
    result: Dict[str, Any],
    out_dir: str | os.PathLike,
    input_path: Optional[str] = None,
    code_root: Optional[str] = None,
    write_components_csv_gz: bool = True,
) -> Dict[str, str]:
    """
    Écrit dd_report.json + dd_series.csv et optionnellement dd_components.csv.gz.
    Ajoute hashes code + données dans dd_report.json.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    report_path = outp / "dd_report.json"
    series_path = outp / "dd_series.csv"
    comp_path = outp / "dd_components.csv.gz"

    # hash data + code
    hashes: Dict[str, Any] = {}
    if input_path is not None and Path(input_path).exists():
        hashes["data_sha256"] = sha256_file(input_path)
        hashes["data_path"] = str(Path(input_path).resolve())
    if code_root is not None and Path(code_root).exists():
        hashes["code_sha256"] = sha256_dir_py(code_root)
        hashes["code_root"] = str(Path(code_root).resolve())

    result2 = dict(result)
    result2["hashes"] = hashes

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result2, f, ensure_ascii=False, indent=2)

    # series csv
    series = result["outputs"]["series"]
    rows = zip(series["t"], series["Phi"], series["F"], series["Cand"])
    with open(series_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "Phi", "F", "Cand"])
        for t, phi, ft, cand in rows:
            w.writerow([t, phi, ft, cand])

    # components : extrait C et delta au moment des transitions via snapshots
    if write_components_csv_gz:
        # format "long"
        with gzip.open(comp_path, "wt", encoding="utf-8", newline="") as gf:
            w = csv.writer(gf)
            w.writerow(["t", "component", "C_tj", "delta_tj", "V_tj", "A_tj", "CHI_tj", "H_tj"])
            for snap in result["outputs"]["snapshots"]:
                t = snap["t_value"]
                for comp, rec in snap["component"].items():
                    w.writerow([t, comp, rec["C_tj"], rec["delta_tj"], rec["V_tj"], rec["A_tj"], rec["CHI_tj"], rec["H_tj"]])

    return {
        "dd_report": str(report_path),
        "dd_series": str(series_path),
        "dd_components": str(comp_path) if write_components_csv_gz else "",
    }
