import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ============================================================
# 1) Configuration
# ============================================================

@dataclass
class ALMConfig:
    # Physical constants / simulation setup
    f: float = 28e9
    theta_i_deg: float = 0.0
    theta_r_deg: float = 60.0
    Rk: float = 100.0
    eta0: float = 377.0
    eps_RM: float = 2e-8
    E0: float = 1.0
    Lx: float = 0.5
    eps_RI: float = 1e-2
    N: int = 60

    # Geometry from the report
    Ly_over_lambda: float = 4.9652
    dy_over_lambda: float = 1.0 / 6.0420

    # Mask angles
    mask_res_deg: float = 0.1

    # ALM hyperparameters from the report
    mu_gamma: float = 2e-12
    beta: float = 0.9
    sigma0: float = 100.0

    # Practical stopping / penalty controls
    tau_sigma: float = 1.05
    sigma_max: float = 1e8
    eps_stop: float = 1e-6
    eta_stop: float = 1e-6
    tighten_factor: float = 0.8
    max_outer_iters: int = 2000

    # Numerical-stability controls (to avoid exploding ALM updates)
    grad_clip_norm: float = 1e6
    momentum_clip_norm: float = 1e2
    x_clip_value: float = 5.0
    lambda_clip_value: float = 1e8

    # device / dtype
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    @property
    def wavelength(self) -> float:
        c = 299792458.0
        return c / self.f

    @property
    def k0(self) -> float:
        return 2.0 * math.pi / self.wavelength

    @property
    def Ly(self) -> float:
        return self.Ly_over_lambda * self.wavelength

    @property
    def dy(self) -> float:
        return self.dy_over_lambda * self.wavelength

    @property
    def alpha_i(self) -> float:
        return math.cos(math.radians(self.theta_i_deg))

    @property
    def alpha_r(self) -> float:
        return math.cos(math.radians(self.theta_r_deg))

    @property
    def alpha_ir(self) -> float:
        return self.alpha_r - self.alpha_i

    @property
    def ax(self) -> float:
        return (abs(self.E0) ** 2) * self.Lx / self.eta0

    @property
    def ci(self) -> float:
        return -2.0 * self.Ly * self.alpha_i / self.dy

    @property
    def eps_RI_tilde(self) -> float:
        # The report uses epsilon_tilde_RI in (10d), but does not define it clearly.
        # From z_n = eta0 * (...) and Re(z_n) <= eps_RI, the consistent choice is eps_RI / eta0.
        return self.eps_RI / self.eta0

    @property
    def theta_mask_deg(self) -> torch.Tensor:
        a1 = torch.arange(-2.0, 2.0 + self.mask_res_deg, self.mask_res_deg)
        a2 = torch.arange(-62.0, -58.0 + self.mask_res_deg, self.mask_res_deg)
        return torch.cat([a1, a2], dim=0)

    @property
    def theta_mask_rad(self) -> torch.Tensor:
        return self.theta_mask_deg * math.pi / 180.0


# ============================================================
# 2) Helper conversions
# ============================================================

def realvec_to_complex_gamma(x: torch.Tensor, N: int) -> torch.Tensor:
    """
    x shape: [2N]
    return gamma shape: [N] complex
    """
    re = x[:N]
    im = x[N:]
    return torch.complex(re, im)


def complex_abs_sq(z: torch.Tensor) -> torch.Tensor:
    return z.real**2 + z.imag**2


# ============================================================
# 3) Electromagnetic model from the report
# ============================================================

class RISSurfaceNetPowerProblem:
    def __init__(self, cfg: ALMConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.dtype = cfg.dtype

        self.theta_mask_rad = cfg.theta_mask_rad.to(device=self.device, dtype=self.dtype)

        # A centered discretization of RIS element positions.
        # This is consistent with 2*Ly ≈ N*dy from the report parameters.
        n = torch.arange(cfg.N, device=self.device, dtype=self.dtype)
        self.y_n = -cfg.Ly + (n + 0.5) * cfg.dy

    def Ps(self, gamma: torch.Tensor) -> torch.Tensor:
        """
        P_s(gamma) = ax * dy * (ci + alpha_r * gamma^H gamma
                               + 0.5 * alpha_ir * (1^T gamma + gamma^H 1))
        """
        cfg = self.cfg
        term_norm = torch.sum(torch.conj(gamma) * gamma).real
        term_sum = torch.sum(gamma).real * 2.0  # (1^T gamma + gamma^H 1) = 2 Re(sum(gamma))
        ps = cfg.ax * cfg.dy * (cfg.ci + cfg.alpha_r * term_norm + 0.5 * cfg.alpha_ir * term_sum)
        return ps

    def objective(self, x: torch.Tensor) -> torch.Tensor:
        gamma = realvec_to_complex_gamma(x, self.cfg.N)
        return torch.abs(self.Ps(gamma))

    def ak(self) -> float:
        cfg = self.cfg
        return (cfg.k0**2) * (abs(cfg.E0) ** 2) * (cfg.Lx**2) / (cfg.eta0 * 8.0 * math.pi**2 * cfg.Rk**2)

    def build_uik(self, theta_rad: torch.Tensor) -> torch.Tensor:
        """
        u_{ik,n} = exp(j * k0 * (sin(theta_k) - sin(theta_i)) * y_n)
        return shape [K, N] complex
        """
        cfg = self.cfg
        phase = cfg.k0 * (torch.sin(theta_rad).unsqueeze(1) - math.sin(math.radians(cfg.theta_i_deg))) * self.y_n.unsqueeze(0)
        return torch.exp(1j * phase)

    def reradiation_power(self, x: torch.Tensor, theta_rad: torch.Tensor) -> torch.Tensor:
        """
        P_theta(gamma) = ak * dy^2 * chi_ik * |gamma^T u_ik|^2
        """
        cfg = self.cfg
        gamma = realvec_to_complex_gamma(x, cfg.N)

        uik = self.build_uik(theta_rad)  # [K, N]
        gammaTu = torch.sum(uik * gamma.unsqueeze(0), dim=1)  # gamma^T u
        chi_ik = (
            math.cos(math.radians(cfg.theta_r_deg))**2
            + torch.cos(theta_rad)**2
            + 2.0 * math.cos(math.radians(cfg.theta_r_deg)) * torch.cos(theta_rad)
        )
        Pk = self.ak() * (cfg.dy**2) * chi_ik * complex_abs_sq(gammaTu)
        return Pk.real

    def z_from_gamma(self, x: torch.Tensor) -> torch.Tensor:
        """
        z_n = eta0 * (1 + gamma_n) / (cos(theta_i) - gamma_n cos(theta_r))
        """
        cfg = self.cfg
        gamma = realvec_to_complex_gamma(x, cfg.N)
        denom = cfg.alpha_i - cfg.alpha_r * gamma
        z = cfg.eta0 * (1.0 + gamma) / denom
        return z

    def g_mask(self, x: torch.Tensor) -> torch.Tensor:
        return self.reradiation_power(x, self.theta_mask_rad) - self.cfg.eps_RM

    def g_low(self, x: torch.Tensor) -> torch.Tensor:
        """
        Re(z_n) >= 0  <=>  -Re(z_n) <= 0
        """
        z = self.z_from_gamma(x)
        return -z.real

    def g_up_direct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Direct upper constraint:
        Re(z_n) <= eps_RI  <=>  Re(z_n) - eps_RI <= 0
        """
        z = self.z_from_gamma(x)
        return z.real - self.cfg.eps_RI

    def g_up_report(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quadratic upper constraint in the report's (10d):
        -(1 + epsRI_tilde * alpha_r) alpha_r |gamma_n|^2
        + alpha_i(1 - epsRI_tilde * alpha_i)
        + (alpha_i - alpha_r + 2 epsRI_tilde alpha_i alpha_r) Re(gamma_n) <= 0
        """
        cfg = self.cfg
        gamma = realvec_to_complex_gamma(x, cfg.N)
        g = (
            -(1.0 + cfg.eps_RI_tilde * cfg.alpha_r) * cfg.alpha_r * complex_abs_sq(gamma)
            + cfg.alpha_i * (1.0 - cfg.eps_RI_tilde * cfg.alpha_i)
            + (cfg.alpha_i - cfg.alpha_r + 2.0 * cfg.eps_RI_tilde * cfg.alpha_i * cfg.alpha_r) * gamma.real
        )
        return g

    def constraints(self, x: torch.Tensor, use_report_upper: bool = True) -> Dict[str, torch.Tensor]:
        gmask = self.g_mask(x)
        glow = self.g_low(x)
        gup = self.g_up_report(x) if use_report_upper else self.g_up_direct(x)
        return {"mask": gmask, "low": glow, "up": gup}

    def stacked_constraints(self, x: torch.Tensor, use_report_upper: bool = True) -> torch.Tensor:
        g = self.constraints(x, use_report_upper=use_report_upper)
        return torch.cat([g["mask"], g["low"], g["up"]], dim=0)


# ============================================================
# 4) Corrected ALM solver matching your flowchart
# ============================================================

class CorrectedALMSolver:
    def __init__(self, problem: RISSurfaceNetPowerProblem, cfg: ALMConfig):
        self.problem = problem
        self.cfg = cfg

    def aug_lagrangian(self, x, s, lam, sigma, use_report_upper=True):
        g = self.problem.stacked_constraints(x, use_report_upper=use_report_upper)
        r = g + s
        f = self.problem.objective(x)
        return f + torch.dot(lam, r) + 0.5 * sigma * torch.dot(r, r)

    def grad_x(self, x, s, lam, sigma, use_report_upper=True):
        x_var = x.detach().clone().requires_grad_(True)
        L = self.aug_lagrangian(x_var, s, lam, sigma, use_report_upper=use_report_upper)
        grad = torch.autograd.grad(L, x_var)[0]
        return grad.detach(), L.detach()

    def solve(self, x0: torch.Tensor, use_report_upper: bool = True) -> Dict:
        cfg = self.cfg
        x = x0.detach().clone().to(cfg.device, cfg.dtype)

        g0 = self.problem.stacked_constraints(x, use_report_upper=use_report_upper)
        s = torch.zeros_like(g0)
        lam = torch.zeros_like(g0)
        w = torch.zeros_like(x)

        sigma = cfg.sigma0
        eps = cfg.eps_stop
        eta = cfg.eta_stop

        history: Dict[str, List[float]] = {
            "obj": [],
            "v": [],
            "gnorm": [],
            "sigma": [],
        }

        for t in range(cfg.max_outer_iters):
            # 1) primal update with old variables
            grad, _ = self.grad_x(x, s, lam, sigma, use_report_upper=use_report_upper)
            grad_norm = torch.norm(grad, p=2)
            if torch.isfinite(grad_norm) and grad_norm > cfg.grad_clip_norm:
                grad = grad * (cfg.grad_clip_norm / (grad_norm + 1e-12))

            w = cfg.beta * w - cfg.mu_gamma * grad
            w_norm = torch.norm(w, p=2)
            if torch.isfinite(w_norm) and w_norm > cfg.momentum_clip_norm:
                w = w * (cfg.momentum_clip_norm / (w_norm + 1e-12))

            x_new = x + w
            x_new = torch.clamp(x_new, min=-cfg.x_clip_value, max=cfg.x_clip_value)

            # 2) slack update
            g_new = self.problem.stacked_constraints(x_new, use_report_upper=use_report_upper)
            s_new = torch.clamp(-g_new - lam / max(sigma, 1e-30), min=0.0)

            # 3) residual
            r_new = g_new + s_new

            # 4) stopping quantities
            v = torch.norm(r_new, p=1)
            grad_new, _ = self.grad_x(x_new, s_new, lam, sigma, use_report_upper=use_report_upper)
            gnorm = torch.norm(grad_new, p=2)
            obj = self.problem.objective(x_new)

            # NaN / Inf guard: stop early with last finite iterate
            if not (torch.isfinite(obj) and torch.isfinite(v) and torch.isfinite(gnorm)):
                print(f"Warning: non-finite value detected at iter={t + 1}, stopping early.")
                break

            history["obj"].append(obj.item())
            history["v"].append(v.item())
            history["gnorm"].append(gnorm.item())
            history["sigma"].append(float(sigma))

            # 5) feasibility / stationarity check
            if v <= eps:
                if gnorm <= eta:
                    return {
                        "x_star": x_new.detach(),
                        "gamma_star": realvec_to_complex_gamma(x_new.detach(), cfg.N),
                        "slack_star": s_new.detach(),
                        "lambda_star": lam.detach(),
                        "iterations": t + 1,
                        "history": history,
                    }
                sigma_next = sigma
                eps *= cfg.tighten_factor
                eta *= cfg.tighten_factor
            else:
                sigma_next = min(cfg.tau_sigma * sigma, cfg.sigma_max)

            # 6) corrected multiplier update
            # equality-form ALM must use residual g+s
            lam_new = lam + sigma * r_new
            lam_new = torch.clamp(lam_new, min=-cfg.lambda_clip_value, max=cfg.lambda_clip_value)

            x = x_new.detach()
            s = s_new.detach()
            lam = lam_new.detach()
            sigma = sigma_next

            if (t + 1) % 100 == 0:
                print(
                    f"iter={t + 1:4d}, obj={obj.item():.4e}, "
                    f"v={v.item():.4e}, gnorm={gnorm.item():.4e}, sigma={sigma:.4e}"
                )

        return {
            "x_star": x.detach(),
            "gamma_star": realvec_to_complex_gamma(x.detach(), cfg.N),
            "slack_star": s.detach(),
            "lambda_star": lam.detach(),
            "iterations": cfg.max_outer_iters,
            "history": history,
        }


# ============================================================
# 5) Initialization / visualization helpers
# ============================================================

def init_gamma_as_real_vector(cfg: ALMConfig, init_val: float = 1e-3) -> torch.Tensor:
    re = init_val * torch.ones(cfg.N, dtype=cfg.dtype, device=cfg.device)
    im = torch.zeros(cfg.N, dtype=cfg.dtype, device=cfg.device)
    return torch.cat([re, im], dim=0)


def save_result_figures(
    problem: RISSurfaceNetPowerProblem,
    result: Dict,
    out_dir: Path,
    use_report_upper: bool,
) -> List[Path]:
    """Save optimization history and reradiation pattern figures."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save image outputs.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    history = result["history"]
    iters = list(range(1, len(history["obj"]) + 1))

    fig1, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axs[0].plot(iters, history["obj"], linewidth=1.5)
    axs[0].set_ylabel("|Ps(gamma)|")
    axs[0].set_yscale("log")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(iters, history["v"], linewidth=1.5)
    axs[1].set_ylabel("||g+s||_1")
    axs[1].set_yscale("log")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(iters, history["gnorm"], linewidth=1.5)
    axs[2].set_ylabel("||grad L||_2")
    axs[2].set_xlabel("outer iteration")
    axs[2].set_yscale("log")
    axs[2].grid(True, alpha=0.3)
    fig1.suptitle("ALM convergence diagnostics")
    fig1.tight_layout()

    history_path = out_dir / "alm_history.png"
    fig1.savefig(history_path, dpi=150)
    plt.close(fig1)
    saved.append(history_path)

    theta_deg = torch.linspace(-90.0, 90.0, 1801, dtype=problem.dtype, device=problem.device)
    theta_rad = torch.deg2rad(theta_deg)
    p_theta = problem.reradiation_power(result["x_star"], theta_rad).detach().cpu()
    theta_mask = problem.cfg.theta_mask_deg.detach().cpu()
    p_mask = problem.reradiation_power(
        result["x_star"],
        problem.cfg.theta_mask_rad.to(device=problem.device, dtype=problem.dtype),
    ).detach().cpu()

    fig2, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(theta_deg.cpu().numpy(), p_theta.numpy(), label="reradiation power", linewidth=1.3)
    ax.scatter(theta_mask.numpy(), p_mask.numpy(), s=10, c="red", label="mask samples", alpha=0.8)
    ax.axhline(problem.cfg.eps_RM, color="black", linestyle="--", linewidth=1.0, label="eps_RM")
    upper_mode = "report" if use_report_upper else "direct"
    ax.set_title(f"Reradiation pattern (upper constraint: {upper_mode})")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("P_theta")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig2.tight_layout()

    pattern_path = out_dir / "reradiation_pattern.png"
    fig2.savefig(pattern_path, dpi=150)
    plt.close(fig2)
    saved.append(pattern_path)

    return saved


def _load_complex_vector(path: Path) -> torch.Tensor:
    """Load a complex vector from .pt/.pth/.npy/.npz/.txt/.csv file."""
    suffix = path.suffix.lower()

    if suffix in {".pt", ".pth"}:
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            for key in ("gamma", "gamma_star", "x"):
                if key in data:
                    data = data[key]
                    break
        tensor = torch.as_tensor(data)
        if tensor.is_complex():
            return tensor.flatten()
        if tensor.numel() % 2 == 0 and tensor.ndim == 1:
            n = tensor.numel() // 2
            return torch.complex(tensor[:n], tensor[n:])
        return torch.complex(tensor.flatten(), torch.zeros_like(tensor.flatten()))

    import numpy as np

    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        arr = data[data.files[0]]
    else:
        arr = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)

    arr = np.asarray(arr)
    if np.iscomplexobj(arr):
        return torch.from_numpy(arr.astype(np.complex128)).flatten()
    if arr.ndim == 2 and arr.shape[1] == 2:
        return torch.from_numpy((arr[:, 0] + 1j * arr[:, 1]).astype(np.complex128)).flatten()
    if arr.ndim == 1 and arr.size % 2 == 0:
        n = arr.size // 2
        return torch.from_numpy((arr[:n] + 1j * arr[n:]).astype(np.complex128)).flatten()
    return torch.from_numpy(arr.astype(np.float64)).to(torch.complex128).flatten()


def _build_gamma_plot_axis(problem: RISSurfaceNetPowerProblem, axis_mode: str) -> torch.Tensor:
    """
    根据用户需求，支持两种横轴：
    1) "position": 与论文 Figure 4/5 一致，横轴为每个 cell 的物理位置 y (m)。
    2) "cell": 横轴为 cell 编号 1..N。
    """
    if axis_mode == "position":
        return problem.y_n.detach().cpu()
    if axis_mode == "cell":
        return torch.arange(1, problem.cfg.N + 1, dtype=problem.dtype).cpu()
    raise ValueError(f"Unknown axis_mode: {axis_mode}")


def _trim_gamma(gamma: torch.Tensor, n_ref: int, tag: str) -> torch.Tensor:
    """
    将外部加载的 gamma 对齐到当前仿真的 N。
    - 长度大于 N: 截断到前 N 个元素；
    - 长度小于 N: 直接报错，避免 silently 画错图。
    """
    g = gamma.flatten().detach().cpu()
    if g.numel() < n_ref:
        raise ValueError(f"{tag} length={g.numel()} is smaller than required N={n_ref}.")
    return g[:n_ref]


def _plot_gamma_curve(ax, x_vals, gamma_vals: torch.Tensor, value_kind: str, label: str, style: str) -> None:
    """
    统一三种方法(CVX/ALM/Deep Unfolding)的曲线绘制入口，避免重复逻辑。
    value_kind:
      - "abs":   绘制 |gamma|
      - "angle": 绘制 angle(gamma)
    """
    if value_kind == "abs":
        y_vals = torch.abs(gamma_vals).numpy()
    elif value_kind == "angle":
        y_vals = torch.angle(gamma_vals).numpy()
    else:
        raise ValueError(f"Unknown value_kind: {value_kind}")

    if style == "cvx":
        ax.plot(x_vals, y_vals, "k-*", linewidth=1.0, markersize=6, label=label)
    elif style == "alm":
        ax.plot(
            x_vals,
            y_vals,
            color="red",
            marker="s",
            markerfacecolor="none",
            linewidth=1.0,
            markersize=6,
            label=label,
        )
    elif style == "deep":
        ax.plot(
            x_vals,
            y_vals,
            color="blue",
            marker=">",
            markerfacecolor="none",
            linewidth=1.0,
            markersize=6,
            label=label,
        )
    else:
        raise ValueError(f"Unknown style: {style}")


def draw_gamma_with_turtle(
    problem: RISSurfaceNetPowerProblem,
    gamma_aug: torch.Tensor,
    gamma_cvx: Optional[torch.Tensor] = None,
    gamma_deep: Optional[torch.Tensor] = None,
    axis_mode: str = "position",
) -> None:
    """
    使用 python turtle 直接在窗口里绘图（不依赖 matplotlib）。
    主要用于“本地想马上看图”的场景。
    """
    import turtle

    n_ref = problem.cfg.N
    x_axis = _build_gamma_plot_axis(problem, axis_mode).numpy()

    g_alm = _trim_gamma(gamma_aug, n_ref=n_ref, tag="ALM gamma")
    g_cvx = _trim_gamma(gamma_cvx, n_ref=n_ref, tag="CVX gamma") if gamma_cvx is not None else None
    g_deep = _trim_gamma(gamma_deep, n_ref=n_ref, tag="Deep-unfolding gamma") if gamma_deep is not None else None

    curves = [
        ("ALM abs", torch.abs(g_alm).numpy(), "red"),
        ("ALM angle", torch.angle(g_alm).numpy(), "orange"),
    ]
    if g_cvx is not None:
        curves += [
            ("CVX abs", torch.abs(g_cvx).numpy(), "black"),
            ("CVX angle", torch.angle(g_cvx).numpy(), "gray"),
        ]
    if g_deep is not None:
        curves += [
            ("Deep abs", torch.abs(g_deep).numpy(), "blue"),
            ("Deep angle", torch.angle(g_deep).numpy(), "green"),
        ]

    all_y = [y for _, ys, _ in curves for y in ys]
    y_min, y_max = float(min(all_y)), float(max(all_y))
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    screen = turtle.Screen()
    screen.title("Gamma plot by turtle (ABS and ANGLE)")
    screen.setup(width=1200, height=800)
    screen.bgcolor("white")

    pen = turtle.Turtle(visible=False)
    pen.speed(0)
    pen.pensize(2)

    left, right = -520, 520
    bottom, top = -320, 320

    def map_x(v):
        x0, x1 = float(min(x_axis)), float(max(x_axis))
        if abs(x1 - x0) < 1e-12:
            return (left + right) / 2
        return left + (v - x0) / (x1 - x0) * (right - left)

    def map_y(v):
        return bottom + (v - y_min) / (y_max - y_min) * (top - bottom)

    # axis
    pen.color("black")
    pen.penup(); pen.goto(left, 0); pen.pendown(); pen.goto(right, 0)
    pen.penup(); pen.goto(0, bottom); pen.pendown(); pen.goto(0, top)

    # plot curves
    for name, ys, color in curves:
        pen.color(color)
        pen.penup()
        pen.goto(map_x(x_axis[0]), map_y(ys[0]))
        pen.pendown()
        for xv, yv in zip(x_axis[1:], ys[1:]):
            pen.goto(map_x(float(xv)), map_y(float(yv)))

    # legend text
    pen.penup()
    pen.goto(left, top + 10)
    pen.color("black")
    pen.write(" | ".join([f"{name}" for name, _, _ in curves]), font=("Arial", 10, "normal"))

    turtle.done()


def save_gamma_comparison_figures(
    problem: RISSurfaceNetPowerProblem,
    gamma_aug: torch.Tensor,
    out_dir: Path,
    gamma_cvx: Optional[torch.Tensor] = None,
    gamma_deep: Optional[torch.Tensor] = None,
    axis_mode: str = "position",
) -> List[Path]:
    """
    生成论文 Figure 4/5 风格的两张图：
      - abs_gamma.png
      - angle_gamma.png

    说明：
    - 三条曲线分别对应 CVX / ALM / Deep unfolding；
    - 横轴可选 cell 编号或物理位置。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save image outputs.") from exc

    out_dir.mkdir(parents=True, exist_ok=True)

    n_ref = problem.cfg.N
    axis_tensor = _build_gamma_plot_axis(problem, axis_mode)
    axis_np = axis_tensor.numpy()

    # 统一裁剪长度，确保三种方法在同一个 N 上对比。
    g_alm = _trim_gamma(gamma_aug, n_ref=n_ref, tag="ALM gamma")
    g_cvx = _trim_gamma(gamma_cvx, n_ref=n_ref, tag="CVX gamma") if gamma_cvx is not None else None
    g_deep = _trim_gamma(gamma_deep, n_ref=n_ref, tag="Deep-unfolding gamma") if gamma_deep is not None else None

    xlabel = "RIS element position y" if axis_mode == "position" else "Cell index n"

    # -------------------- Figure 4: ABS(gamma) --------------------
    fig_abs, ax_abs = plt.subplots(figsize=(9, 6))
    if g_cvx is not None:
        _plot_gamma_curve(ax_abs, axis_np, g_cvx, value_kind="abs", label="CVX", style="cvx")
    _plot_gamma_curve(ax_abs, axis_np, g_alm, value_kind="abs", label="Augment method", style="alm")
    if g_deep is not None:
        _plot_gamma_curve(ax_abs, axis_np, g_deep, value_kind="abs", label="Deep unfolding model", style="deep")

    ax_abs.set_title("ABS(gamma)")
    ax_abs.set_xlabel(xlabel)
    ax_abs.set_ylabel("|gamma|")
    ax_abs.grid(True, alpha=0.3)
    ax_abs.legend(loc="upper right")
    fig_abs.tight_layout()

    abs_path = out_dir / "abs_gamma.png"
    fig_abs.savefig(abs_path, dpi=200)
    plt.close(fig_abs)

    # -------------------- Figure 5: ANGLE(gamma) --------------------
    fig_ang, ax_ang = plt.subplots(figsize=(9, 6))
    if g_cvx is not None:
        _plot_gamma_curve(ax_ang, axis_np, g_cvx, value_kind="angle", label="CVX", style="cvx")
    _plot_gamma_curve(ax_ang, axis_np, g_alm, value_kind="angle", label="Augment method", style="alm")
    if g_deep is not None:
        _plot_gamma_curve(ax_ang, axis_np, g_deep, value_kind="angle", label="Deep unfolding model", style="deep")

    ax_ang.set_title("ANGLE(gamma)")
    ax_ang.set_xlabel(xlabel)
    ax_ang.set_ylabel("angle(gamma) [rad]")
    ax_ang.set_ylim([-4.0, 4.0])
    ax_ang.grid(True, alpha=0.3)
    ax_ang.legend(loc="lower center")
    fig_ang.tight_layout()

    angle_path = out_dir / "angle_gamma.png"
    fig_ang.savefig(angle_path, dpi=200)
    plt.close(fig_ang)

    return [abs_path, angle_path]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected ALM solver for RIS surface net-power optimization.")
    parser.add_argument("--max-iters", type=int, default=2000, help="Maximum ALM outer iterations.")
    parser.add_argument("--save-figures", action="store_true", help="Save output figures after optimization.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where PNG figures are saved when --save-figures is enabled.",
    )
    parser.add_argument(
        "--use-direct-upper",
        action="store_true",
        help="Use direct upper impedance constraint instead of report quadratic form.",
    )
    parser.add_argument(
        "--save-gamma-figures",
        action="store_true",
        help="Save both ABS(gamma) and ANGLE(gamma) comparison figures (Figure 4/5 style).",
    )
    parser.add_argument(
        "--cvx-gamma-file",
        type=Path,
        default=None,
        help="Optional path to CVX gamma vector (.pt/.pth/.npy/.npz/.txt/.csv).",
    )
    parser.add_argument(
        "--deep-gamma-file",
        type=Path,
        default=None,
        help="Optional path to deep-unfolding gamma vector (.pt/.pth/.npy/.npz/.txt/.csv).",
    )
    parser.add_argument(
        "--gamma-x-axis",
        type=str,
        choices=["position", "cell"],
        default="position",
        help="X-axis for gamma plots: 'position' (paper-style y coordinate) or 'cell' (index 1..N).",
    )
    parser.add_argument(
        "--draw-gamma-turtle",
        action="store_true",
        help="Draw ABS/ANGLE gamma curves with python turtle for immediate local visualization.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip ALM solving and only draw/save plots using provided gamma files.",
    )
    return parser.parse_args()


# ============================================================
# 6) Example run
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    cfg = ALMConfig(
        mu_gamma=2e-12,
        beta=0.9,
        sigma0=100.0,
        tau_sigma=1.05,
        eps_stop=1e-6,
        eta_stop=1e-6,
        max_outer_iters=args.max_iters,
        device="cpu",
        dtype=torch.float64,
    )

    problem = RISSurfaceNetPowerProblem(cfg)
    solver = CorrectedALMSolver(problem, cfg)

    use_report_upper = not args.use_direct_upper

    gamma_cvx = _load_complex_vector(args.cvx_gamma_file) if args.cvx_gamma_file else None
    gamma_deep = _load_complex_vector(args.deep_gamma_file) if args.deep_gamma_file else None

    if args.plot_only:
        # plot-only 模式下，不再跑 ALM；优先用 deep 曲线当主曲线，否则用 cvx。
        if gamma_deep is not None:
            gamma_aug = gamma_deep
        elif gamma_cvx is not None:
            gamma_aug = gamma_cvx
        else:
            raise ValueError("--plot-only requires at least one of --deep-gamma-file or --cvx-gamma-file.")

        if args.save_gamma_figures:
            gamma_fig_paths = save_gamma_comparison_figures(
                problem=problem,
                gamma_aug=gamma_aug,
                out_dir=args.figure_dir,
                gamma_cvx=gamma_cvx,
                gamma_deep=gamma_deep,
                axis_mode=args.gamma_x_axis,
            )
            print("saved gamma comparison figures (plot-only mode):")
            for path in gamma_fig_paths:
                print(" -", path)

        if args.draw_gamma_turtle:
            draw_gamma_with_turtle(
                problem=problem,
                gamma_aug=gamma_aug,
                gamma_cvx=gamma_cvx,
                gamma_deep=gamma_deep,
                axis_mode=args.gamma_x_axis,
            )
    else:
        x0 = init_gamma_as_real_vector(cfg, init_val=1e-3)
        result = solver.solve(x0, use_report_upper=use_report_upper)

        gamma_star = result["gamma_star"]
        print("\nFinished.")
        print("iterations:", result["iterations"])
        print("final ||gamma||2:", torch.norm(torch.abs(gamma_star), p=2).item())
        print("final objective |Ps(gamma)|:", problem.objective(result["x_star"]).item())

        if args.save_figures:
            figure_paths = save_result_figures(
                problem=problem,
                result=result,
                out_dir=args.figure_dir,
                use_report_upper=use_report_upper,
            )
            print("saved figures:")
            for path in figure_paths:
                print(" -", path)

        if args.save_gamma_figures:
            gamma_fig_paths = save_gamma_comparison_figures(
                problem=problem,
                gamma_aug=result["gamma_star"],
                out_dir=args.figure_dir,
                gamma_cvx=gamma_cvx,
                gamma_deep=gamma_deep,
                axis_mode=args.gamma_x_axis,
            )
            print("saved gamma comparison figures:")
            for path in gamma_fig_paths:
                print(" -", path)

        if args.draw_gamma_turtle:
            draw_gamma_with_turtle(
                problem=problem,
                gamma_aug=result["gamma_star"],
                gamma_cvx=gamma_cvx,
                gamma_deep=gamma_deep,
                axis_mode=args.gamma_x_axis,
            )
