import math
from dataclasses import dataclass
from typing import Dict, List

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
    tau_sigma: float = 1.1
    eps_stop: float = 1e-6
    eta_stop: float = 1e-6
    tighten_factor: float = 0.8
    max_outer_iters: int = 2000

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

        self.ones_c = torch.ones(cfg.N, device=self.device, dtype=self.dtype)

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

        uik = self.build_uik(theta_rad)                    # [K, N]
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
            w = cfg.beta * w - cfg.mu_gamma * grad
            x_new = x + w

            # 2) slack update
            g_new = self.problem.stacked_constraints(x_new, use_report_upper=use_report_upper)
            s_new = torch.clamp(-g_new - lam / sigma, min=0.0)

            # 3) residual
            r_new = g_new + s_new

            # 4) stopping quantities
            v = torch.norm(r_new, p=1)
            grad_new, _ = self.grad_x(x_new, s_new, lam, sigma, use_report_upper=use_report_upper)
            gnorm = torch.norm(grad_new, p=2)
            obj = self.problem.objective(x_new)

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
                else:
                    sigma_next = sigma
                    eps *= cfg.tighten_factor
                    eta *= cfg.tighten_factor
            else:
                sigma_next = cfg.tau_sigma * sigma

            # 6) corrected multiplier update
            # equality-form ALM must use residual g+s
            lam_new = lam + sigma * r_new

            x = x_new.detach()
            s = s_new.detach()
            lam = lam_new.detach()
            sigma = sigma_next

            if (t + 1) % 100 == 0:
                print(
                    f"iter={t+1:4d}, obj={obj.item():.4e}, "
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
# 5) Initialization
# ============================================================

def init_gamma_as_real_vector(cfg: ALMConfig, init_val: float = 1e-3) -> torch.Tensor:
    re = init_val * torch.ones(cfg.N, dtype=cfg.dtype, device=cfg.device)
    im = torch.zeros(cfg.N, dtype=cfg.dtype, device=cfg.device)
    return torch.cat([re, im], dim=0)


# ============================================================
# 6) Example run
# ============================================================

if __name__ == "__main__":
    cfg = ALMConfig(
        mu_gamma=2e-12,
        beta=0.9,
        sigma0=100.0,
        tau_sigma=1.1,
        eps_stop=1e-6,
        eta_stop=1e-6,
        max_outer_iters=2000,
        device="cpu",
        dtype=torch.float64,
    )

    problem = RISSurfaceNetPowerProblem(cfg)
    solver = CorrectedALMSolver(problem, cfg)

    x0 = init_gamma_as_real_vector(cfg, init_val=1e-3)

    # use_report_upper=True means use Shumin's quadratic upper-bound form (10d)
    result = solver.solve(x0, use_report_upper=True)

    gamma_star = result["gamma_star"]
    print("\nFinished.")
    print("iterations:", result["iterations"])
    print("final ||gamma||2:", torch.norm(torch.abs(gamma_star), p=2).item())
    print("final objective |Ps(gamma)|:", problem.objective(result["x_star"]).item())
