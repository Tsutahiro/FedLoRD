"""
FedLoRD client: low-rank personalization with nuclear-norm regularisation, solved by a
Ditto-style bilevel local update.

Personalized model  X_i = Theta + P_i Q_i  (shared backbone + low-rank personal
correction). The nuclear norm ||X_i - Theta||_* is replaced by its variational form
(Srebro et al., 2005): min_{PQ=Delta} 0.5(||P||_F^2 + ||Q||_F^2), so the regularizer becomes a
plain ridge on the factors P, Q -> NO SVD / NO proximal operator.

Bilevel (Ditto-style) structure, per round, with the round-start global Theta^t as reference:
  (lower) w_i <- Theta^t, then T_Theta steps of PLAIN SGD on f_i(w_i)  [no personal part]
  (upper) T_PQ steps updating P_i, Q_i at the fixed reference:
            g = grad f_i(Theta^t + P_i Q_i)
            P_i <- P_i - eta_pq (g Q_i^T + lam P_i)
            Q_i <- Q_i - eta_pq (P_i^T g + lam Q_i)
The client uploads only w_i (global part); P_i, Q_i never leave the client.

Local budget: either the equal-update protocol of the paper (-li K > 0: T_Theta = round(kth*K)
and T_PQ = K - T_Theta mini-batch steps per round) or legacy local epochs (-ls / -EO / -EPQ).

P,Q update mode is switchable (-pqu): 'jacobi' (simultaneous, one grad), 'gs' (Gauss-Seidel:
update P, recompute grad, update Q) or 'svd' (ablation WITHOUT variable separation: a full
Delta_i updated by proximal SGD with singular-value thresholding). 2D weights (conv reshaped,
linear) use the low-rank P,Q; 1D params (bias / GroupNorm) use a full personal vector d_i with
the same ridge.
"""
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientFedLoRD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.rank = getattr(args, 'rank', -1)                    # <=0 -> min(m,n) per layer
        self.lam = getattr(args, 'nuc_lambda', 0.01)
        self.eta_pq = getattr(args, 'eta_pq', -1.0)              # <=0 -> match global (decaying) lr
        et = getattr(args, 'E_theta', -1)
        self.E_theta = et if et > 0 else self.local_epochs       # lower epochs
        ep = getattr(args, 'E_pq', -1)
        self.E_pq = ep if ep > 0 else self.local_epochs          # upper epochs
        _pqu = getattr(args, 'pq_update', 'jacobi')
        self.pq_gs  = (_pqu == 'gs')
        self.pq_svd = (_pqu == 'svd')        # ablation: NO variable separation -> full Delta + proximal SGD (SVT)
        self.rank_ratio = getattr(args, 'rank_ratio', -1.0)   # >0: r = max(1, round(ratio*min(m,n))) per layer
        self.tlog = []                        # per-round local-time log: (round, t_lower, t_upper, t_total) [sec]
        # equal-update protocol (-li K > 0): K_theta lower steps + K_pq upper steps = K (default split 0.8 = EO4/EPQ1 ratio)
        self.k_theta_frac = getattr(args, 'k_theta_frac', 0.8)

        # Personal factors: 2D layers -> (P[m x r], Q[r x n]);  1D layers -> d (full vector).
        # LoRA-style init: Q ~ small Gaussian, P = 0  => P Q = 0 initially (X_i = Theta).
        # svd mode: 2D layers -> full Delta[m x n] (init 0), updated by proximal SGD (no factors).
        self.P, self.Q, self.d, self.shape2d, self.Dlt = {}, {}, {}, {}, {}
        for name, p in self.model.named_parameters():
            if p.dim() >= 2:
                m, n = self._mat_shape(tuple(p.shape))
                if self.rank_ratio > 0:
                    r = max(1, int(round(self.rank_ratio * min(m, n))))
                else:
                    r = min(m, n) if self.rank <= 0 else min(self.rank, m, n)   # rank upper bound
                self.shape2d[name] = (m, n, tuple(p.shape))
                if self.pq_svd:
                    self.Dlt[name] = torch.zeros(m, n)
                else:
                    self.Q[name] = (torch.randn(r, n) * 0.01)
                    self.P[name] = torch.zeros(m, r)
            else:
                self.d[name] = torch.zeros_like(p.data, device='cpu')

        # ---- per-layer nuclear-norm strength (opt-in) --------------------------------
        # nlam_embed >= 0 overrides lam for embedding rows (which are high-rank on text,
        # so a small/zero lam lets them personalize freely instead of forcing low-rank).
        self.lam_embed = getattr(args, 'nlam_embed', -1.0)
        self.lam_map = {}
        for name, p in self.model.named_parameters():
            lm = self.lam
            if self.lam_embed >= 0 and 'embedding' in name.lower():
                lm = self.lam_embed
            self.lam_map[name] = lm

        self.theta_ref = None    # round-start global Theta^t (CPU)
        self.w_upload = None     # locally-trained global part w_i

    # ---- matrix reshape helpers (match svt_lowrank convention) ----
    @staticmethod
    def _mat_shape(shape):
        if len(shape) == 4:
            o, i, kh, kw = shape
            return o * kh, i * kw
        return shape[0], int(np.prod(shape[1:]))

    @staticmethod
    def _to_mat(t, shape):
        if len(shape) == 4:
            o, i, kh, kw = shape
            return t.permute(0, 2, 1, 3).reshape(o * kh, i * kw)
        return t.reshape(shape[0], -1)

    @staticmethod
    def _from_mat(mat, shape):
        if len(shape) == 4:
            o, i, kh, kw = shape
            return mat.reshape(o, kh, i, kw).permute(0, 2, 1, 3).contiguous()
        return mat.reshape(shape)

    # ---- server -> client ----
    def set_global_model(self, z: dict):
        self.theta_ref = {k: v.detach().cpu().clone() for k, v in z.items()}

    def _delta(self, name, dev):
        """Personal correction Delta_i in native param shape (on device)."""
        if name in self.Dlt:                                    # svd mode: full Delta
            return self._from_mat(self.Dlt[name].to(dev), self.shape2d[name][2])
        if name in self.P:
            m, n, shape = self.shape2d[name]
            mat = self.P[name].to(dev) @ self.Q[name].to(dev)   # m x n
            return self._from_mat(mat, shape)
        return self.d[name].to(dev)

    def _set_personalized(self, dev):
        """Load Theta_ref + Delta_i into the model (in place)."""
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                p.copy_(self.theta_ref[name].to(dev) + self._delta(name, dev))

    def _batches(self, loader, dev):
        for x, y in loader:
            if type(x) == type([]):
                x[0] = x[0].to(dev)
            else:
                x = x.to(dev)
            yield x, y.to(dev)

    # ---- one federated round ----
    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        dev = self.device
        _sync = (lambda: torch.cuda.synchronize()) if torch.cuda.is_available() else (lambda: None)
        _sync(); _t0 = time.time()

        # ===== (lower) global local update: plain SGD from Theta^t =====
        self.model.load_state_dict(self._full_state(self.theta_ref))
        self.model.train()
        E_theta = self.E_theta
        if self.train_slow:
            E_theta = np.random.randint(1, max(2, E_theta // 2))
        K = self.local_steps()
        if K is not None:
            K_theta = max(1, int(round(self.k_theta_frac * K))); K_pq = max(0, K - K_theta)
            lower_loaders = [self.iter_batches(trainloader, K_theta)]
            upper_loaders = [self.iter_batches(trainloader, K_pq)] if K_pq > 0 else []
            self.k_split = (K_theta, K_pq)
        else:
            lower_loaders = [trainloader] * E_theta
            upper_loaders = [trainloader] * self.E_pq
        for _loader in lower_loaders:
            for x, y in self._batches(_loader, dev):
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.w_upload = {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}
        _sync(); _t1 = time.time(); t_lower = _t1 - _t0

        # ===== (upper) personal update: P,Q (and 1D d) at fixed reference Theta^t =====
        # eta_pq: match the global update's (decaying) lr unless overridden (>0).
        if self.eta_pq > 0:
            eta = self.eta_pq
        elif self.learning_rate_decay:
            eta = self.learning_rate_scheduler.get_last_lr()[0]
        else:
            eta = self.learning_rate
        lam = self.lam
        for _loader in upper_loaders:
            for x, y in self._batches(_loader, dev):
                # first grad at Theta^t + Delta
                g = self._grad_at_personalized(x, y, dev)
                with torch.no_grad():
                    for name in self.d:                       # 1D full-vector personal
                        dd = self.d[name].to(dev)
                        self.d[name] = (dd - eta * (g[name] + self.lam_map[name] * dd)).cpu()
                    for name in self.P:                       # 2D low-rank: update P
                        m, n, shape = self.shape2d[name]
                        lam = self.lam_map[name]              # per-layer nuclear strength
                        G = self._to_mat(g[name], shape)
                        P, Q = self.P[name].to(dev), self.Q[name].to(dev)
                        Pn = P - eta * (G @ Q.t() + lam * P)
                        if self.pq_gs:
                            self.P[name] = Pn.cpu()           # commit P before recompute
                        else:
                            Qn = Q - eta * (P.t() @ G + lam * Q)   # Jacobi: same G, OLD P
                            self.P[name], self.Q[name] = Pn.cpu(), Qn.cpu()
                    for name in self.Dlt:                     # svd mode: full Delta, proximal SGD
                        m, n, shape = self.shape2d[name]      #   Delta <- SVT(Delta - eta*G, eta*lam)
                        lam = self.lam_map[name]
                        G = self._to_mat(g[name], shape)
                        Z = self.Dlt[name].to(dev) - eta * G
                        U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
                        S = torch.clamp(S - eta * lam, min=0.0)
                        self.Dlt[name] = ((U * S) @ Vh).cpu()
                if self.pq_gs:
                    # Gauss-Seidel: recompute grad at Theta^t + P_new Q, then update Q
                    g2 = self._grad_at_personalized(x, y, dev)
                    with torch.no_grad():
                        for name in self.P:
                            m, n, shape = self.shape2d[name]
                            G2 = self._to_mat(g2[name], shape)
                            P, Q = self.P[name].to(dev), self.Q[name].to(dev)
                            self.Q[name] = (Q - eta * (P.t() @ G2 + self.lam_map[name] * Q)).cpu()

        _sync(); _t2 = time.time(); t_upper = _t2 - _t1
        self.tlog.append((self.train_time_cost['num_rounds'], t_lower, t_upper, _t2 - _t0))

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _grad_at_personalized(self, x, y, dev):
        """grad of f_i at X_i = Theta^t + Delta_i, returned per-layer in native shape."""
        self._set_personalized(dev)
        self.model.train()
        output = self.model(x)
        loss = self.loss(output, y)
        self.model.zero_grad()
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in self.model.named_parameters()}

    def _full_state(self, param_dict):
        state = self.model.state_dict()
        return {k: (param_dict[k] if k in param_dict else state[k]) for k in state.keys()}

    def get_upload(self):
        return self.w_upload

    # ---- evaluation: personalized X_i = Theta + Delta_i ----
    def _load_personalized(self):
        if self.theta_ref is None:
            return
        dev = self.device
        with torch.no_grad():
            self._set_personalized(dev)

    def test_metrics(self):
        self._load_personalized()
        return super().test_metrics()

    def train_metrics(self):
        self._load_personalized()
        return super().train_metrics()
