"""
FedLoRD server: Ditto-style bilevel low-rank personalization.

The server solves ONLY the lower problem (the plain global FL / FedAvg on f_i(Theta)):
it broadcasts Theta^t, each client runs the bilevel local update, and the server
aggregates ONLY the global part w_i by weighted averaging over the PARTICIPATING
clients (denominator = sum of participants' data sizes, so partial participation is
handled naturally):
    Theta^{t+1} = sum_{i in S_t} (n_i / sum_{j in S_t} n_j) w_i.
Personal factors P_i, Q_i never leave the client.
"""
import time
import torch
from flcore.clients.clientfedlord import clientFedLoRD
from flcore.servers.serverbase import Server


class FedLoRD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientFedLoRD)

        c0 = self.clients[0]
        rank_str = "min(m,n)" if c0.rank <= 0 else str(c0.rank)
        etapq_str = "auto(=global lr)" if c0.eta_pq <= 0 else str(c0.eta_pq)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        pqu_str = 'GaussSeidel' if c0.pq_gs else ('SVD-prox (ablation)' if c0.pq_svd else 'Jacobi')
        if c0.rank_ratio > 0:
            rank_str = f"ratio {c0.rank_ratio} x min(m,n)"
        if c0.local_iters is not None and c0.local_iters > 0:
            K = c0.local_iters; K_theta = max(1, int(round(c0.k_theta_frac * K)))
            budget_str = f"K={K} steps/round (T_Theta={K_theta}, T_PQ={K - K_theta})"
        else:
            budget_str = f"E_theta={c0.E_theta}, E_pq={c0.E_pq} epochs/round"
        print(f"[FedLoRD] rank={rank_str}, lambda={c0.lam}, eta_pq={etapq_str}, {budget_str}, PQ-update={pqu_str}")
        print("Finished creating server and clients.")
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
        self._dump_localtime()
        if 'rankdiag' in self.goal:
            self._report_lowrank()

    def _report_lowrank(self):
        """Diagnostic: is the personalized correction Delta_i = P_i Q_i actually LOW-rank?
        Reports, per 2D layer (averaged over clients), the max possible rank min(m,n) vs the
        effective rank (numerical rank = #singular-values > 1% of the largest; energy rank =
        #sv to reach 90% Frobenius energy; stable rank = ||D||_F^2 / sigma_max^2)."""
        import numpy as np
        from collections import defaultdict
        print("\n===== FedLoRD LOW-RANK DIAGNOSTIC: Delta_i = P_i Q_i (avg over clients) =====", flush=True)
        stats = defaultdict(list)
        with torch.no_grad():
            for client in self.clients:
                for name in client.P:
                    m, n, _ = client.shape2d[name]
                    delta = client.P[name].float() @ client.Q[name].float()      # m x n
                    sv = torch.linalg.svdvals(delta)
                    smax = sv[0].item() if sv.numel() and sv[0].item() > 0 else 0.0
                    if smax <= 0:
                        stats[name].append((min(m, n), 0, 0, 0.0)); continue
                    num_rank = int((sv > 0.01 * smax).sum().item())
                    energy = torch.cumsum(sv ** 2, 0) / (sv ** 2).sum()
                    energy_rank = int((energy < 0.90).sum().item()) + 1
                    stable = float((sv ** 2).sum() / (smax ** 2))
                    stats[name].append((min(m, n), num_rank, energy_rank, stable))
        print(f"  {'layer':28s} {'max_rank':>9s} {'num_rank(>1%)':>14s} {'energy_rank(90%)':>17s} {'stable_rank':>12s} {'num/max':>8s}", flush=True)
        for name, vals in stats.items():
            maxr = vals[0][0]
            nr = np.mean([v[1] for v in vals]); er = np.mean([v[2] for v in vals]); sr = np.mean([v[3] for v in vals])
            print(f"  {name:28s} {maxr:9d} {nr:14.1f} {er:17.1f} {sr:12.1f} {nr/max(maxr,1):8.1%}", flush=True)

    def _dump_localtime(self):
        """Per-client, per-round local-training time (round, t_lower, t_upper, t_total) [sec] -> JSON."""
        import json, os
        log = {int(c.id): [[int(r), float(a), float(b), float(t)] for r, a, b, t in getattr(c, 'tlog', [])]
               for c in self.clients}
        rdir = getattr(self.args, 'result_dir', '../results/').rstrip('/')
        os.makedirs(rdir, exist_ok=True)
        fn = f"{rdir}/{self.dataset}_{self.algorithm}_{self.goal}_localtime.json"
        with open(fn, 'w') as f:
            json.dump(log, f)
        tot = [v[3] for vs in log.values() for v in vs]
        low = [v[1] for vs in log.values() for v in vs]
        upp = [v[2] for vs in log.values() for v in vs]
        if tot:
            print(f"[localtime] per (round,client): lower={sum(low)/len(low):.4f}s  upper={sum(upp)/len(upp):.4f}s  "
                  f"total={sum(tot)/len(tot):.4f}s  (n={len(tot)}) -> {fn}")

    def send_models(self):
        """Broadcast the global backbone Theta^t (the round-start reference)."""
        assert len(self.clients) > 0
        z = self.global_model.state_dict()
        for client in self.clients:
            client.set_global_model(z)

    def receive_models(self):
        """Collect the locally-trained global parts w_i and data sizes (participants)."""
        assert len(self.selected_clients) > 0
        self.uploaded_w = []
        self.uploaded_weights = []
        tot = 0.0
        for client in self.selected_clients:
            self.uploaded_w.append(client.get_upload())
            self.uploaded_weights.append(client.train_samples)
            tot += client.train_samples
        self.uploaded_weights = [w / tot for w in self.uploaded_weights]   # normalize over participants

    def aggregate_parameters(self):
        """Weighted FedAvg of the global part over participating clients (Theta^{t+1})."""
        assert len(self.uploaded_w) > 0
        global_state = self.global_model.state_dict()
        param_names = set(n for n, _ in self.global_model.named_parameters())

        z_new = {}
        for key in global_state.keys():
            if key not in param_names:
                z_new[key] = global_state[key].detach().cpu().clone()
                continue
            acc = torch.zeros(global_state[key].shape, dtype=torch.float)
            for w_i, wt in zip(self.uploaded_w, self.uploaded_weights):
                acc += wt * w_i[key].float()
            z_new[key] = acc.to(global_state[key].dtype)

        self.global_model.load_state_dict(z_new)
