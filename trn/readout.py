"""Linear readouts fitted by ridge regression on int8 reservoir features.

Sufficient statistics (X^T X, X^T Y) are accumulated with exact int8 tensor-core
matmuls, so any amount of data can be added and the readout re-solved in closed form.
The regularisation strength is picked on held-out validation rows.
"""
import os, torch


class Ridge:
    FLUSH_BYTES = int(os.environ.get("TRN_FLUSH_MB", 512)) * 2 ** 20      # buffered rows are folded into the Gram at this size

    def __init__(self, D, C, device="cuda", val_frac=0.1, val_cap=20000, seed=0):
        self.D, self.C, self.dev = D, C, device
        self.XtX = torch.zeros(D, D, dtype=torch.float32, device=device)   # int32-exact chunks summed in fp32
        self.Xs = torch.zeros(D, dtype=torch.float64, device=device)
        self.XtY = torch.zeros(D, C, dtype=torch.float64, device=device)
        self.Ys = torch.zeros(C, dtype=torch.float64, device=device)
        self.n = 0
        self.vX, self.vY, self.val_frac, self.val_cap = [], [], val_frac, val_cap
        self.g = torch.Generator(device=device).manual_seed(seed)
        self.W = None   # (D+1, C)

    def nval(self):
        return sum(x.shape[0] for x in self.vX)

    @torch.no_grad()
    def add(self, X, Y):
        """X: (n, D) int8, Y: (n, C) float. Rows are buffered and folded in large exact int8 matmuls."""
        if len(X) == 0:
            return
        self.buf = getattr(self, "buf", [])
        self.buf.append((X.clone(), Y.to(self.dev, torch.float32)))
        self.nbuf = getattr(self, "nbuf", 0) + len(X)
        if self.nbuf * self.D >= self.FLUSH_BYTES:
            self.flush()

    @torch.no_grad()
    def flush(self):
        b, self.buf, self.nbuf = getattr(self, "buf", []), [], 0
        if b:
            self._add(torch.cat([x for x, _ in b]), torch.cat([y for _, y in b]))

    @torch.no_grad()
    def _add(self, X, Y):
        self._to_dev()                       # more data after a fit (racing): bring the Gram back
        m = torch.rand(len(X), generator=self.g, device=self.dev) < self.val_frac
        if m.any() and self.nval() < self.val_cap:
            self.vX.append(X[m].clone()); self.vY.append(Y[m].clone())
            X, Y = X[~m], Y[~m]
        for i in range(0, len(X), 65536):                   # int32-exact chunks
            x, y = X[i:i + 65536], Y[i:i + 65536]
            pad = (-len(x)) % 8
            if pad:
                x = torch.cat([x, torch.zeros(pad, self.D, dtype=torch.int8, device=self.dev)])
            if len(x) <= 16:
                self.XtX += (x.float().T @ x.float())
            else:
                xt = x.T.contiguous()
                for j in range(0, self.D, 2048):              # column tiles: no DxD int32/fp32 transients
                    self.XtX[:, j:j + 2048] += torch._int_mm(xt, x[:, j:j + 2048].contiguous()).float()
                del xt
            x = x[: len(y)]
            self.Xs += x.sum(0, dtype=torch.int32).double()        # no float copy of the whole block
            for k in range(0, len(y), 4096):                        # bounded fp32 temporaries
                self.XtY += (x[k:k + 4096].float().T @ y[k:k + 4096]).double()
            self.Ys += y.sum(0).double()
            self.n += len(y)

    def _solve(self, alpha):
        n = max(self.n, 1)
        mx, my = self.Xs / n, self.Ys / n                     # centre (= unpenalised bias)
        A = self.XtX.to(self.dev, copy=True)          # fresh GPU copy (the Gram itself stays parked in host RAM)
        A.div_(n).addr_(mx.float(), mx.float(), alpha=-1.0)          # in place: no DxD temporary
        lam = float(alpha * A.diagonal().mean().clamp_min(1e-6))
        A.diagonal().add_(lam)
        Bm = (self.XtY / n - torch.outer(mx, my)).float()
        info = torch.empty((), dtype=torch.int32, device=self.dev)
        torch.linalg.cholesky_ex(A, out=(A, info))
        if info.item() != 0:
            return None
        w = torch.cholesky_solve(Bm, A)
        del A
        b = my.float() - mx.float() @ w
        return torch.cat([w, b[None]], 0)

    def predict(self, X, W=None):
        W = self.W if W is None else W
        return X.float() @ W[:-1] + W[-1]

    def _to_host(self):
        """park the Gram in pinned host RAM: solves rebuild a fresh GPU copy, so the GPU never holds a Gram
        and its clone at once (peak = the 3 collection Grams, not 3 + 2 solve buffers)."""
        if self.XtX.is_cuda:
            h = torch.empty(self.XtX.shape, dtype=self.XtX.dtype, pin_memory=True)
            h.copy_(self.XtX)
            self.XtX = h
            torch.cuda.empty_cache()

    def _to_dev(self):
        if not self.XtX.is_cuda:
            self.XtX = self.XtX.to(self.dev)

    def fit(self, metric, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), groups=None):
        """metric(pred, Y) -> higher is better, evaluated on validation rows.
        groups (BIND_PATCH): list of column slices, each gets its own alpha (score = first group's)."""
        if groups and self.vX:
            self.flush()
            self._to_host()
            vX, vY = torch.cat(self.vX), torch.cat(self.vY)
            best = [(-1e9, None) for _ in groups]
            Wout = None
            for a in alphas:
                W = self._solve(a)
                if W is None:
                    continue
                P = self.predict(vX, W)
                if Wout is None:
                    Wout = W.clone()
                for gi, g in enumerate(groups):
                    sc = metric(P[:, g], vY[:, g])
                    if sc > best[gi][0]:
                        best[gi] = (sc, a)
                        Wout[:, g] = W[:, g]
                del W, P
            self.val_score, self.alpha, self.W = best[0][0], best[0][1], Wout
            return best[0][0]
        self.flush()
        self._to_host()
        vX = torch.cat(self.vX) if self.vX else None
        vY = torch.cat(self.vY) if self.vY else None
        best = (-1e9, None, None)
        for a in alphas:
            W = self._solve(a)
            if W is None:
                continue
            sc = metric(self.predict(vX, W), vY) if vX is not None else -a
            if sc > best[0]:
                best = (sc, a, W)
        self.val_score, self.alpha, self.W = best
        return best[0]


def acc_metric(P, Y):
    return (P.argmax(1) == Y.argmax(1)).float().mean().item()


def multilabel_metric(P, Y):
    pb, yb = P > 0.5, Y > 0.5
    tp = (pb & yb).sum().item(); fp = (pb & ~yb).sum().item(); fn = (~pb & yb).sum().item()
    return 2 * tp / max(1, 2 * tp + fp + fn)


def auc_metric(P, Y):
    """ROC-AUC over all (row, output) entries: how well positives outrank negatives (selection is by ranking)."""
    p, y = P.flatten().float(), (Y.flatten() > 0.5)
    npos = int(y.sum())
    if npos == 0 or npos == len(y):
        return 0.0
    r = torch.empty_like(p)
    r[p.argsort()] = torch.arange(1, len(p) + 1, device=p.device, dtype=p.dtype)
    return ((r[y].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))).item()
