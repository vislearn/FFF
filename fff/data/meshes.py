import os
from enum import Enum

import igl
import numpy as np
import scipy
import torch
from geomstats.geometry.riemannian_metric import RiemannianMetric

from fff.data.manifold import ManifoldDataset
from geomstats.geometry.manifold import Manifold
from torch.utils.data import Dataset
import geomstats.backend as gs


class Metric(Enum):
    DIFFUSION = "diffusion"
    BIHARMONIC = "biharmonic"
    COMMUTETIME = "commutetime"
    HEAT = "heat"


def csr_row_set_nz_to_val(csr, row, value=0):
    """Set all nonzero elements (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.
    """
    if not isinstance(csr, scipy.sparse.csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    csr.data[csr.indptr[row]: csr.indptr[row + 1]] = value


def csr_rows_set_nz_to_val(csr, rows, value=0):
    for row in rows:
        csr_row_set_nz_to_val(csr, row)
    if value == 0:
        csr.eliminate_zeros()


def face_normal(a, b, c):
    """Computes face normal based on three vertices. Ordering matters.

    Inputs:
        a, b, c: (N, 3)
    """
    u = b - a
    v = c - a
    n = torch.linalg.cross(u, v)
    n = n / torch.linalg.norm(n, dim=-1, keepdim=True)
    return n


def project_edge(p, a, b):
    x = p - a
    v = b - a
    r = torch.sum(x * v, dim=-1, keepdim=True) / torch.sum(v * v, dim=-1, keepdim=True)
    r = r.clamp(max=1.0, min=0.0)
    projx = v * r
    return projx + a


def closest_point(p, v, f):
    """Returns the point on the mesh closest to the query point p.
    Algorithm follows https://www.youtube.com/watch?v=9MPr_XcLQuw&t=204s.

    Inputs:
        p : (#query, 3)
        v : (#vertices, 3)
        f : (#faces, 3)

    Return:
        A projected tensor of size (#query, 3) and an index (#query,) indicating the closest triangle.
    """

    orig_p = p

    nq = p.shape[0]
    nf = f.shape[0]

    vs = v[f]
    a, b, c = vs[:, 0], vs[:, 1], vs[:, 2]

    # calculate normal of each triangle
    n = face_normal(a, b, c)

    n = n.reshape(1, nf, 3)
    p = p.reshape(nq, 1, 3)

    a = a.reshape(1, nf, 3)
    b = b.reshape(1, nf, 3)
    c = c.reshape(1, nf, 3)

    # project onto the plane of each triangle
    p = p + (n * (a - p)).sum(-1, keepdim=True) * n

    # if barycenter coordinate is negative,
    # then point is outside of the edge on the opposite side of the vertex.
    bc = barycenter_coordinates(p, a, b, c)

    # for each outside edge, project point onto edge.
    p = torch.where((bc[..., 0] < 0)[..., None], project_edge(p, b, c), p)
    p = torch.where((bc[..., 1] < 0)[..., None], project_edge(p, c, a), p)
    p = torch.where((bc[..., 2] < 0)[..., None], project_edge(p, a, b), p)

    # compute distance to all points and take the closest one
    idx = torch.argmin(torch.linalg.norm(orig_p[:, None] - p, dim=-1), dim=-1)
    p_idx = torch.func.vmap(lambda p_, idx_: torch.index_select(p_, 0, idx_))(
        p, idx.reshape(-1, 1)
    ).reshape(nq, 3)
    return p_idx, idx


def sample_simplex_uniform(K, shape=(), dtype=torch.float32, device="cpu"):
    x = torch.sort(torch.rand(shape + (K,), dtype=dtype, device=device))[0]
    x = torch.cat(
        [
            torch.zeros(*shape, 1, dtype=dtype, device=device),
            x,
            torch.ones(*shape, 1, dtype=dtype, device=device),
        ],
        dim=-1,
    )
    diffs = x[..., 1:] - x[..., :-1]
    return diffs


def barycenter_coordinates(p, a, b, c):
    """Assumes inputs are (N, D).
    Follows https://ceng2.ktu.edu.tr/~cakir/files/grafikler/Texture_Mapping.pdf
    """
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return torch.stack([u, v, w], dim=-1)


class MeshMetric(RiemannianMetric):
    pass


class Mesh(Manifold):
    ndim = 1
    name = "Mesh"
    reversible = False

    def __init__(
            self,
            v,
            f,
            numeigs=100,
            metric=Metric.COMMUTETIME,
            temp=1.0,
            dirichlet_bc: bool = False,
            upsample: int = 0,
    ):
        super().__init__(
            2, (3,), "extrinsic", True
        )

        if upsample > 0:
            v_np, f_np = v.cpu().numpy(), f.cpu().numpy()
            v_np, f_np = igl.upsample(v_np, f_np, upsample)
            v, f = torch.tensor(v_np).to(v), torch.tensor(f_np).to(f)

        v_np, f_np = v.cpu().numpy(), f.cpu().numpy()
        self.areas = torch.tensor(igl.doublearea(v_np, f_np)).reshape(-1) / 2

        self.v = v
        self.f = f

        print("#vertices: ", v.shape[0], "#faces: ", f.shape[0])

        self.numeigs = numeigs
        self.metric = metric
        self.temp = temp
        self.dirichlet_bc = dirichlet_bc

        self._preprocess_eigenfunctions()

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return MeshMetric

    def is_tangent(self, vector, base_point, atol=gs.atol):
        raise NotImplementedError

    def random_point(self, n_samples=1, bound=1.0):
        return self.random_uniform(n_samples)

    def to_tangent(self, vector, base_point):
        return self.proju(vector, base_point)

    def belongs(self, point, atol=gs.atol):
        raise NotImplementedError

    def _preprocess_eigenfunctions(self):
        assert (
                self.numeigs <= self.v.shape[0]
        ), "Cannot compute more eigenvalues than the number of vertices."

        ## ---- in numpy ----  ##
        v_np = self.v.detach().cpu().numpy()
        f_np = self.f.detach().cpu().numpy()

        M = igl.massmatrix(v_np, f_np, igl.MASSMATRIX_TYPE_VORONOI)
        L = -igl.cotmatrix(v_np, f_np)

        if self.dirichlet_bc:
            b = igl.boundary_facets(f_np)
            b = np.unique(b.flatten())
            L = scipy.sparse.csr_matrix(L)
            csr_rows_set_nz_to_val(L, b, 0)
            for i in b:
                L[i, i] = 1.0

        eigvals, eigfns = scipy.sparse.linalg.eigsh(
            L, self.numeigs + 1, M, sigma=0, which="LM", maxiter=100000
        )
        # Remove the zero eigenvalue.
        eigvals = eigvals[..., 1:]
        eigfns = eigfns[..., 1:]
        ## ---- end in numpy ----  ##

        self.eigvals = torch.tensor(eigvals).to(self.v)
        self.eigfns = torch.tensor(eigfns).to(self.v)

        print(
            "largest eigval: ",
            self.eigvals.max().item(),
            ", smallest eigval: ",
            self.eigvals.min().item(),
        )

    def projection(self, x):
        return self.projx(x)

    ## Below is copied from Ricky's code

    def projx(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = closest_point(x, self.v.to(x.device, non_blocking=True), self.f.to(x.device, non_blocking=True))
        return x

    def proju(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # determine which face the point is on
        _, f_idx = closest_point(x, self.v.to(x.device, non_blocking=True), self.f.to(x.device, non_blocking=True))
        vs = self.v.to(x.device, non_blocking=True)[self.f.to(x.device, non_blocking=True)[f_idx]]

        # compute normal for each face
        n = face_normal(a=vs[:, 0], b=vs[:, 1], c=vs[:, 2])

        # project by removing the normal direction
        u = u - (n * u).sum(-1, keepdim=True) * n
        return u

    def dist(self, x: torch.Tensor, y: torch.Tensor, squared: bool = False):
        eigfns_x = self.get_eigfns(x)
        eigfns_y = self.get_eigfns(y)

        fn = (lambda x: x) if squared else torch.sqrt

        if self.metric == Metric.BIHARMONIC:
            return fn(torch.sum(((eigfns_x - eigfns_y) / self.eigvals) ** 2, axis=1))
        elif self.metric == Metric.DIFFUSION:
            return fn(
                torch.sum(
                    torch.exp(-2 * self.temp * self.eigvals)
                    * (eigfns_x - eigfns_y) ** 2,
                    axis=1,
                )
            )
        elif self.metric == Metric.COMMUTETIME:
            return fn(torch.sum((eigfns_x - eigfns_y) ** 2 / self.eigvals, axis=1))
        elif self.metric == Metric.HEAT:
            k = torch.sum(
                eigfns_x * eigfns_y * torch.exp(-self.temp * self.eigvals), axis=1
            )
            dist = fn(-4 * self.temp * torch.log(k))
            return dist
        else:
            return ValueError(f"Unknown distance type option, metric={self.metric}.")

    def get_eigfns(self, x: torch.Tensor):
        """x, y : (N, 3) torch.Tensor representing points on the mesh."""

        N = x.shape[0]

        _, ix = closest_point(x, self.v, self.f)

        # compute barycentric coordinates
        vfx = self.v[self.f[ix]]
        vfx_a, vfx_b, vfx_c = vfx[..., 0, :], vfx[..., 1, :], vfx[..., 2, :]
        bc_x = barycenter_coordinates(x, vfx_a, vfx_b, vfx_c)[..., None]  # (N, 3, 1)

        # compute interpolated eigenfunction
        eigfns = torch.sum(bc_x * self.eigfns[self.f[ix]], dim=-2)

        return eigfns

    def random_uniform(self, n_samples, dim=3):
        assert dim == 3

        f_idx = torch.multinomial(self.areas, n_samples, replacement=True)
        barycoords = sample_simplex_uniform(
            2, (n_samples,), dtype=self.v.dtype, device=self.v.device
        )
        return torch.sum(self.v[self.f[f_idx]] * barycoords[..., None], axis=1)

    def uniform_logprob(self, x):
        tot_area = torch.sum(self.areas)
        return torch.full_like(x[..., 0], -torch.log(tot_area))

    def random_base(self, *args, **kwargs):
        return self.random_uniform(*args, **kwargs)

    def base_logprob(self, *args, **kwargs):
        return self.uniform_logprob(*args, **kwargs)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.proju(x, u)

    def _check_point_on_manifold(self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5):
        return True, None

    def _check_vector_on_tangent(
            self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5
    ):
        return True, None

    def expmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def logmap(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner(
            self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor = None, *, keepdim=False
    ) -> torch.Tensor:
        return torch.sum(u * v, dim=-1, keepdim=keepdim)

    def retr(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MeshDataset(Dataset):
    dim = 3

    def __init__(self, root: str, data_file: str, obj_file: str, scale=1 / 250):
        with open(os.path.join(root, data_file), "rb") as f:
            data = np.load(f)

        v, f = igl.read_triangle_mesh(os.path.join(root, obj_file))

        self.v = torch.tensor(v).float() * scale
        self.f = torch.tensor(f).long()
        self.data = torch.tensor(data).float() * scale

    def manifold(self, *args, **kwargs):
        return Mesh(self.v, self.f, *args, **kwargs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx],


def make_bunny_data(data_seed=0, **kwargs):
    dataset = MeshDataset(**kwargs)
    manifold = dataset.manifold()

    N = len(dataset)
    N_val = N_test = N // 10
    N_train = N - N_val - N_test

    if data_seed is None:
        raise ValueError("seed for data generation must be provided")
    datasets = torch.utils.data.random_split(
        dataset,
        [N_train, N_val, N_test],
        generator=torch.Generator().manual_seed(data_seed),
    )
    return [
        ManifoldDataset(dataset, manifold=manifold)
        for dataset in datasets
    ]
