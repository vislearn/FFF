import torch

from geomstats.geometry.poincare_ball import PoincareBall, PoincareBallMetric
from geomstats.geometry.hyperboloid import Hyperboloid
import geomstats.backend as gs


class PoincareBall_(PoincareBall):
    """fixes vmap incompatibility of PoincareBall projection method."""

    def projection(self, point: torch.Tensor):
        l2_norm = gs.linalg.norm(point, axis=-1)
        projected_point = gs.einsum(
            "...j,...->...j", point * (1 - gs.atol), 1.0 / (l2_norm + gs.atol)
        )
        return torch.where(l2_norm[..., None] >= (1 - gs.atol), projected_point, point)

    def default_metric(self):
        return PoincareBallMetric_


class PoincareBallMetric_(PoincareBallMetric):
    """Adds metric_matrix_log_det method to PoincareBallMetric for fast computation of
    log determinant of metric matrix."""

    def metric_matrix_log_det(self, base_point: torch.Tensor):
        lambda_base = 2 / (1 - gs.sum(base_point * base_point, axis=-1))
        return 2 * self._space.dim * gs.log(lambda_base)

    def exp0_with_jac_log_det(self, tangent_vec):
        origin = gs.zeros_like(tangent_vec[0, ...])
        point = self.exp(tangent_vec, base_point=origin)
        norm_tangent_vec = gs.linalg.norm(tangent_vec, axis=-1)
        jac_log_det = torch.log(
            gs.tanh(norm_tangent_vec)
            / (norm_tangent_vec * gs.cosh(norm_tangent_vec) ** 2)
        )
        return point, jac_log_det


class Hyperboloid_(Hyperboloid):
    """fixes vmap incompatibility of Hyperboloid regularize method."""

    def regularize(self, point):
        sq_norm = self.embedding_space.metric.squared_norm(point)
        real_norm = gs.sqrt(gs.abs(sq_norm))
        return gs.einsum("...i,...->...i", point, 1.0 / real_norm)
