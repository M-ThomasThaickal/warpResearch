import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

# ??? CONFUSED: im not really sure what tau means here, or what this does
# Answer: tau is stress test function, this connect displacement change with the change of stress in the system
@fem.integrand
def displacement_gradient_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
):
    """grad(u) : tau"""
    return wp.ddot(tau(s) fem.grad(u, s))

# ??? QUESTION: from what it understand, i dont need to make this a vec3 actually
# answer: this is correct
# ??? QUESTION: what is this? Linear elasticity is lame paramaters, this modifies it for nonlinear materials
@wp.func
def nh_parameters_from_lame(lame: wp.vec2):
    """Parameters such that for small strains model behaves according to Hooke's Law"""
    mu_nh = lame[1]
    lambda_nh = lame[0] + lame[1]

    return mu_nh, lambda_nh

# I THINK THIS MEANS: finding the stress coefficient?
# correction: this computes the actual stress based on cur deformation
@fem.integrand
def nh_stress_form(
    s: fem.Sample,
    tau: fem.Field,
    u_cur: fem.Field,
    lame: wp.vec2
):
    """d Psi/dF : tau"""

    # Deformation gradient
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)

    # Area term and its derivative w.r.t F
    J = wp.determinant(F)
    dJ_dF = cofactor_matrix_3d(F)

    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    nh_stress = mu_nh * F + (lambda_nh * (J - 1.0) - mu_nh) * dJ_dF

    return wp.ddot(tau(s), nh_stress)

@wp.func
def cofactor_matrix_3d(F: wp.mat33):
    """Computing dJ_dF"""
    c00 = F[1,1] * F[2, 2] - F[1,2] * F[2,1]
    c01 = -(F[1,0] * F[2,2] - F[1,2] * F[2,0])
    c02 = F[1,0] * F[2,1] - F[1,1] * F[2,0]

    c10 = -(F[0,1] * F[2,2] - F[0,2] * F[2,1])
    c11 = F[0,0] * F[2,2] - F[0,2] * F[2,0]
    c12 = -(F[0,0] * F[2,1] - F[0,1] * F[2,0])

    c20 = F[0,1] * F[1,2] - F[0,2] * F[1,1]
    c21 = -(F[0,0] * F[1,2] - F[0,2] * F[1,0])
    c22 = F[0,0] * F[1,1] - F[0,1] * F[1,0]

    return wp.mat33(
        c00, c01, c02,
        c10, c11, c12,
        c20, c21, c22
    )

# ??? QUESTION: does lame need to be vec3 here? no lame is always vec2, its just holding two values
# I THINK THIS MEANS: change in stress coefficient as deformation is applied
# correction: this computes the tangent stiffness, how stress changes when a small displacement increment (u) is applied
@fem.integrand
def nh_stress_delta_form(
    s: fem.Sample,
    tau: fem.Field,
    u: fem.Field,
    u_cur: fem.Field,
    lame: wp.vec2
):
    """grad(u) : d2 Psi/dF2 : tau"""

    tau_s = tau(s)
    sigma_s = fem.grad(u, s)

    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    dJ_dF = cofactor_matrix_3d(F)

    # Gauss--Newton approximation; ignore d2J,dF2 term
    # ??? QUESTION: What is the Gauss-Newton approximation?
    # answer: think approximating curve with a straight line instead of full curvature, full Newton method requires second derivative which is expensive to compute, this only keeps first order for compute efficiency
    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    return mu_nh * wp.ddot(tau_s, sigma_s) + lambda_nh * wp.ddot(dJ_dF, tau_s) * wp.ddot(dJ_dF, sigma_s)

# creates a projection operator that only acts on vertical boundaries
@fem.integrand
def vertical_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # non zero on vertical boundary of domain only
    nor = fem.normal(domain, s)
    # ??? QUESTION: whats the difference between ddot and dot? is it the derivative of the dot?
    # answer: double dot product, used for matrices and tensors
    return wp.dot(u(s), v(s)) * wp.abs(nor[0])

@fem.integrand
def vertical_displacement_form(
    s: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    displacement: float,
):
    # opposed to normal on vertical boundary of domain only
    nor = fem.normal(domain,s)
    # ??? QUESTION: So is this the calculation of deformation just on the z?
    # answer: no, its displacement on vertical **boundaries** so its actually x-direction faces
    return -wp.abs(nor[0]) * displacement * wp.dot(nor, v(s))

# ??? CONFUSED: on what this is
# it is the stress matrix, used in the solver
@fem.integrand
def tensor_mass_form(
    s: fem.Sample,
    sig: fem.Field,
    tau: fem.Field,
):
    return wp.ddot(tau(s), sig(s))

# ??? CONFUSED: on what this is
# computes the volume change ratio, used to descipt materials incompressibility
def area_form(s: fem.Sample, u_cur: fem.Field):
    F = wp.identity(n=3, dtype=float) + fem.grad(u_cur, s)
    return wp.determinant(F)
