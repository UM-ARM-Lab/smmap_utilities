#ifndef GUROBI_SOLVERS_H
#define GUROBI_SOLVERS_H

#include <Eigen/Dense>
#include <arc_utilities/eigen_helpers.hpp>

namespace smmap_utilities
{
    Eigen::VectorXd minSquaredNorm(
            const Eigen::MatrixXd& A,
            const Eigen::VectorXd& b,
            const double max_x_norm);

    Eigen::VectorXd minSquaredNorm(
            const Eigen::MatrixXd& A,
            const Eigen::VectorXd& b,
            const double max_x_norm,
            const Eigen::VectorXd& weights);

    Eigen::VectorXd minSquaredNormLinearConstraints(
            const Eigen::MatrixXd& A,
            const Eigen::VectorXd& b,
            const double max_x_norm,
            const Eigen::VectorXd& weights,
            const std::vector<Eigen::RowVectorXd>& linear_constraint_linear_terms,
            const std::vector<double>& linear_constraint_affine_terms);

    Eigen::VectorXd minSquaredNormSE3VelocityConstraints(
            const Eigen::MatrixXd& A,
            const Eigen::VectorXd& b,
            const double max_se3_velocity,
            const Eigen::VectorXd& weights);

    EigenHelpers::VectorVector3d denoiseWithDistanceConstraints(
            const EigenHelpers::VectorVector3d& observations,
            const Eigen::VectorXd& observation_strength,
            const Eigen::MatrixXd& distance_sq_constraints,
            const double variable_bound = 1000.0);
}

#endif // GUROBI_SOLVERS_H
