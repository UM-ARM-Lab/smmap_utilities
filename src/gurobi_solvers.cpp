#include "smmap_utilities/gurobi_solvers.h"
#include <gurobi_c++.h>
#include <iostream>
#include <mutex>
#include <Eigen/Eigenvalues>
#include <arc_utilities/eigen_helpers.hpp>

using namespace Eigen;
using namespace EigenHelpers;

static std::mutex gurobi_env_construct_mtx;

// TODO: this loop is highly inefficient, this ought to be doable in a better way
GRBQuadExpr buildQuadraticTerm(GRBVar* left_vars, GRBVar* right_vars, const Eigen::MatrixXd& Q)
{
    GRBQuadExpr expr;

    for (ssize_t right_ind = 0; right_ind < Q.rows(); ++right_ind)
    {
        for (ssize_t left_ind = 0; left_ind < Q.cols(); ++left_ind)
        {
            expr += left_vars[(size_t)left_ind] * Q(left_ind, right_ind) * right_vars[(size_t)right_ind];
        }
    }

    return expr;
}

GRBQuadExpr normSquared(const std::vector<GRBLinExpr>& exprs)
{
    GRBQuadExpr vector_norm_squared = 0;

    // TODO: replace with a single call to addTerms?
    for (size_t expr_ind = 0; expr_ind < exprs.size(); expr_ind++)
    {
        vector_norm_squared += exprs[expr_ind] * exprs[expr_ind];
    }

    return vector_norm_squared;
}

GRBQuadExpr normSquared(const std::vector<GRBLinExpr>& exprs, const VectorXd& weights)
{
    assert(exprs.size() == (size_t)weights.rows());
    GRBQuadExpr vector_norm_squared = 0;

    // TODO: replace with a single call to addTerms?
    for (size_t expr_ind = 0; expr_ind < exprs.size(); expr_ind++)
    {
        vector_norm_squared += weights((ssize_t)expr_ind) * exprs[expr_ind] * exprs[expr_ind];
    }

    return vector_norm_squared;
}

GRBQuadExpr normSquared(GRBVar* vars, const size_t num_vars)
{
    GRBQuadExpr vector_norm_squared = 0;

    // TODO: replace with a single call to addTerms?
    for (size_t var_ind = 0; var_ind < num_vars; var_ind++)
    {
        vector_norm_squared += vars[var_ind] * vars[var_ind];
    }

    return vector_norm_squared;
}

std::vector<GRBLinExpr> buildVectorOfExperssions(const MatrixXd& A, GRBVar* vars, const VectorXd& b)
{
    const ssize_t num_expr = A.rows();
    const ssize_t num_vars = A.cols();
    std::vector<GRBLinExpr> exprs(num_expr, 0);

    for (ssize_t expr_ind = 0; expr_ind < num_expr; expr_ind++)
    {
        for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
        {
            exprs[expr_ind] += A(expr_ind, var_ind) * vars[var_ind];
        }
        exprs[expr_ind] -= b(expr_ind);
    }

    return exprs;
}

VectorXd smmap_utilities::minSquaredNorm(const MatrixXd& A, const VectorXd& b, const double max_x_norm)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();
        const std::vector<double> lb(num_vars, -max_x_norm);
        const std::vector<double> ub(num_vars, max_x_norm);

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
        model.update();

        model.addQConstr(normSquared(vars, num_vars), GRB_LESS_EQUAL, max_x_norm * max_x_norm);
        model.setObjective(normSquared(buildVectorOfExperssions(A, vars, b)), GRB_MINIMIZE);
        model.update();
        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        {
            x.resize(num_vars);
            for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
            {
                x(var_ind) = vars[var_ind].get(GRB_DoubleAttr_X);
            }
        }
        else
        {
            std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
            exit(-1);
        }
    }
    catch(GRBException& e)
    {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception during optimization" << std::endl;
    }

    delete[] vars;
    return x;
}

VectorXd smmap_utilities::minSquaredNorm(const MatrixXd& A, const VectorXd& b, const double max_x_norm, const VectorXd& weights)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();
        const std::vector<double> lb(num_vars, -max_x_norm);
        const std::vector<double> ub(num_vars, max_x_norm);

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
        model.update();

        model.addQConstr(normSquared(vars, num_vars), GRB_LESS_EQUAL, max_x_norm * max_x_norm);

        GRBQuadExpr objective_fn = normSquared(buildVectorOfExperssions(A, vars, b), weights);
        // Check if we need to add anything extra to the main diagonal.
        const VectorXd eigenvalues = (A.transpose() * weights.asDiagonal() * A).selfadjointView<Upper>().eigenvalues();
        if ((eigenvalues.array() < 1.1e-4).any())
        {
//            const std::vector<double> diagonal(num_vars, 1.1e-4 - eigenvalues.minCoeff());
            const std::vector<double> diagonal(num_vars, 1.1e-4);
            objective_fn.addTerms(diagonal.data(), vars, vars, (int)num_vars);
        }
        model.setObjective(objective_fn, GRB_MINIMIZE);

        model.update();
        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        {
            x.resize(num_vars);
            for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
            {
                x(var_ind) = vars[var_ind].get(GRB_DoubleAttr_X);
            }
        }
        else
        {
            std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
            exit(-1);
        }
    }
    catch(GRBException& e)
    {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception during optimization" << std::endl;
    }

    delete[] vars;
    return x;
}

// Minimizes || Ax - b || subject to SE3 velocity constraints on x
Eigen::VectorXd smmap_utilities::minSquaredNormSE3VelocityConstraints(
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& b,
        const double max_se3_velocity,
        const Eigen::VectorXd& weights)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();
        // Make sure that we are still within reasonable limits
        assert(num_vars < (ssize_t)(std::numeric_limits<int>::max()));
        // Verify that the input data is of the correct size
        assert(num_vars % 6 == 0);
        assert(A.rows() == b.rows());
        assert(weights.rows() == b.rows());

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        // Add the vars to the model
        {
            const std::vector<double> lb(num_vars, -max_se3_velocity);
            const std::vector<double> ub(num_vars, max_se3_velocity);
            vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            model.update();
        }

        // Add the SE3 velocity constraints
        {
            for (ssize_t i = 0; i < num_vars / 6; ++i)
            {
                model.addQConstr(normSquared(&vars[i * 6], 6), GRB_LESS_EQUAL, max_se3_velocity * max_se3_velocity);
            }
            model.update();
        }

        // Build the objective function
        {
            // Build up the matrix expressions
            // min || A x - b ||^2_W is the same as min x^T A^T W A x - 2 b^T W A x = x^T Q x + L x
            Eigen::MatrixXd Q = A.transpose() * weights.asDiagonal() * A;
            // Gurobi requires a minimum eigenvalue for the problem, so if the given problem does
            // not have sufficient eigenvalues, make them have such
            const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
            if (min_eigenvalue <= 1.1e-4)
            {
                Q += Eigen::MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
                std::cout << "Poorly conditioned matrix for Gurobi, adding conditioning." << std::endl;
            }

            const Eigen::RowVectorXd L = -2.0 * b.transpose() * weights.asDiagonal() * A;

            GRBQuadExpr objective_fn = buildQuadraticTerm(vars, vars, Q);
            objective_fn.addTerms(L.data(), vars, (int)num_vars);
            model.setObjective(objective_fn, GRB_MINIMIZE);
            model.update();
        }

        // Find the optimal solution and extract it
        {
            model.optimize();
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
            {
                x.resize(num_vars);
                for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
                {
                    x(var_ind) = vars[var_ind].get(GRB_DoubleAttr_X);
                }
            }
            else
            {
                std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
                exit(-1);
            }
        }
    }
    catch(GRBException& e)
    {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception during optimization" << std::endl;
    }

    delete[] vars;
    return x;
}


// Minimizes sum (obs_strength(i) * ||x(i) - obxervation(i)||
// subject to      distance_scale * ||x(i) - x(j)||^2 < distance(i,j)^2
//
// This is custom designed for R^3 distances, but it could be done generically
EigenHelpers::VectorVector3d smmap_utilities::denoiseWithDistanceConstraints(
        const EigenHelpers::VectorVector3d& observations,
        const Eigen::VectorXd& observation_strength,
        const Eigen::MatrixXd& distance_sq_constraints)
{
    VectorVector3d x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vectors = (ssize_t)observations.size();
        const ssize_t num_vars = 3 * num_vectors;
        // Make sure that we are still within reasonable limits
        assert(num_vars < (ssize_t)(std::numeric_limits<int>::max()));
        // Verify that all the input data is of the correct size
        assert(observation_strength.rows() == num_vectors);
        assert(distance_sq_constraints.rows() == num_vectors);
        assert(distance_sq_constraints.cols() == num_vectors);

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        // Add the vars to the model
        {
            const std::vector<double> lb(num_vars, -100.0);
            const std::vector<double> ub(num_vars, 100.0);
            vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            model.update();
        }

        // Add the distance constraints
        {
            Matrix<double, 6, 6> Q = Matrix<double, 6, 6>::Identity();
            Q.bottomLeftCorner<3, 3>() = -Matrix3d::Identity();
            Q.topRightCorner<3, 3>() = -Matrix3d::Identity();
            for (ssize_t i = 0; i < num_vectors; ++i)
            {
                for (ssize_t j = i + 1; j < num_vectors; ++j)
                {
                    {
                        model.addQConstr(
                                    buildQuadraticTerm(&vars[i * 3], &vars[i * 3], Q),
                                    GRB_LESS_EQUAL,
                                    distance_sq_constraints(i, j),
                                    "distance_sq_" + std::to_string(i) + std::to_string(j));
                    }
                }
            }
            model.update();
        }

        // Build the objective function
        {
            // TODO: this is naive, and could be done faster
            // min w * || x - z ||^2 is the same as min w x^T x - 2 w z^T x = x^T Q x + L x
            GRBQuadExpr objective_fn;
            for (ssize_t i = 0; i < num_vectors; ++i)
            {
                const Eigen::Matrix3d Q = observation_strength(i) * Eigen::Matrix3d::Identity();
                const Eigen::Vector3d L = - 2.0 * observation_strength(i) * observations[i];
                objective_fn += buildQuadraticTerm(&vars[i * 3], &vars[i * 3], Q);
                objective_fn.addTerms(L.data(), &vars[i * 3], 3);
            }
            model.setObjective(objective_fn, GRB_MINIMIZE);
            model.update();
        }

        // Find the optimal solution, and extract it
        {
            model.optimize();
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
            {
                x.resize(num_vectors);
                for (ssize_t i = 0; i < num_vectors; i++)
                {
                    x[i](0) = vars[i * 3 + 0].get(GRB_DoubleAttr_X);
                    x[i](1) = vars[i * 3 + 1].get(GRB_DoubleAttr_X);
                    x[i](2) = vars[i * 3 + 2].get(GRB_DoubleAttr_X);
                }
            }
            else
            {
                std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
                exit(-1);
            }
        }

    }
    catch(GRBException& e)
    {
        std::cout << "Error code = " << e.getErrorCode() << std::endl;
        std::cout << e.getMessage() << std::endl;
    }
    catch(...)
    {
        std::cout << "Exception during optimization" << std::endl;
    }

    delete[] vars;
    return x;
}
