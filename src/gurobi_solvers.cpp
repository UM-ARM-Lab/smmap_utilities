#include "smmap_utilities/gurobi_solvers.h"
#include <gurobi_c++.h>
#include <iostream>
#include <mutex>
#include <Eigen/Eigenvalues>
#include <arc_utilities/eigen_helpers.hpp>

using namespace Eigen;
using namespace EigenHelpers;

////////////////////////////////////////////////////////////////
// Internally used functions and static objects
////////////////////////////////////////////////////////////////

static std::mutex gurobi_env_construct_mtx;

// TODO: this loop is highly inefficient, this ought to be doable in a better way
GRBQuadExpr buildQuadraticTerm(GRBVar* left_vars, GRBVar* right_vars, const MatrixXd& Q)
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

// Builds the quadratic term ||point_a - point_b||^2
// This is equivalent to [point_a' point_b'] * Q * [point_a' point_b']'
// where Q is [ I, -I
//             -I,  I]
GRBQuadExpr buildDifferencingQuadraticTerm(GRBVar* point_a, GRBVar* point_b, const size_t num_vars_per_point)
{
    GRBQuadExpr expr;

    // Build the main diagonal
    const std::vector<double> main_diag(num_vars_per_point, 1.0);
    expr.addTerms(main_diag.data(), point_a, point_a, (int)num_vars_per_point);
    expr.addTerms(main_diag.data(), point_b, point_b, (int)num_vars_per_point);

    // Build the off diagonal - use -2 instead of -1 because the off diagonal terms are the same
    const std::vector<double> off_diagonal(num_vars_per_point, -2.0);
    expr.addTerms(off_diagonal.data(), point_a, point_b, (int)num_vars_per_point);

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

////////////////////////////////////////////////////////////////
// Externally accessed functions
////////////////////////////////////////////////////////////////

// Minimizes || Ax - b || subject to norm constraints on x
VectorXd smmap_utilities::minSquaredNorm(
        const MatrixXd& A,
        const VectorXd& b,
        const double max_x_norm)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        // Add the vars to the model
        {
            const std::vector<double> lb(num_vars, -max_x_norm);
            const std::vector<double> ub(num_vars, max_x_norm);
            vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            model.update();
        }

        // Add the x norm constraint
        {
            model.addQConstr(normSquared(vars, num_vars), GRB_LESS_EQUAL, max_x_norm * max_x_norm);
            model.update();
        }

        // Build the objective function
        {
            // Build up the matrix expressions
            // min || A x - b ||^2 is the same as min x^T A^T A x - 2 b^T A x = x^T Q x + L x
            MatrixXd Q = A.transpose() * A;
            // Gurobi requires a minimum eigenvalue for the problem, so if the given problem does
            // not have sufficient eigenvalues, make them have such
            const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
            if (min_eigenvalue <= 1.1e-4)
            {
                Q += MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
                std::cout << "Poorly conditioned matrix for Gurobi, adding conditioning." << std::endl;
            }

            const RowVectorXd L = -2.0 * b.transpose() * A;

            GRBQuadExpr objective_fn = buildQuadraticTerm(vars, vars, Q);
            objective_fn.addTerms(L.data(), vars, (int)num_vars);
            model.setObjective(objective_fn, GRB_MINIMIZE);
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

// Minimizes || Ax - b ||_w subject to norm constraints on x
VectorXd smmap_utilities::minSquaredNorm(
        const MatrixXd& A,
        const VectorXd& b,
        const double max_x_norm,
        const VectorXd& weights)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        // Add the vars to the model
        {
            const std::vector<double> lb(num_vars, -max_x_norm);
            const std::vector<double> ub(num_vars, max_x_norm);
            vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            model.update();
        }

        // Add the x norm constraint
        {
            model.addQConstr(normSquared(vars, num_vars), GRB_LESS_EQUAL, max_x_norm * max_x_norm);
            model.update();
        }

        // Build the objective function
        {
            // Build up the matrix expressions
            // min || A x - b ||^2_w is the same as min x^T A^T diag(w) A x - 2 b^T * diag(w) * A x = x^T Q x + L x
            MatrixXd Q = A.transpose() * weights.asDiagonal() * A;
            // Gurobi requires a minimum eigenvalue for the problem, so if the given problem does
            // not have sufficient eigenvalues, make them have such
            const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
            if (min_eigenvalue <= 1.1e-4)
            {
                Q += MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
                std::cout << "Poorly conditioned matrix for Gurobi, adding conditioning." << std::endl;
            }

            const RowVectorXd L = -2.0 * b.transpose() * weights.asDiagonal() * A;

            GRBQuadExpr objective_fn = buildQuadraticTerm(vars, vars, Q);
            objective_fn.addTerms(L.data(), vars, (int)num_vars);
            model.setObjective(objective_fn, GRB_MINIMIZE);
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

// Minimizes || Ax - b ||_w subject to norm constraints on x, and linear constraints.
// Linear constraint terms are of the form C * x <= d
//
// If lower bound is passed, upper bound must also be passed.
VectorXd smmap_utilities::minSquaredNormLinearConstraints(
        const MatrixXd& A,
        const VectorXd& b,
        const double max_x_norm,
        const VectorXd& weights,
        const std::vector<RowVectorXd>& linear_constraint_linear_terms,
        const std::vector<double>& linear_constraint_affine_terms,
        const VectorXd& x_lower_bound,
        const VectorXd& x_upper_bound)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        // Add the vars to the model
        {
            if (x_lower_bound.size() > 0)
            {
                assert(x_lower_bound.size() == num_vars);
                assert(x_lower_bound.size() == x_upper_bound.size());
                vars = model.addVars(x_lower_bound.data(), x_upper_bound.data(), nullptr, nullptr, nullptr, (int)num_vars);
            }
            else
            {
                const std::vector<double> lb(num_vars, -max_x_norm);
                const std::vector<double> ub(num_vars, max_x_norm);
                vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            }
            model.update();
        }

        // Add the x norm constraint
        {
            model.addQConstr(normSquared(vars, num_vars), GRB_LESS_EQUAL, max_x_norm * max_x_norm);
            model.update();
        }

        // Add the linear constraint terms
        {
            assert(linear_constraint_linear_terms.size() == linear_constraint_affine_terms.size());
            const size_t num_linear_constraints = linear_constraint_linear_terms.size();
            for (size_t ind = 0; ind < num_linear_constraints; ++ind)
            {
                assert(linear_constraint_linear_terms[ind].size() == num_vars);
                GRBLinExpr expr(0.0);
                expr.addTerms(linear_constraint_linear_terms[ind].data(), vars, (int)num_vars);
                model.addConstr(expr <= linear_constraint_affine_terms[ind]);
            }
            model.update();
        }

        // Build the objective function
        {
            // Build up the matrix expressions
            // min || A x - b ||^2_w is the same as min x^T A^T diag(w) A x - 2 b^T * diag(w) * A x = x^T Q x + L x
            MatrixXd Q = A.transpose() * weights.asDiagonal() * A;
            // Gurobi requires a minimum eigenvalue for the problem, so if the given problem does
            // not have sufficient eigenvalues, make them have such
            const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
            if (min_eigenvalue <= 1.1e-4)
            {
                Q += MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
                std::cout << "Poorly conditioned matrix for Gurobi, adding conditioning." << std::endl;
            }

            const RowVectorXd L = -2.0 * b.transpose() * weights.asDiagonal() * A;

            GRBQuadExpr objective_fn = buildQuadraticTerm(vars, vars, Q);
            objective_fn.addTerms(L.data(), vars, (int)num_vars);
            model.setObjective(objective_fn, GRB_MINIMIZE);
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

// Minimizes || Ax - b ||_w subject to SE3 velocity constraints on x
VectorXd smmap_utilities::minSquaredNormSE3VelocityConstraints(
        const MatrixXd& A,
        const VectorXd& b,
        const double max_se3_velocity,
        const VectorXd& weights)
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
            MatrixXd Q = A.transpose() * weights.asDiagonal() * A;
            // Gurobi requires a minimum eigenvalue for the problem, so if the given problem does
            // not have sufficient eigenvalues, make them have such
            const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
            if (min_eigenvalue <= 1.1e-4)
            {
                Q += MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
                std::cout << "Poorly conditioned matrix for Gurobi, adding conditioning." << std::endl;
            }

            const RowVectorXd L = -2.0 * b.transpose() * weights.asDiagonal() * A;

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

// Minimizes sum { obs_strength(i) * ||x(i) - obxervation(i)|| }
// subject to      distance_scale * ||x(i) - x(j)||^2 <= distance(i,j)^2 for all i,j
//
// This is custom designed for R^3 distances, but it could be done generically
//
// Variable bound is an extra contraint on each individual variable (not vector), it defines the upper and lower bound
VectorVector3d smmap_utilities::denoiseWithDistanceConstraints(
        const VectorVector3d& observations,
        const VectorXd& observation_strength,
        const MatrixXd& distance_sq_constraints,
        const double variable_bound)
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
            // Note that variable bound is important, without a bound, Gurobi defaults to 0, which is clearly unwanted
            const std::vector<double> lb(num_vars, -variable_bound);
            const std::vector<double> ub(num_vars, variable_bound);
            vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            model.update();
        }

        // Add the distance constraints
        {
            for (ssize_t i = 0; i < num_vectors; ++i)
            {
                for (ssize_t j = i + 1; j < num_vectors; ++j)
                {
                    model.addQConstr(
                                buildDifferencingQuadraticTerm(&vars[i * 3], &vars[j * 3], 3),
                                GRB_LESS_EQUAL,
                                distance_sq_constraints(i, j),
                                "distance_sq_" + std::to_string(i) + std::to_string(j));
                }
            }
            model.update();
        }

        // Build the objective function
        {
            // TODO: building this function icrementally is naive, and could be done faster
            GRBQuadExpr objective_fn;
            for (ssize_t i = 0; i < num_vectors; ++i)
            {
                // min w * || x - z ||^2 is the same as min w x^T x - 2 w z^T x = x^T Q x + L x
                const Matrix3d Q = observation_strength(i) * Matrix3d::Identity();
                const Vector3d L = - 2.0 * observation_strength(i) * observations[i];
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


// Designed to find a feasbible point for problems of the form:
// Minimize     f(x)
// subject to   linear * x + affine <= 0
//              lb <= x
//                    x <= ub
//              || x || <= max_norm
//
// Returns the x that minimizes constraint violations, and the minimim constraint violation itself
std::pair<Eigen::VectorXd, double> smmap_utilities::minimizeConstraintViolations(
        const ssize_t num_vars,
        const std::vector<Eigen::RowVectorXd>& linear_constraint_linear_terms,
        const std::vector<double>& linear_constraint_affine_terms,
        const double max_x_norm,
        const Eigen::VectorXd& x_lower_bound,
        const Eigen::VectorXd& x_upper_bound,
        const double constraint_lower_bound,
        const double constraint_upper_bound)
{
    VectorXd x;
    double c_vio = std::numeric_limits<double>::quiet_NaN();
    GRBVar* x_vars = nullptr;
    GRBVar c_violation_var;
    try
    {
        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 1);
        GRBModel model(env);

        // Add the vars to the model
        {
            if (x_lower_bound.size() > 0)
            {
                assert(x_lower_bound.size() == num_vars);
                assert(x_lower_bound.size() == x_upper_bound.size());
                x_vars = model.addVars(x_lower_bound.data(), x_upper_bound.data(), nullptr, nullptr, nullptr, (int)num_vars);
            }
            else
            {
                const std::vector<double> lb(num_vars, -max_x_norm);
                const std::vector<double> ub(num_vars, max_x_norm);
                x_vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            }
            const double coeff = 1.0;
            c_violation_var = model.addVar(constraint_lower_bound, constraint_upper_bound, coeff, GRB_CONTINUOUS);

            model.update();
        }

        // Add the x norm constraint
        {
            GRBQuadExpr constr;
            constr += x_vars[0] * x_vars[0];
            constr += x_vars[1] * x_vars[1];
            constr += x_vars[2] * x_vars[2];
            constr += x_vars[3] * x_vars[3] / 400.0;
            constr += x_vars[4] * x_vars[4] / 400.0;
            constr += x_vars[5] * x_vars[5] / 400.0;
            model.addQConstr(constr <= max_x_norm * max_x_norm);
            model.update();
        }

        // Add the linear constraint terms
        {
            assert(linear_constraint_linear_terms.size() == linear_constraint_affine_terms.size());
            const size_t num_linear_constraints = linear_constraint_linear_terms.size();
            for (size_t ind = 0; ind < num_linear_constraints; ++ind)
            {
                assert(linear_constraint_linear_terms[ind].size() == num_vars);
                GRBLinExpr lhs(0.0);
                lhs.addTerms(linear_constraint_linear_terms[ind].data(), x_vars, (int)num_vars);
                model.addConstr(lhs + linear_constraint_affine_terms[ind] <= c_violation_var);
            }
        }

        // No need to add an objective, it's already set when we created c_violation_var
//        std::cout << "Objective: " << model.getObjective() << std::endl;
//        std::cout << model << std::endl;
//        model.write("/home/dmcconac/test.lp");

        // Find the optimal solution and extract it
        {
            model.optimize();
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
            {
                x.resize(num_vars);
                for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
                {
                    x(var_ind) = x_vars[var_ind].get(GRB_DoubleAttr_X);
                }
                c_vio = c_violation_var.get(GRB_DoubleAttr_X);
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

    delete[] x_vars;
    return {x, c_vio};
}


// Designed to find a feasbible point for problems of the form:
// Minimize     || x - starting_point ||
// subject to   linear * x + affine <= 0
//              lb <= x
//                    x <= ub
//              || x || <= max_norm
Eigen::VectorXd smmap_utilities::findClosestValidPoint(
        const Eigen::VectorXd& starting_point,
        const std::vector<Eigen::RowVectorXd>& linear_constraint_linear_terms,
        const std::vector<double>& linear_constraint_affine_terms,
        const double max_x_norm,
        const Eigen::VectorXd& x_lower_bound,
        const Eigen::VectorXd& x_upper_bound)
{
    VectorXd x;
    GRBVar* x_vars = nullptr;
    try
    {
        const ssize_t num_vars = starting_point.size();

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        // Disables logging to file and logging to console (with a 0 as the value of the flag)
        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        // Add the vars to the model
        {
            if (x_lower_bound.size() > 0)
            {
                assert(x_lower_bound.size() == num_vars);
                assert(x_lower_bound.size() == x_upper_bound.size());
                x_vars = model.addVars(x_lower_bound.data(), x_upper_bound.data(), nullptr, nullptr, nullptr, (int)num_vars);
            }
            else
            {
                const std::vector<double> lb(num_vars, -max_x_norm);
                const std::vector<double> ub(num_vars, max_x_norm);
                x_vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
            }
            model.update();
        }

        // Add the x norm constraint
        {
            model.addQConstr(normSquared(x_vars, num_vars), GRB_LESS_EQUAL, max_x_norm * max_x_norm);
            model.update();
        }

        // Add the linear constraint terms
        {
            assert(linear_constraint_linear_terms.size() == linear_constraint_affine_terms.size());
            const size_t num_linear_constraints = linear_constraint_linear_terms.size();
            for (size_t ind = 0; ind < num_linear_constraints; ++ind)
            {
                assert(linear_constraint_linear_terms[ind].size() == num_vars);
                GRBLinExpr lhs = 0.0;
                lhs.addTerms(linear_constraint_linear_terms[ind].data(), x_vars, (int)num_vars);
                model.addConstr(lhs + linear_constraint_affine_terms[ind] <= 0.0);
            }
        }

        // Build the objective function
        {
            GRBQuadExpr objective_fn = 0;
            for (ssize_t ind = 0; ind < num_vars; ++ind)
            {
                objective_fn += (x_vars[ind] - starting_point(ind)) * (x_vars[ind] - starting_point(ind));
            }
            model.setObjective(objective_fn, GRB_MINIMIZE);
        }

        // Find the optimal solution and extract it
        {
            model.optimize();
            if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
            {
                x.resize(num_vars);
                for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
                {
                    x(var_ind) = x_vars[var_ind].get(GRB_DoubleAttr_X);
                }
            }
            else
            {
                std::cout << "Optimal solution not found" << std::endl;
                std::cout << "Status: " << model.get(GRB_IntAttr_Status) << std::endl;
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

    delete[] x_vars;
    return x;
}
