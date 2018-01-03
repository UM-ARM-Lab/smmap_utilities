#include "smmap_utilities/gurobi_solvers.h"
#include <gurobi_c++.h>
#include <iostream>
#include <mutex>
#include <Eigen/Eigenvalues>
#include <arc_utilities/eigen_helpers.hpp>

using namespace Eigen;

static std::mutex gurobi_env_construct_mtx;

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


GRBLinExpr linearSum(const MatrixXd& row_coeff, GRBVar* vars)
{
    GRBLinExpr linear_sum = 0;
    ssize_t num_rows = row_coeff.cols();
    for (ssize_t ind = 0; ind < num_rows; ind++)
    {
        linear_sum += row_coeff(ind) * vars[ind];
    }
    return linear_sum;
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

Eigen::VectorXd smmap_utilities::minSquaredNormSE3VelocityConstraints(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const double max_se3_velocity, const Eigen::VectorXd& weights)
{
    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t num_vars = A.cols();
        assert(num_vars % 6 == 0);

        const std::vector<double> lb(num_vars, -max_se3_velocity);
        const std::vector<double> ub(num_vars, max_se3_velocity);

        // TODO: Find a way to put a scoped lock here
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
        model.update();

        // Add the SE3 velocity constraints
        for (ssize_t i = 0; i < num_vars / 6; ++i)
        {
            model.addQConstr(normSquared(&vars[i * 6], 6), GRB_LESS_EQUAL, max_se3_velocity * max_se3_velocity);
        }



        // Build up the matrix expressions
        // min || A x - b ||^2_W is the same as min x^T A^T W A x - 2 b^T W A x = x^T Q x + L x
        Eigen::MatrixXd Q = A.transpose() * weights.asDiagonal() * A;
        const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
        if (min_eigenvalue < 1.1e-4)
        {
            Q += Eigen::MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
        }

        const Eigen::RowVectorXd L = -2.0 * b.transpose() * weights.asDiagonal() * A;

        GRBQuadExpr objective_fn = buildQuadraticTerm(vars, vars, Q);
        objective_fn.addTerms(L.data(), vars, (int)num_vars);
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

// Least Square with 1-norm regularization
Eigen::VectorXd smmap_utilities::minSquaredNormL1NormRegularization(
        const Eigen::MatrixXd& A,
        const Eigen::VectorXd& b,
        const Eigen::VectorXd& x_last,
        const Eigen::VectorXd& weights,
        const double c_regularization,
        const double max_coordinate)
{
    /*
    //
    // min L = || b - Ax ||^2 + c |x_last - x|
    // r = |x_last - x|
    // beta = [ x^T r^T]^T
    // W = diag( weights ) (W is a diagnol matrix)
    //
    // L = (b - Ax)^T W (b - Ax) + c 1^T r
    //   = x^T A^T W A x - 2 b^T W A x + b^T W b + c 1^T r
    // Q = [A^T W A, 0]
    //     [0      , 0];
    // K1 = [-2 A^T W^T b]
    //      [ 0         ];
    // K2 = [ 0   ]
    //      [ c 1 ];
    // L = beta^T Q beta + (K1 + K2)^T beta + b^T W b
    //
    // (x_last - x) .<= r;   (x - x_last) .<= r;
    // C = [ -I   -I ]
    //     [  I   -I ];
    // l = [ -x_last ]
    //     [  x_last ];
    // C beta <= l
    //
    */
    (void) max_coordinate;

    const ssize_t dim_x = A.cols();
    const ssize_t dim_r = dim_x;
    const ssize_t num_vars = dim_x + dim_r;
    assert(A.rows() == weights.rows());

    VectorXd beta = Eigen::MatrixXd::Zero(num_vars, 1);
    GRBVar* vars = nullptr;

    try
    {
        // TODO: Find a way to put a scoped lock here. What's the functionality of this block of code? (to ".update()")
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);

        //vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
        vars = model.addVars(nullptr, nullptr, nullptr, nullptr, nullptr, (int)num_vars);
        model.update();

        // Quadratic term
        MatrixXd Q = MatrixXd::Zero(num_vars, num_vars);
        Q.topLeftCorner(dim_x, dim_x) = A.transpose() * weights.asDiagonal() * A;

        // Linear term
        Eigen::VectorXd K1 = Eigen::MatrixXd::Zero(num_vars, 1);
        K1.head(dim_x) = -2 * A.transpose() * weights.asDiagonal() * b;
        Eigen::VectorXd K2 = Eigen::MatrixXd::Zero(num_vars, 1);
        K2.tail(dim_r) = c_regularization * (Eigen::MatrixXd::Ones(dim_r, 1));
        const Eigen::VectorXd K = K1 + K2;

        // Constant term
        const double L0 = b.transpose() * weights.asDiagonal() * b;

        // Constraint
        const Eigen::MatrixXd I_dim_x = Eigen::MatrixXd::Identity(dim_x, dim_x);
        //const Eigen::MatrixXd I_dim_r = Eigen::MatrixXd::Identity(dim_r, dim_r);
        Eigen::MatrixXd C = -Eigen::MatrixXd::Identity(num_vars, num_vars);
        C.topRightCorner(dim_r, dim_r) = -I_dim_x;
        C.bottomLeftCorner(dim_x, dim_x) = I_dim_x;
        Eigen::VectorXd l = Eigen::MatrixXd::Zero(dim_x * 2, 1);
        l.head(dim_x) = -x_last;
        l.tail(dim_x) = x_last;

        for (ssize_t ind = 0; ind < num_vars; ind++)
        {
            model.addConstr(linearSum(C.row(ind), vars), GRB_LESS_EQUAL, l(ind));
        }

        // TODO: Why Dale has these step here?
        const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
        if (min_eigenvalue < 1.1e-4)
        {
            Q += Eigen::MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
        }        

        GRBQuadExpr objective_fn = buildQuadraticTerm(vars, vars, Q);
        objective_fn.addTerms(K.data(), vars, (int)num_vars);
        objective_fn.addConstant(L0);
        model.setObjective(objective_fn, GRB_MINIMIZE);

        model.update();
        model.optimize();

        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        {
            beta.resize(num_vars);
            for (ssize_t var_ind = 0; var_ind < num_vars; var_ind++)
            {
                beta(var_ind) = vars[var_ind].get(GRB_DoubleAttr_X);
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
    return beta.head(dim_x);
}

