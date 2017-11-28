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

// TODO: Least Square with 1-norm regularization
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
    //     [  I    I ];
    // l = [ -x_last ]
    //     [  x_last ];
    // C beta <= l
    //
    */

    const ssize_t dim_x = A.cols();
    const ssize_t dim_r = dim_x;
    assert(dim_r == A.rows());
    const ssize_t num_vars = dim_x + dim_r;

    VectorXd beta = Eigen::MatrixXd::Zero(num_vars, 1);
    GRBVar* vars = nullptr;

    try
    {
        //const double max_coordinate = 1.0;
        Eigen::VectorXd eigen_lb = max_coordinate * (Eigen::MatrixXd::Ones(num_vars, 1));
        Eigen::VectorXd eigen_ub = -max_coordinate * (Eigen::MatrixXd::Ones(num_vars, 1));

        // residual is greater than 0
        eigen_lb.tail(dim_r) = -Eigen::MatrixXd::Zero(dim_r, 1);
        eigen_ub.tail(dim_r) = Eigen::MatrixXd::Ones(dim_r, 1) * INFINITY;

        const std::vector<double> lb = EigenHelpers::EigenVectorXdToStdVectorDouble(eigen_lb);
        const std::vector<double> ub = EigenHelpers::EigenVectorXdToStdVectorDouble(eigen_ub);;

        // TODO: Find a way to put a scoped lock here. What's the functionality of this block of code? (to ".update()")
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
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


        // Add the SE3 velocity constraints
        for (ssize_t i = 0; i < num_vars / 6; ++i)
        {
            model.addQConstr(normSquared(&vars[i * 6], 6), GRB_LESS_EQUAL, max_coordinate * max_coordinate);
        }

        /* // TODO: Why Dale has these step here?
        // Build up the matrix expressions
        // min || A x - b ||^2_W is the same as min x^T A^T W A x - 2 b^T W A x = x^T Q x + L x
        Eigen::MatrixXd Q = A.transpose() * weights.asDiagonal() * A;
        const double min_eigenvalue = Q.selfadjointView<Upper>().eigenvalues().minCoeff();
        if (min_eigenvalue < 1.1e-4)
        {
            Q += Eigen::MatrixXd::Identity(num_vars, num_vars) * (1.400001e-4 - min_eigenvalue);
        }
        */

        // const Eigen::RowVectorXd L = -2.0 * b.transpose() * weights.asDiagonal() * A;

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



// TODO: Return "infeasible". Need to figure out the reason
//Eigen::VectorXd smmap_utilities::minAbsoluteDeviation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y, const Eigen::VectorXd& weights)
Eigen::VectorXd smmap_utilities::minAbsoluteDeviation(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    // min ||y - X*beta||_1;    X in R ^ n*m   ===>
    // min(f^T * x); subject to A * x <= b
    // f = [ 0_m; 1_n]
    // A = [X, -I_n;  -X, -I_n]   [2n * (m+n)]
    // x = [beta; r] (vars)           [(n+m) * 1]
    // b = [y, -y]
    //(void) weights;

    VectorXd x;
    GRBVar* vars = nullptr;
    try
    {
        const ssize_t dim_beta = X.cols();
        const ssize_t dim_y = y.rows();
        assert(dim_y == X.rows());
        const ssize_t num_vars = dim_beta + dim_y;

        //const double max_coordinate = 1.0;
        Eigen::VectorXd eigen_lb = Eigen::MatrixXd::Ones(num_vars, 1);
        Eigen::VectorXd eigen_ub = -Eigen::MatrixXd::Ones(num_vars, 1);
        eigen_lb.tail(dim_y) = Eigen::MatrixXd::Zero(dim_y, 1);
        eigen_ub.tail(dim_y) = Eigen::MatrixXd::Ones(dim_y, 1) * INFINITY;

        //const std::vector<double> lb(num_vars, -max_coordinate);
        //const std::vector<double> ub(num_vars, max_coordinate);
        const std::vector<double> lb = EigenHelpers::EigenVectorXdToStdVectorDouble(eigen_lb);
        const std::vector<double> ub = EigenHelpers::EigenVectorXdToStdVectorDouble(eigen_ub);;

        // TODO: Find a way to put a scoped lock here. What's the functionality of this block of code? (to ".update()")
        gurobi_env_construct_mtx.lock();
        GRBEnv env;
        gurobi_env_construct_mtx.unlock();

        env.set(GRB_IntParam_OutputFlag, 0);
        GRBModel model(env);
        vars = model.addVars(lb.data(), ub.data(), nullptr, nullptr, nullptr, (int)num_vars);
        model.update();

        // Construct the Linear Program
        VectorXd f = MatrixXd::Zero(num_vars, 1);
        f.tail(dim_y) = MatrixXd::Ones(dim_y, 1);

        MatrixXd A = MatrixXd::Zero(2 * dim_y, num_vars);
        A.topLeftCorner(dim_y, dim_beta) = X;
        A.bottomLeftCorner(dim_y, dim_beta) = -X;

        VectorXd b = MatrixXd::Zero(dim_y * 2, 1);
        b.head(dim_y) = y;
        b.tail(dim_y) = -y;


        // Add constraint: "subject to A * x <= b"
        for (ssize_t ind = 0; ind < A.rows(); ind++)
        {
            model.addConstr(linearSum(A.row(ind), vars), GRB_LESS_EQUAL, b(ind));
        }

        // Build the objective function: min(f^T * x);
        GRBQuadExpr objective_fn = 0;
        // GRBLinExpr objective_fn = 0;
        objective_fn.addTerms(f.data(), vars, (int)num_vars);

        model.setObjective(objective_fn, GRB_MINIMIZE);

        model.update();
        model.optimize();

        // Only return the values of beta; not include the residual
        if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL)
        {
            x.resize(dim_beta);
            for (ssize_t var_ind = 0; var_ind < dim_beta; var_ind++)
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

