#include "smmap_utilities/nomad_solvers.h"

#include <iostream>
#include <arc_utilities/arc_helpers.hpp>

namespace smmap_utilities
{
    AllGrippersSinglePoseDelta minFunctionPointerSE3Delta(
            const std::string& log_file_path,
            const bool fix_step,
            const int max_count,
            const ssize_t num_grippers,
            const double max_step_size,
            std::mt19937_64& generator,
            std::uniform_real_distribution<double>& uniform_unit_distribution,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& eval_error_cost_fn,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& collision_constraint_fn,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& stretching_constraint_fn,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& gripper_motion_constraint_fn)
    {
        AllGrippersSinglePoseDelta optimal_gripper_command;

        // TODO: figure out a way to deal with logging, for now leave extra code here for reference
        ofstream out(log_file_path.c_str(), ios::out);
        // NOMAD::Display out (std::cout);
        out.precision (NOMAD::DISPLAY_PRECISION_STD);

        try
        {
            // NOMAD initializations:
            NOMAD::begin(0, nullptr);

            // parameters creation:
            NOMAD::Parameters p(out);
    //        NOMAD::Parameters p;
            p.set_DIMENSION((int)(6 * num_grippers));  // number of variables

            vector<NOMAD::bb_output_type> bbot (4); // definition of
            bbot[0] = NOMAD::OBJ;                   // output types
            // TODO: might need to decide which kind of constraint to use
            bbot[1] = NOMAD::PB;
            bbot[2] = NOMAD::PB;
            bbot[3] = NOMAD::PB;

            if (fix_step)
            {
                bbot.push_back(NOMAD::EB);
            }

            p.set_BB_OUTPUT_TYPE(bbot);

            const int x_dim = (int)(6 * num_grippers);
            const int size_of_initial_batch = 5;

            for (int sample_ind = 0; sample_ind < size_of_initial_batch; sample_ind++)
            {
                NOMAD::Point x0 = NOMAD::Point(x_dim, 0.0);
                for (int coord_ind = 0; coord_ind < x_dim; coord_ind++)
                {
                    x0.set_coord(coord_ind, EigenHelpers::Interpolate(-max_step_size, max_step_size, uniform_unit_distribution(generator)));
                }
                p.set_X0(x0);
            }

            p.set_LOWER_BOUND(NOMAD::Point((int)(6 * num_grippers), -max_step_size));
            p.set_UPPER_BOUND(NOMAD::Point((int)(6 * num_grippers), max_step_size));

            p.set_MAX_BB_EVAL(max_count);     // the algorithm terminates after max_count_ black-box evaluations
            p.set_DISPLAY_DEGREE(2);
            //p.set_SGTELIB_MODEL_DISPLAY("");
            p.set_SOLUTION_FILE("sol.txt");

            // parameters validation:
            p.check();

            // custom evaluator creation:
            GripperMotionNomadEvaluator ev(p,
                                           num_grippers,
                                           eval_error_cost_fn,
                                           collision_constraint_fn,
                                           stretching_constraint_fn,
                                           gripper_motion_constraint_fn,
                                           fix_step);

            // algorithm creation and execution:
            NOMAD::Mads mads(p, &ev);
            mads.run();

            const NOMAD::Eval_Point* best_x = mads.get_best_feasible();

            optimal_gripper_command = ev.evalPointToGripperPoseDelta(*best_x);

            if (optimal_gripper_command.size() == 0)
            {
                std::cerr << "   ---------  No output from NOMAD evaluator ------------" << std::endl;
                const kinematics::Vector6d no_movement = kinematics::Vector6d::Zero();
                optimal_gripper_command = AllGrippersSinglePoseDelta(num_grippers, no_movement);
            }
        }
        catch (std::exception& e)
        {
            cerr << "\nNOMAD has been interrupted (" << e.what() << ")\n\n";
        }

        NOMAD::Slave::stop_slaves(out);
        NOMAD::end();

        return optimal_gripper_command;
    }

    GripperMotionNomadEvaluator::GripperMotionNomadEvaluator(
            const NOMAD::Parameters & p,
            const ssize_t num_grippers,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& eval_error_cost_fn,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& collision_constraint_fn,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& stretching_constraint_fn,
            const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& gripper_motion_constraint_fn,
            const bool fix_step_size)
        : NOMAD::Evaluator(p)
        , num_grippers_(num_grippers)
        , eval_error_cost_fn_(eval_error_cost_fn)
        , collision_constraint_fn_(collision_constraint_fn)
        , stretching_constraint_fn_(stretching_constraint_fn)
        , gripper_motion_constraint_fn_(gripper_motion_constraint_fn)
        , fix_step_size_(fix_step_size)
    {}

    AllGrippersSinglePoseDelta GripperMotionNomadEvaluator::evalPointToGripperPoseDelta(const NOMAD::Eval_Point& x)
    {
        if (&x == nullptr)
        {
            return AllGrippersSinglePoseDelta(num_grippers_, kinematics::Vector6d::Zero());
        }

        const int single_gripper_dimension = 6;
        if (x.size() != num_grippers_ * single_gripper_dimension)
        {
            assert(false && "grippers data and eval_point x have different size");
        }

        AllGrippersSinglePoseDelta grippers_motion(num_grippers_);
        for (int gripper_ind = 0; gripper_ind < num_grippers_; gripper_ind ++)
        {
            kinematics::Vector6d& single_gripper_delta = grippers_motion[gripper_ind];

            single_gripper_delta(0) = x[gripper_ind * single_gripper_dimension].value();
            single_gripper_delta(1) = x[gripper_ind * single_gripper_dimension + 1].value();
            single_gripper_delta(2) = x[gripper_ind * single_gripper_dimension + 2].value();

            single_gripper_delta(3) = x[gripper_ind * single_gripper_dimension + 3].value();
            single_gripper_delta(4) = x[gripper_ind * single_gripper_dimension + 4].value();
            single_gripper_delta(5) = x[gripper_ind * single_gripper_dimension + 5].value();
        }

        return grippers_motion;
    }

    bool GripperMotionNomadEvaluator::eval_x(
            NOMAD::Eval_Point& x,
            const NOMAD::Double& h_max,
            bool& count_eval)
    {
        UNUSED(h_max); // TODO: Why don't we use h_max?

        // count a black-box evaluation
        count_eval = true;

        // Convert NOMAD points into
        const AllGrippersSinglePoseDelta test_grippers_motions = evalPointToGripperPoseDelta(x);

        NOMAD::Double c1_error_cost = eval_error_cost_fn_(test_grippers_motions);
        NOMAD::Double c2_collision_constraint = collision_constraint_fn_(test_grippers_motions);
        NOMAD::Double c3_stretching_constraint = stretching_constraint_fn_(test_grippers_motions);
        NOMAD::Double c4_gripper_motion_constraint = gripper_motion_constraint_fn_(test_grippers_motions);

        // objective value
        x.set_bb_output(0, c1_error_cost);

        // constraints
        x.set_bb_output(1, c2_collision_constraint);
        x.set_bb_output(2, c3_stretching_constraint);
        x.set_bb_output(3, c4_gripper_motion_constraint);

        if (fix_step_size_)
        {
            if (x.get_bb_outputs().size() < 5)
            {
                assert(false && "size of x not match due to the fix step size constraint");
            }
            x.set_bb_output(4, -c4_gripper_motion_constraint);
        }

        return count_eval;
    }
}






