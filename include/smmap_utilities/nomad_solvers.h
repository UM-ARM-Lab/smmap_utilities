#ifndef NOMAD_SOLVERS_H
#define NOMAD_SOLVERS_H

#include <nomad.hpp>
#include <Eigen/Dense>
#include <arc_utilities/eigen_helpers.hpp>
#include <kinematics_toolbox/kinematics.h>

namespace smmap_utilities
{
    ///////////////////////////////////////////////////////////////////
    // Class interface for the evaluator
    ///////////////////////////////////////////////////////////////////
    class GripperMotionNomadEvaluator : public NOMAD::Evaluator
    {
        typedef kinematics::VectorVector6d AllGrippersSinglePoseDelta;

        public:
          GripperMotionNomadEvaluator(
                  const NOMAD::Parameters& p,
                  const ssize_t num_grippers,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& eval_error_cost_fn,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& collision_constraint_fn,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& stretching_constraint_fn,
                  const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)>& gripper_motion_constraint_fn,
                  const bool fix_step_size = false);

          bool eval_x(
                  NOMAD::Eval_Point& x,
                  const NOMAD::Double& h_max,
                  bool& count_eval);

          AllGrippersSinglePoseDelta evalPointToGripperPoseDelta(
                  const NOMAD::Eval_Point& x);

        private:
          const ssize_t num_grippers_;

          const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)> eval_error_cost_fn_;
          const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)> collision_constraint_fn_;
          const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)> stretching_constraint_fn_;
          const std::function<double(const AllGrippersSinglePoseDelta& test_gripper_motion)> gripper_motion_constraint_fn_;
          const bool fix_step_size_;
          // const sdf_tools::SignedDistanceField enviroment_sdf_;

    };
}

#endif // NOMAD_SOLVERS_H