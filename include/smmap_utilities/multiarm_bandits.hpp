#ifndef MULTIARM_BANDITS_HPP
#define MULTIARM_BANDITS_HPP

#include <assert.h>
#include <vector>
#include <random>
#include <utility>

#include <Eigen/Eigenvalues>

#include <arc_utilities/arc_helpers.hpp>
#include <arc_utilities/pretty_print.hpp>
#include <arc_utilities/eigen_helpers.hpp>

namespace smmap_utilities
{
    class MABBase
    {
        public:
            MABBase(size_t num_arms)
                : num_arms_(num_arms)
            {}

            virtual ssize_t selectArmToPull(std::mt19937_64& generator) const = 0;

            virtual bool generateAllModelActions() const = 0;

            virtual void updateArms(const ssize_t arm_pulled, const double reward)
            {
                (void)arm_pulled;
                (void)reward;
                // Note that all 3 versions of this function are included for ease of
                // selection of the individual methods for the dervied classes
                // TODO: fix this
                assert(false && "This base version should never be called");
            }

            virtual void updateArms(
                    const Eigen::VectorXd& transition_variance,
                    const ssize_t arm_pulled,
                    const double observed_reward,
                    const double observation_variance)
            {
                (void)transition_variance;
                (void)arm_pulled;
                (void)observed_reward;
                (void)observation_variance;
                // Note that all 3 versions of this function are included for ease of
                // selection of the individual methods for the dervied classes
                // TODO: fix this
                assert(false && "This base version should never be called");
            }

            virtual void updateArms(
                    const Eigen::MatrixXd& transition_covariance,
                    const Eigen::MatrixXd& observation_matrix,
                    const Eigen::VectorXd& observed_reward,
                    const Eigen::MatrixXd& observation_covariance)
            {
                (void)transition_covariance;
                (void)observation_matrix;
                (void)observed_reward;
                (void)observation_covariance;
                // Note that all 3 versions of this function are included for ease of
                // selection of the individual methods for the dervied classes
                // TODO: fix this
                assert(false && "This base version should never be called");
            }

            virtual Eigen::VectorXd getMean() const = 0;

            virtual Eigen::MatrixXd getSecondStat() const = 0;

        protected:
            size_t num_arms_;
    };

    class UCB1NormalBandit : public MABBase
    {
        public:
            UCB1NormalBandit(size_t num_arms)
                : MABBase(num_arms)
                , total_reward_(num_arms_, 0.0)
                , sum_of_squared_reward_(num_arms_, 0.0)
                , num_pulls_(num_arms_, 0)
                , total_pulls_(0)
            {}

            virtual ssize_t selectArmToPull(std::mt19937_64& generator) const override final
            {
                (void)generator;
                if (total_pulls_ == 0UL)
                {
                    return 0L;
                }

                const double explore_threshold = 8.0 * std::log((double)total_pulls_ + 1.0);
                double highest_ucb = -std::numeric_limits<double>::infinity();
                ssize_t best_arm = -1;
                size_t lowest_num_exploration_pulls = (size_t)(-1);

                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    if (num_pulls_[arm_ind] < explore_threshold)
                    {
                        if (num_pulls_[arm_ind] < lowest_num_exploration_pulls)
                        {
                            best_arm = arm_ind;
                            lowest_num_exploration_pulls = num_pulls_[arm_ind];
                        }
                    }
                }

                // If we found an arm that qualifies for exploration, pull that arm
                if (best_arm != -1)
                {
                    return best_arm;
                }

                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    assert(num_pulls_[arm_ind] > 1);

                    const double average_reward = total_reward_[arm_ind] / (double)num_pulls_[arm_ind];
                    const double term1_numer = (sum_of_squared_reward_[arm_ind] - (double)num_pulls_[arm_ind] * average_reward * average_reward);
                    const double term1_denom = (double)(num_pulls_[arm_ind] - 1);
                    const double term1 = std::abs(term1_numer)/term1_denom;
                    const double term2 = std::log((double)total_pulls_) / (double)num_pulls_[arm_ind];
                    const double ucb = average_reward + std::sqrt(16.0 * term1 * term2);

                    assert(std::isfinite(ucb));

                    if (ucb > highest_ucb)
                    {
                        highest_ucb = ucb;
                        best_arm = (ssize_t)arm_ind;
                    }
                }

                assert(best_arm >= 0);
                return best_arm;
            }

            virtual bool generateAllModelActions() const override final
            {
                return false;
            }

            virtual void updateArms(const ssize_t arm_pulled, const double reward) override final
            {
                total_reward_[(size_t)arm_pulled] += reward;
                sum_of_squared_reward_[(size_t)arm_pulled] += reward * reward;
                num_pulls_[(size_t)arm_pulled]++;
                total_pulls_++;
            }

            virtual Eigen::VectorXd getMean() const override final
            {
                Eigen::VectorXd mean((ssize_t)num_arms_);
                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    mean((ssize_t)arm_ind) = total_reward_[arm_ind] / (double)num_pulls_[arm_ind];
                }
                return mean;
            }

            virtual Eigen::MatrixXd getSecondStat() const override final
            {
                return getUCB();
            }

            Eigen::VectorXd getUCB() const
            {
                Eigen::VectorXd ucb((ssize_t)num_arms_);
                for (size_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    const double average_reward = total_reward_[arm_ind] / (double)num_pulls_[arm_ind];
                    const double term1_numer = (sum_of_squared_reward_[arm_ind] - (double)num_pulls_[arm_ind] * average_reward * average_reward);
                    const double term1_denom = (double)(num_pulls_[arm_ind] - 1);
                    const double term1 = std::abs(term1_numer)/term1_denom;
                    const double term2 = std::log((double)total_pulls_) / (double)num_pulls_[arm_ind];
                    ucb((ssize_t)arm_ind) = average_reward + std::sqrt(16.0 * term1 * term2);
                }
                return ucb;
            }

        private:
            std::vector<double> total_reward_;
            std::vector<double> sum_of_squared_reward_;
            std::vector<size_t> num_pulls_;
            size_t total_pulls_;
    };

    class KalmanFilterMANB : public MABBase
    {
        public:
            KalmanFilterMANB(
                    const Eigen::VectorXd& prior_mean,
                    const Eigen::VectorXd& prior_var)
                : MABBase(prior_mean.cols())
                , arm_mean_(prior_mean)
                , arm_var_(prior_var)
            {
                assert(arm_mean_.cols() == arm_var_.cols());
            }

            /**
             * @brief selectArmToPull Perform Thompson sampling on the bandits,
             *                        and select the bandit with the largest sample.
             * @param generator
             * @return
             */
            virtual ssize_t selectArmToPull(std::mt19937_64& generator) const override final
            {
                // Sample from the current distribuition
                std::normal_distribution<double> normal_dist(0.0, 1.0);

                ssize_t best_arm = -1;
                double best_sample = -std::numeric_limits<double>::infinity();
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    const double sample = std::sqrt(arm_var_(arm_ind)) * normal_dist(generator) + arm_mean_(arm_ind);

                    if (sample > best_sample)
                    {
                        best_arm = arm_ind;
                        best_sample = sample;
                    }
                }

                assert(best_arm >= 0);
                return best_arm;
            }

            virtual bool generateAllModelActions() const
            {
                return false;
            }

            /**
             * @brief updateArms
             * @param transition_variance
             * @param arm_pulled
             * @param observed_reward
             * @param observation_variance
             */
            virtual void updateArms(
                    const Eigen::VectorXd& transition_variance,
                    const ssize_t arm_pulled,
                    const double observed_reward,
                    const double observation_variance) override final
            {
                for (ssize_t arm_ind = 0; arm_ind < num_arms_; arm_ind++)
                {
                    if (arm_ind != arm_pulled)
                    {
                        arm_var_(arm_ind) += transition_variance(arm_ind);
                    }
                    else
                    {
                        arm_mean_(arm_ind) = ((arm_var_(arm_ind) + transition_variance(arm_ind)) * observed_reward + observation_variance * arm_mean_(arm_ind))
                                            / (arm_var_(arm_ind) + transition_variance(arm_ind) + observation_variance);

                        arm_var_(arm_ind) = (arm_var_(arm_ind) + transition_variance(arm_ind)) * observation_variance
                                           / (arm_var_(arm_ind) + transition_variance(arm_ind) + observation_variance);
                    }
                }
            }

            virtual Eigen::VectorXd getMean() const override final
            {
                return arm_mean_;
            }

            virtual Eigen::MatrixXd getSecondStat() const override final
            {
                return getVariance();
            }

            const Eigen::MatrixXd& getVariance() const
            {
                return arm_var_;
            }

        private:
            Eigen::VectorXd arm_mean_;
            Eigen::VectorXd arm_var_;
    };

    class KalmanFilterMANDB : public MABBase
    {
        public:
            KalmanFilterMANDB(
                    const Eigen::VectorXd& prior_mean,
                    const Eigen::MatrixXd& prior_covar)
                : MABBase(prior_mean.cols())
                , arm_mean_(prior_mean)
                , arm_covar_(prior_covar)
            {
                assert(arm_covar_.rows() == arm_covar_.cols());
                assert(arm_covar_.rows() == arm_mean_.rows());
            }

            /**
             * @brief selectArmToPull Perform Thompson sampling on the bandits,
             *                        and select the bandit with the largest sample.
             * @param generator
             * @return
             */
            virtual ssize_t selectArmToPull(std::mt19937_64& generator) const override final
            {
                // Sample from the current distribuition
                arc_helpers::MultivariteGaussianDistribution distribution(arm_mean_, arm_covar_);
                const Eigen::VectorXd sample = distribution(generator);

                // Find the arm with the highest sample
                ssize_t best_arm = -1;
                sample.maxCoeff(&best_arm);

                return best_arm;
            }

            virtual bool generateAllModelActions() const override final
            {
                return true;
            }

            /**
             * @brief updateArms
             * @param transition_covariance
             * @param arm_pulled
             * @param obs_reward
             * @param obs_var
             */
            virtual void updateArms(
                    const Eigen::MatrixXd& transition_covariance,
                    const Eigen::MatrixXd& observation_matrix,
                    const Eigen::VectorXd& observed_reward,
                    const Eigen::MatrixXd& observation_covariance) override final
            {
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Wconversion"
                const Eigen::MatrixXd& C = observation_matrix;

                // Kalman predict
                const Eigen::VectorXd& predicted_mean = arm_mean_;                      // No change to mean
                const auto predicted_covariance = arm_covar_ + transition_covariance;   // Add process noise

                // Kalman update - symbols from wikipedia article
                const auto innovation = observed_reward - C * predicted_mean;                                            // tilde y_k

                const auto innovation_covariance = C * predicted_covariance.selfadjointView<Eigen::Lower>() * C.transpose() + observation_covariance;    // S_k
                const auto kalman_gain = predicted_covariance.selfadjointView<Eigen::Lower>() * C.transpose() * innovation_covariance.inverse();         // K_k


                arm_mean_ = predicted_mean + kalman_gain * innovation;                                                              // hat x_k|k
                arm_covar_ = predicted_covariance - kalman_gain * C * predicted_covariance.selfadjointView<Eigen::Lower>();         // P_k|k
                #pragma GCC diagnostic pop

                // Numerical problems fixing
                arm_covar_ = ((arm_covar_ + arm_covar_.transpose()) * 0.5).selfadjointView<Eigen::Lower>();

                assert(!(arm_mean_.unaryExpr([] (const double &val) { return std::isnan(val); })).any() && "NaN Found in arm_mean_ in kalman banidt!");
                assert(!(arm_mean_.unaryExpr([] (const double &val) { return std::isinf(val); })).any() && "Inf Found in arm_mean_ in kalman banidt!");
                assert(!(arm_covar_.unaryExpr([] (const double &val) { return std::isinf(val); })).any() && "NaN Found in arm_covar_ in kalman bandit!");
                assert(!(arm_covar_.unaryExpr([] (const double &val) { return std::isinf(val); })).any() && "Inf Found in arm_covar_ in kalman bandit!");
            }

            virtual Eigen::VectorXd getMean() const override final
            {
                return arm_mean_;
            }

            virtual Eigen::MatrixXd getSecondStat() const override final
            {
                return getCovariance();
            }

            const Eigen::MatrixXd& getCovariance() const
            {
                return arm_covar_;
            }

        private:
            Eigen::VectorXd arm_mean_;
            Eigen::MatrixXd arm_covar_;
    };
}

#endif // MULTIARM_BANDITS_HPP
