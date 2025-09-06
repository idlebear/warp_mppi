//
// Created by bjgilhul on 9/6/23.
//

#ifndef POLYCHECK_MPPI_H
#define POLYCHECK_MPPI_H

#include <array>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <utility>
#include <fstream>

namespace mppi {

    Eigen::VectorXd
    Ackermann_5_Euler_Step(const Eigen::VectorXd& state, const Eigen::Vector2d& control, double L, double dt = 1.0);

    Eigen::VectorXd
    RK4_Step(const Eigen::VectorXd &state, const Eigen::Vector2d &control, double L, double dt);

    namespace random {
        // https://stackoverflow.com/questions/45069219/how-to-succinctly-portably-and-thoroughly-seed-the-mt19937-prng
        inline size_t sysrandom(void* dst, size_t dstlen)
        {
          char* buffer = reinterpret_cast<char*>(dst);
          std::ifstream stream("/dev/urandom", std::ios_base::binary | std::ios_base::in);
          stream.read(buffer, dstlen);

          return dstlen;
        }
    };

    class mppi {
        public:
        mppi(
                Eigen::VectorXd (*step_fn)(const Eigen::VectorXd &, const Eigen::Vector2d &, double, double),
                double lambda, uint32_t seed, std::pair<double, double> disturbance_limits
        ) : step_fn(step_fn), disturbance_limits(std::move(disturbance_limits)), lambda(lambda), seed(seed) {

          if( seed == -1 ) {
            random::sysrandom(&seed, sizeof(seed));
          }
          rng.seed(seed);

        };

        ~mppi() = default;

        Eigen::Matrix2Xd
        find_control(const Eigen::MatrixXd &costmap,
                     const std::pair<double,double>& origin,
                     double resolution,
                     const Eigen::MatrixXd &u_nom,
                     const Eigen::VectorXd &initial_state, size_t samples, double L,
                     double dt);

        // for each rollout, create disturbed input rollout based on some nominal input
        // return the resulting value based on the map, and the variations used
        template <typename M> double
        rollout(const Eigen::MatrixXd &costmap,
                std::pair<double,double> origin,
                double resolution,
                const Eigen::Matrix2Xd &u,
                const Eigen::VectorXd &initial_state,
                double L,
                double dt,
                Eigen::MatrixBase<M>&  u1_dist_rec,
                Eigen::MatrixBase<M>&  u2_dist_rec
        );

        private:
        Eigen::VectorXd (*step_fn)(const Eigen::VectorXd &, const Eigen::Vector2d &, double, double);

        std::pair<double, double> disturbance_limits;
        uint32_t seed;
        double lambda;
        std::mt19937 rng;
    };



}

#endif //POLYCHECK_MPPI_H
