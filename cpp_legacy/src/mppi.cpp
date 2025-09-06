//
// Created by bjgilhul on 9/6/23.
//

#include <iostream>
#include "mppi.h"

namespace mppi {

    Eigen::VectorXd
    Ackermann_5_Euler_Step(const Eigen::VectorXd& state, const Eigen::Vector2d& control, double L, double dt) {
      auto dx = state(2) * cos(state(3));
      auto dy = state(2) * sin(state(3));
      auto dv = control(0);
      auto dth = state(2) * tan(state(4)) / L;
      auto dd = control(1);

      Eigen::VectorXd res(state.rows());
      res << dx, dy, dv, dth, dd;
      return res;
    };

    Eigen::VectorXd
    RK4_Step(const Eigen::VectorXd &state, const Eigen::Vector2d &control, double L, double dt) {
      auto k1 = Ackermann_5_Euler_Step(state, control, L, 1);
      auto k2 = Ackermann_5_Euler_Step(state + k1 * (dt / 2), control, L, 1);
      auto k3 = Ackermann_5_Euler_Step(state + k2 * (dt / 2), control, L, 1);
      auto k4 = Ackermann_5_Euler_Step(state + k3 * dt, control, L, 1);

      return ((k1 + 2 * (k2 + k3) + k4) / 6.0);
    };

    Eigen::Matrix2Xd
    mppi::find_control(const Eigen::MatrixXd &costmap, const std::pair<double,double>& origin, double resolution,
                       const Eigen::MatrixXd &u_nom,
                       const Eigen::VectorXd &initial_state, size_t samples, double L, double dt) {
      Eigen::VectorXd weights(samples);
      Eigen::MatrixXd u1_dist(samples, u_nom.cols());
      Eigen::MatrixXd u2_dist(samples, u_nom.cols());

      for (Eigen::Index i = 0; i < samples; i++) {
        auto u1_block = u1_dist.row(i);
        auto u2_block = u2_dist.row(i);
        double reward = rollout(costmap, origin, resolution, u_nom, initial_state, L, dt, u1_block, u2_block);
        weights(i) = reward;
        u1_block *= reward;
        u2_block *= reward;
      }

      // apply the weighted results to the nominal trajectory
      Eigen::MatrixXd u_weighted(2, u_nom.cols());
      double weight_total = weights.sum();
      u_weighted.row(0) = u_nom.row(0) + u1_dist.colwise().sum() / weight_total;
      u_weighted.row(1) = u_nom.row(1) + u2_dist.colwise().sum() / weight_total;

      return u_weighted;
    }

    // for each rollout, create disturbed input rollout based on some nominal input
    // return the resulting value based on the map, and the variations used
    template <typename M>
    double  mppi::rollout(
        const Eigen::MatrixXd &costmap,
        std::pair<double,double> origin,
        double resolution,
        const Eigen::Matrix2Xd &u,
        const Eigen::VectorXd &initial_state,
        double L,
        double dt,
        Eigen::MatrixBase<M>& u1_dist_rec,
        Eigen::MatrixBase<M>& u2_dist_rec
    ) {

      double score = 0;

      std::uniform_real_distribution<double> u1_dist(-disturbance_limits.first, disturbance_limits.first);
      std::uniform_real_distribution<double> u2_dist(-disturbance_limits.second, disturbance_limits.second);

      Eigen::VectorXd state = initial_state;
      for (Eigen::Index i = 0; i < u.cols(); i++) {
        u1_dist_rec(0, i) = u1_dist(rng);
        auto u1 = u(0, i) + u1_dist_rec(0, i);
        u2_dist_rec(0, i) = u2_dist(rng);
        auto u2 = u(1, i) + u2_dist_rec(0, i);
        Eigen::Vector2d u_i;
        u_i << u1, u2;
        Eigen::VectorXd step = step_fn(state, u_i, L, dt);
        state += step * dt;
        // std::cout << state << std::endl;
        auto x = int( (state.coeff(0) - origin.first + resolution/2.0 ) / resolution );
        auto y = int( (state.coeff(1) - origin.second + resolution/2.0 ) / resolution );

        if( x < 0 || x > costmap.cols() - 1 || y < 0 || y > costmap.rows() - 1 ) {
          score += 10000;   // high 'reward' for going out of bounds
        } else {
          score += costmap.coeff(y,x);
        }
//        std::cout << "costmap: " << x <<", " << y << ", " << reward << std::endl;
//        std::cout << "state: " << state(0) <<", " << state(1) << " -- control: "  << u_i(0) << ", "  << u_i(1) << ", " << std::endl;
      }

      // recast the score as a weight before returning the result
      return exp((-1.0 / lambda ) * score );
    }

}
