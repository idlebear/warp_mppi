#include <iostream>
#include "mppi.h"
#include <iostream>

int main() {
  mppi::mppi roller( mppi::RK4_Step, 10, -1, std::make_pair(4, M_PI/2.0 ));
  auto steps = 30;
  auto dt = 0.2;
  auto L = 3.0;

  Eigen::MatrixXd u_nom( 2, steps );
  for( Eigen::Index i = 0; i < steps; i++ ){
    u_nom(0,i) = 1;
    u_nom(1,i) = 0;
  }

  std::pair<double,double> origin = std::make_pair(-1.0,-6.0);
  double resolution = 0.1;

  auto mapsize = 120;
  auto peak_x = 8.0;
  auto peak_y = 2.0;

  auto map_x = (peak_x - origin.first) / resolution;
  auto map_y = (peak_y - origin.second) / resolution;

  Eigen::MatrixXd costmap( mapsize, mapsize );
  for( Eigen::Index i = 0; i < mapsize; i++ ){
    for( Eigen::Index j = 0; j < mapsize; j++ ) {
      auto x_offset = origin.first + i * resolution;
      auto y_offset = origin.second + j * resolution;
      auto value = abs(y_offset - peak_y);
      if( value > 3) {
        value = 100;
      }
      if( peak_x - x_offset < 0) {
        value = 100;
      }
      costmap(j,i) = value;
    }
  }
  Eigen::VectorXd initial(5);
  initial << 0,0,0,0,0;

  auto u = roller.find_control( costmap, origin, resolution, u_nom, initial, 5000, L, dt);

  auto state = initial;
  auto last_distance = 10000.0;
  for( int i = 0; i < steps; i++  ) {
    auto control = u.col(i);
    auto step = mppi::RK4_Step( state, control, L, dt );
    state = state + step * dt;

    auto dx = peak_x - state(0);
    auto dy = peak_y - state(1);
    auto dist = sqrt( dx * dx + dy * dy );

    std::string distance_label = {" -- Distance:"};
    if( last_distance < dist ) {
      distance_label += " (**) ";
    }
    last_distance = dist;

    std::cout << "State: " << state(0) << ", " << state(1) << ", " << state(3) << " --  U:" << control( 0 ) << ", "  << control(1)<< distance_label << dist << std::endl;
  }

  return 0;
}
