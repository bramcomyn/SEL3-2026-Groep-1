import { save_chart_svg, save_chart_png } from './utils/save_chart.js';
import { actions_trajectory_chart } from './charts/actions_trajectory.js';
import { positions_trajectory_chart } from './charts/positions_trajectory.js';

save_chart_png(actions_trajectory_chart(), 'out/action_trajectory')
save_chart_png(positions_trajectory_chart(), 'out/position_trajectory')
