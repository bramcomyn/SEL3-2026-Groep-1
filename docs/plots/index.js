import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

import { save_chart_svg, save_chart_png } from './utils/save_chart.js';
import { actions_trajectory_chart } from './charts/actions_trajectory.js';
import { positions_trajectory_chart } from './charts/positions_trajectory.js';
import { learning_curve_comparison_chart } from './charts/learning_curve_comparison.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const args = process.argv.slice(2);

const project_root = join(__dirname, "../../");
const out_dir = join(__dirname, "out");

const mode = args[0];

if(mode === 'eval') {
    const actions_trajectory_csv = join(project_root, args[1]);
    const breakpoints_trajectory_csv = join(project_root, args[2]);
    const positions_trajectory_csv = join(project_root, args[3]);

    save_chart_png(
        actions_trajectory_chart(actions_trajectory_csv, breakpoints_trajectory_csv), 
        join(out_dir, 'action_trajectory')
    );

    save_chart_png(
        positions_trajectory_chart(positions_trajectory_csv, breakpoints_trajectory_csv), 
        join(out_dir, 'position_trajectory')
    );
} else if(mode === 'comparison') {
    const learning_curve_comparison_csv = join(project_root, args[1]);

    save_chart_svg(
        learning_curve_comparison_chart(learning_curve_comparison_csv),
        join(out_dir, 'learning_curve_comparison')
    );
} else {
    console.error(`Unknown mode: ${mode}`);
}
