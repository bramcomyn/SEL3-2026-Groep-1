import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function actions_trajectory_chart() {
    const trajectory = load_csv('../out/eval_actions.csv');
    const actions = Array.from({ length: 5 }, (_, i) => i);
    const steps = Array.from({ length: Math.max(...trajectory.map(row => row.step_id))+1 }, (_, i) => i)

    return vl.markPoint({ strokeWidth: 1.5, interpolate: "step-after" })
        .data(trajectory)
        .transform(    
            vl.filter("datum.environment_id === 0"), // Plot only one environment
            vl.calculate("datum.action === 2 ? 3 : datum.action === 3 ? 2 : datum.action")
                .as("swapped_action")
        )
        .encode(
            vl.x()
                .fieldQ("step_id")
                .scale({ nice: false })
                .axis({ 
                    grid: true, 
                    gridOpacity: 0.75, 
                    values: steps
                })
                .title("Step"),
            vl.y()
                .fieldO("swapped_action")
                .scale({
                    domain: actions
                })
                .axis({ 
                    grid: false,
                    labelExpr: "['Leading','Left primary','Left secondary','Right primary','Right secondary'][+datum.value]"
                })
                .title("Action"),
            vl.detail()
                .fieldN("agent_id"),
            vl.row()
                .fieldN("agent_id")
                .title("Arm")
        )
        .title({
            text: "Taken actions of each agent during evaluation",
            anchor: "middle",
            offset: 30
        })
        .height(120)
        .width(400)
        .config({
            font: "RedHatDisplay",
            title: {
                fontSize: 18
            }
        })
        .toSpec();
}
