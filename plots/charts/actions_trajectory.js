import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function actions_trajectory_chart() {
    const trajectory = load_csv('../out/eval_actions.csv');

    return vl.markLine({ strokeWidth: 1.5 })
        .data(trajectory)
        .transform(    
            vl.calculate("toNumber(datum.action) === 2 ? 3 : toNumber(datum.action) === 3 ? 2 : toNumber(datum.action)")
                .as("swapped_action")
        )
        .encode(
            vl.x()
                .fieldQ("step_id")
                .scale({ nice: false })
                .axis({ 
                    grid: true, 
                    gridOpacity: 0.75, 
                    values: Array.from({ length: 20 }, (_, i) => i) 
                })
                .title("Step"),
            vl.y()
                .fieldQ("swapped_action")
                .scale({ padding: 5 })
                .axis({ 
                    grid: false,
                    values: Array.from({ length: 5 }, (_, i) => i),
                    labelExpr: "datum.value === 0 ? 'Leading' : datum.value === 1 ? 'Left primary' : datum.value === 2 ? 'Left secondary' : datum.value === 3 ? 'Right primary' : datum.value === 4 ? 'Right secondary' : ''"
                })
                .title("Action"),
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
