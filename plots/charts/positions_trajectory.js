import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function positions_trajectory_chart() {
    const trajectory = load_csv('../out/eval_positions.csv');

    const path = vl.markLine({ point: true, color: "#557086" })
        .transform(
            vl.filter("datum.in_trajectory")
        )
        .encode(
            vl.x().fieldQ("x").axis(null).scale({ padding: 20 }),
            vl.y().fieldQ("y").axis(null).scale({ padding: 20 }),
            vl.order().fieldO("step_id")
        );

    const target = vl.markCircle({ color: "#ff0000", size: 1000 })
        .transform(
            vl.filter("!datum.in_trajectory")
        )
        .encode(
            vl.x().fieldQ("x"),
            vl.y().fieldQ("y")
        );

    return vl.layer(path, target)
        .facet(
            vl.column().fieldO("environment_id").title("Environment")
        )
        .data(trajectory)
        .title({
            text: "Trajectory taking by the Brittle Star",
            anchor: "middle",
            offset: 30
        })
        .resolve({
            scale: { x: 'shared', y: 'shared' } 
        })
        .config({
            font: "RedHatDisplay",
            title: {
                fontSize: 18
            },
            view: {
                aspect: 1,
                strokeOpacity: 0.5
            }
        })
        .toSpec();
}
