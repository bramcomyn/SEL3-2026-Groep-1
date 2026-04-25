import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function positions_trajectory_chart() {
    const trajectory = load_csv('../out/eval_positions.csv');

    const maxAbs = Math.max(
        ...trajectory.map(d => Math.max(Math.abs(d.x), Math.abs(d.y)))
    ) - 0.5;
    const domain = [-maxAbs, maxAbs];

    const path = vl.markLine({ point: { size: 15 }, strokeWidth: 1, color: "#557086" })
        .transform(
            vl.filter("datum.in_trajectory")
        )
        .encode(
            vl.x().fieldQ("x").axis(null).scale({ domain, padding: 20 }),
            vl.y().fieldQ("y").axis(null).scale({ domain, padding: 20 }),
            vl.order().fieldO("step_id"),
            vl.detail().fieldO("environment_id")
        );

    const target = vl.markCircle({ color: "#ff0000", opacity: 0.4, size: 200 })
        .transform(
            vl.filter("!datum.in_trajectory")
        )
        .encode(
            vl.x().fieldQ("x"),
            vl.y().fieldQ("y")
        );

    return vl.layer(path, target)
        .data(trajectory)
        .title({
            text: "Trajectories taken by the Brittle Star",
            subtitle: "Targets indicated in red",
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
