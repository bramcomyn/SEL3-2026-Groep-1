import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function positions_trajectory_chart() {
    const data = preprocess_data();

    const maxAbs = Math.max(
        ...data.map(d => Math.max(Math.abs(d.x), Math.abs(d.y)))
    ) - 0.5;
    const domain = [-maxAbs, maxAbs];

    const no_damage_path = vl.markLine({ point: { size: 8 }, strokeWidth: 1 })
        .transform(
            vl.filter("datum.in_trajectory"),
            vl.filter("datum.step_id < datum.breakpoint")
        )
        .encode(
            vl.x().fieldQ("x").axis(null).scale({ domain, padding: 20 }),
            vl.y().fieldQ("y").axis(null).scale({ domain, padding: 20 }),
            vl.order().fieldO("step_id"),
            vl.detail().fieldO("environment_id"),
            vl.color().value("#3a7fb8")
        );

    const damage_path = vl.markLine({ point: { size: 8 }, strokeWidth: 1 })
        .transform(
            vl.filter("datum.in_trajectory"),
            vl.filter("datum.step_id >= datum.breakpoint - 1")
        )
        .encode(
            vl.x().fieldQ("x").axis(null).scale({ domain, padding: 20 }),
            vl.y().fieldQ("y").axis(null).scale({ domain, padding: 20 }),
            vl.order().fieldO("step_id"),
            vl.detail().fieldO("environment_id"),
            vl.color().value("#f16161")
        );

    const target = vl.markCircle({ color: "#000000", opacity: 0.4, size: 150 })
        .transform(
            vl.filter("!datum.in_trajectory")
        )
        .encode(
            vl.x().fieldQ("x"),
            vl.y().fieldQ("y")
        );

    return vl.layer(no_damage_path, damage_path, target)
        .data(data)
        .title({
            text: "Trajectories taken by the Brittle Star",
            subtitle: "Targets indicated in grey, damaged interval in red",
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

function preprocess_data() {
    const trajectory = load_csv('../out/eval_positions.csv');
    const breakpoints = load_csv('../out/eval_breakpoints.csv');

    // Convert breakpoints into a lookup map
    const breakpointMap = {};
    breakpoints.forEach(b => {
    breakpointMap[b.environment_id] = b.breakpoint;
    });

    // Merge
    return trajectory.map(t => {
        const bp = breakpointMap[t.environment_id];

        return {
            ...t,
            breakpoint: bp
        };
    });
}
