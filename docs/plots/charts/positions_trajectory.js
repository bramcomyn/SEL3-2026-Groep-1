import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

// registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function positions_trajectory_chart(
    positions_trajectory_csv, 
    breakpoints_trajectory_csv
) {
    const data = preprocess_data(
        positions_trajectory_csv,
        breakpoints_trajectory_csv
    );

    const maxAbs = Math.max(
        ...data.map(d => Math.max(Math.abs(d.x), Math.abs(d.y)))
    ) - 1;
    const domain = [-maxAbs, maxAbs];

    const no_damage_path = vl.markLine({ point: { size: 8 }, strokeWidth: 1 })
        .transform(
            vl.filter("datum.in_trajectory"),
            vl.filter("datum.step_id < datum.breakpoint")
        )
        .encode(
            vl.x().fieldQ("x").axis(null).scale({ domain, nice: false, padding: 20 }),
            vl.y().fieldQ("y").axis(null).scale({ domain, nice: false, padding: 20 }),
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
            vl.x().fieldQ("x").axis(null).scale({ domain, nice: false, padding: 20 }),
            vl.y().fieldQ("y").axis(null).scale({ domain, nice: false, padding: 20 }),
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

    const damaged_annotation = vl
        .markText({ color: "#f16161", align: 'left', dx: 0, dy: -35, fontSize: 13 })
        .transform(
            vl.filter(`
                datum.in_trajectory && datum.step_id === datum.breakpoint - 1
                && datum.environment_id === 0
            `)
        )
        .encode(
            vl.x().fieldQ("x"),
            vl.y().fieldQ("y"),
            vl.text().value("Trajectory with damage"),
        );

    const undamaged_annotation = vl
        .markText({ color: "#3a7fb8", align: 'left', dx: 30, dy: 35, fontSize: 13 })
        .transform(
            vl.filter(`
                datum.in_trajectory && datum.step_id === datum.breakpoint - 1
                && datum.environment_id === 0
            `)
        )
        .encode(
            vl.x().fieldQ("x"),
            vl.y().fieldQ("y"),
            vl.text().value("Trajectory with no damage"),
        );

    return vl.layer(no_damage_path, damage_path, target, damaged_annotation, undamaged_annotation)
        .data(data)
        .transform(
            vl.filter(`datum.x < ${maxAbs} && datum.x > ${-maxAbs} && datum.y < ${maxAbs} && datum.y > ${-maxAbs}`)
        )
        .title({
            text: "Trajectories taken by the Brittle Star",
            anchor: "middle",
            offset: -30,
            dx: 15,
            fontWeight: "normal"
        })
        .config({
            font: "RedHatDisplay",
            title: {
                fontSize: 13
            },
            view: {
                aspect: 1,
                stroke: null,
                strokeOpacity: 0.5,
            }
        })
        .toSpec();
}

function preprocess_data(
    positions_trajectory_csv, 
    breakpoints_trajectory_csv
) {
    const positions_trajectory = load_csv(positions_trajectory_csv);
    const breakpoints_trajectory = load_csv(breakpoints_trajectory_csv);

    // Convert breakpoints into a lookup map
    const breakpointMap = {};
    breakpoints_trajectory.forEach(b => {
        breakpointMap[b.environment_id] = 
            (b.breakpoint === 'inf')
            ? Number.MAX_SAFE_INTEGER
            : parseInt(b.breakpoint);
    });

    // Merge
    return positions_trajectory.map(t => {
        const bp = breakpointMap[t.environment_id];

        return {
            ...t,
            breakpoint: parseInt(bp)
        };
    });
}
