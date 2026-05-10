import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function actions_trajectory_chart(
    actions_trajectory_csv, 
    breakpoints_trajectory_csv
) {
    const data = preprocess_data(
        actions_trajectory_csv, 
        breakpoints_trajectory_csv
    );

    const actions = Array.from({ length: 5 }, (_, i) => i);
    const steps = Array.from({ length: Math.max(...data.map(row => row.step_id))+1 }, (_, i) => i)

    const breakpoint_chart = vl
        .markRect({ color: "#f16161", opacity: 0.15 })
        .transform(
            vl.filter("datum.step_id === datum.breakpoint")
        )
        .encode(
            vl.x().fieldQ("breakpoint"),
            vl.x2().value(400),
            vl.y().value(0),
            vl.y2().value(120)
        );

    const actions_chart = vl
        .markCircle({ opacity: 0.9 })
        .transform(
            vl.filter("datum.step_id < datum.breakpoint || datum.agent_id !== datum.breakpoint_agent_id"),
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
                    title: false,
                    grid: false,
                    labelExpr: "['Leading','Left primary','Left secondary','Right primary','Right secondary'][+datum.value]"
                })
        )
        .height(120)
        .width(400);

    return vl
        .data(data)
        .transform(
            vl.filter("datum.environment_id === 0")
        )
        .facet(
            vl.row().fieldN("agent_id").title(null),
            vl.layer(actions_chart, breakpoint_chart)
        )
        .columns(1)
        .title({
            text: "Taken actions of each agent during evaluation",
            subtitle: "Damaged interval indicated in red",
            anchor: "middle",
            offset: 30
        })
        .config({
            font: "RedHatDisplay",
            title: {
                fontSize: 18
            }
        })
        .toSpec();
}

function preprocess_data(
    actions_trajectory_csv, 
    breakpoints_trajectory_csv
) {
    const actions_trajectory = load_csv(actions_trajectory_csv);
    const breakpoints_trajectory = load_csv(breakpoints_trajectory_csv);

    // Convert breakpoints into a lookup map
    const breakpointMap = {};
    breakpoints_trajectory.forEach(b => {
        breakpointMap[b.environment_id] = {
            breakpoint: (b.breakpoint === 'inf')
                        ? Number.MAX_SAFE_INTEGER
                        : parseInt(b.breakpoint),
            agent_id: b.agent_id
        };
    });

    // Merge
    return actions_trajectory.map(t => {
        const bp = breakpointMap[t.environment_id];

        return {
            ...t,
            breakpoint: bp.breakpoint,
            breakpoint_agent_id: bp.agent_id
        };
    });
}
