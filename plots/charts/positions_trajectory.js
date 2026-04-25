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
            vl.x()
                .fieldQ("x")
                .axis(null),
            vl.y()
                .fieldQ("y")
                .axis(null),
            vl.order()
                .fieldO("step_id")
        );

    const start_and_end = vl.markCircle({ color: "#ff0000", size: 60 })
        .transform(
            vl.filter("!datum.in_trajectory")
        )
        .encode(
            vl.x().fieldQ("x"),
            vl.y().fieldQ("y")
        );
    
    const labels = vl.markText({
        dx: 20,
        fontSize: 12,
        fontWeight: "bold"
    })
    .transform(
        vl.filter("datum.in_trajectory == false"),
        vl.calculate(
            "datum.step_id == 0 ? 'Start' : 'End'"
        ).as("label")
    )
    .encode(
        vl.x().fieldQ("x"),
        vl.y().fieldQ("y"),
        vl.text().fieldN("label")
    );

    return vl.layer(path, start_and_end, labels)
        .facet(
            vl.column().fieldO("environment_id").title("Environment")
        )
        .data(trajectory)
        .title({
            text: "Trajectory taking by the Brittle Star",
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
