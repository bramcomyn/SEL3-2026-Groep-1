import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function learning_curve_comparison_chart(learning_curve_comparison_csv) {
    const W = 40; // Moving average window width
    const damaged = 1;      // ID of damaged brittle star
    const undamaged = 2;    // ID of undamaged brittle star

    const titleFontSize = 18;
    const subtitleFontSize = 14;
    
    // Transform into Vega-lite friendly data
    const data = load_csv(learning_curve_comparison_csv)
        .flatMap(d => [
            {
                step: d.step,
                brittle_star: damaged,
                terminated_count: d.brittle_star_1
            },
            {
                step: d.step,
                brittle_star: undamaged,
                terminated_count: d.brittle_star_2
            }
        ]);

    // Additional needed data
    const max_step_damaged = Math.max(...data.filter(row => row.brittle_star === damaged && row.terminated_count !== null).map(row => row.step));
    const max_step_undamaged = Math.max(...data.filter(row => row.brittle_star === undamaged && row.terminated_count !== null).map(row => row.step));

    // Charts
    const learning_curves = vl
        .markLine()
        .encode(
            vl.x()
                .fieldQ("step")
                .axis({ grid: false, format: '~s', titleFontSize: subtitleFontSize })
                .title("Training episode"),
            vl.y()
                .fieldQ("terminated_moving_avg")
                .axis({ grid: false, titleFontSize: subtitleFontSize })
                .title("Amount of target reaches"),
            vl.color()
                .fieldN("brittle_star")
                .scale({
                    domain: [undamaged, damaged],
                    range: ["#2D8CA8", "#f16161"] 
                })
                .legend(null)
        )
        .height(150)
        .width(500);

    const training_time_annotation = vl
        .markText({ align: "left", dx: 5, fontSize: subtitleFontSize })
        .transform(
            vl.filter(`
                (datum.step === ${max_step_damaged} && datum.brittle_star === ${damaged}) 
                || (datum.step === ${max_step_undamaged} && datum.brittle_star === ${undamaged})
            `),
            vl.calculate(`
                datum.brittle_star === ${damaged}
                    ? "Damage robust - 8h"
                    : "Not damage robust - 3h"
            `).as("status_label")
        )
        .encode(
            vl.x().fieldQ("step"),
            vl.y().fieldQ("terminated_moving_avg"),
            vl.text().fieldN("status_label"),
            vl.color()
                .fieldN("brittle_star")
                .scale({
                    domain: [undamaged, damaged],
                    range: ["#2D8CA8", "#f16161"] 
                })
                .legend(null)
        );

    return vl
        .data(data)
        .transform(
            // Compute moving average
            vl.window([
                {
                    op: "mean",
                    field: "terminated_count",
                    as: "terminated_moving_avg"
                }
            ])
                .frame([-W, 0])
                .groupby(["brittle_star"])
                .sort([{ field: "step" }]),
            vl.filter(`
                (datum.step <= ${max_step_damaged} && datum.brittle_star === ${damaged}) 
                || (datum.step <= ${max_step_undamaged} && datum.brittle_star === ${undamaged})
            `)
        )
        .layer(learning_curves, training_time_annotation)
        .title({
            text: "Performant damage robust Brittle Star requires double more training",
            fontSize: titleFontSize,
            subtitle: "Compares amount of target reaches",
            subtitleFontSize: subtitleFontSize,
            offset: 22.5
        })
        .config({
            font: "RedHatDisplay",
            view: { stroke: null }
        })
        .toSpec();
}
