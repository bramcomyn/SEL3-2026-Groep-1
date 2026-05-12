import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

export function learning_curve_comparison_chart(learning_curve_comparison_csv) {
    const W = 40; // Moving average window width

    const damaged = 1;      // ID of damaged brittle star
    const undamaged = 2;    // ID of undamaged brittle star

    const damaged_label = "Damage included (±8h)";
    const undamaged_label = "No damage included (±3h)";

    const title = "Achieving similar performance with damage robustness";
    const titleNext = "requires more than twice as much time"

    const ylabel = "Amount of reached targets";
    const xlabel = "Training episode";

    const height = 150;
    const width = 500;

    const titleFontSize = 18;
    const annotationFontSize = 13;
    const titleOffset = 22.5;
    
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
                .axis({ grid: false, format: '~s', titleFontSize: annotationFontSize, titleFontWeight: "normal" })
                .title(xlabel),
            vl.y()
                .fieldQ("terminated_moving_avg")
                .axis({ grid: false, titleFontSize: annotationFontSize, titleFontWeight: "normal" })
                .title(ylabel),
            vl.color()
                .fieldN("brittle_star")
                .scale({
                    domain: [undamaged, damaged],
                    range: ["#2D8CA8", "#f16161"] 
                })
                .legend(null)
        )
        .height(height)
        .width(width);

    const training_time_annotation_damaged = vl
        .markText({ dx: -50, dy: annotationFontSize + 10, fontSize: annotationFontSize })
        .transform(
            vl.filter(`
                datum.step === ${max_step_damaged} && datum.brittle_star === ${damaged} 
            `)
        )
        .encode(
            vl.x().fieldQ("step"),
            vl.y().fieldQ("terminated_moving_avg"),
            vl.text().value(damaged_label),
            vl.color()
                .fieldN("brittle_star")
                .scale({
                    domain: [undamaged, damaged],
                    range: ["#2D8CA8", "#f16161"] 
                })
                .legend(null)
        );

    const training_time_annotation_undamaged = vl
        .markText({ dx: 11, dy: -annotationFontSize, fontSize: annotationFontSize })
        .transform(
            vl.filter(`
                (datum.step === ${max_step_undamaged} && datum.brittle_star === ${undamaged})
            `)
        )
        .encode(
            vl.x().fieldQ("step"),
            vl.y().fieldQ("terminated_moving_avg"),
            vl.text().value(undamaged_label),
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
        .layer(
            learning_curves, 
            training_time_annotation_damaged, 
            training_time_annotation_undamaged
        )
        .title({
            text: title,
            fontSize: titleFontSize,
            subtitle: titleNext,
            subtitleFontSize: titleFontSize,
            offset: titleOffset,
            fontWeight: "normal"
        })
        .config({
            font: "RedHatDisplay",
            view: { stroke: null }
        })
        .toSpec();
}
