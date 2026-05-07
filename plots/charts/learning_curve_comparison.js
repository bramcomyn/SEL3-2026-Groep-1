import * as vl from 'vega-lite-api';
import { load_csv } from '../utils/load_csv.js';
import { registerFont } from 'canvas';

registerFont('/usr/local/share/fonts/Red_Hat_Display/static/RedHatDisplay-Regular.ttf', { family: 'RedHatDisplay' });

export function learning_curve_comparison_chart(learning_curve_comparison_csv) {
    const W = 40; // Moving average window width
    const damaged = 1;      // ID of damaged brittle star
    const undamaged = 2;    // ID of undamaged brittle star
    
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

    const learning_curves = vl
        .markLine()
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
            .sort([{ field: "step" }])
        )
        .encode(
            vl.x()
                .fieldQ("step")
                .axis({ grid: false, format: '~s' })
                .title("Episode"),
            vl.y()
                .fieldQ("terminated_moving_avg")
                .axis({ grid: false })
                .title("Terminated count"),
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
        

    return vl
        .data(data)
        .layer(learning_curves)
        .title({
            text: "Learning curves",
            fontSize: 18,
            subtitle: "Damage and no damage",
            subtitleFontSize: 16
        })
        .config({
            font: "RedHatDisplay",
            view: { stroke: null }
        })
        .toSpec();
}
