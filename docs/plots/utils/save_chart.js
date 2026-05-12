import * as vega from 'vega';
import fs from 'fs';
import { compile } from 'vega-lite';


export async function save_chart_svg(chart, filename) {
  const compiled = compile(chart).spec;
  const view = new vega.View(
    vega.parse(compiled),
    { renderer: "none" } // headless
  );

  const svg = await view.toSVG();
  fs.writeFileSync(filename + '.svg', svg);
}

export async function save_chart_png(chart, filename) {
  const compiled = compile(chart).spec;
  const view = new vega.View(
    vega.parse(compiled),
    { renderer: "none" } // headless
  );

  const canvas = await view.toCanvas(2);
  fs.writeFileSync(filename + '.png', canvas.toBuffer());
}
