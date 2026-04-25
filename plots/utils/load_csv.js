import fs from 'fs';
import { csvParse, autoType } from 'd3-dsv';

export function load_csv(path_to_csv) {
  const raw = fs.readFileSync(path_to_csv, 'utf-8');
  return csvParse(raw, autoType);
}
