import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildSync } from 'esbuild';

const root_dir = fileURLToPath(new URL('..', import.meta.url));
const temp_dir = mkdtempSync(join(tmpdir(), 'jimmytorch-gpu-cards-'));
const bundle_path = join(temp_dir, 'cards.cjs');
buildSync({ stdin: { contents: `export {MetricCards} from './src/section1_live/MetricCards'; export {renderToStaticMarkup} from 'react-dom/server'; export {createElement} from 'react';`, resolveDir: root_dir }, bundle: true, platform: 'node', format: 'cjs', jsx: 'automatic', outfile: bundle_path });
const { MetricCards: metric_cards, renderToStaticMarkup: render_markup, createElement: create_element } = await import(pathToFileURL(bundle_path));
rmSync(temp_dir, { recursive: true });
const progress = { overall: 1, total: 10, percent: 10, epoch: 1, epochs: 1, step: 1, steps_per_epoch: 10, elapsed: 10, rate: 0.1, remaining: 90 };
function render(system) { return render_markup(create_element(metric_cards, { progress, system })); }
function gpu(index, training_ranks = []) {
  return { id: `GPU-${index}`, index, name: `Device ${index}`, utilization: 45, memory_used: 1024, memory_total: 2048, training_ranks };
}

test('renders every GPU in physical index order for one, two and eight devices', () => {
  for (const count of [1, 2, 8]) {
    const markup = render({ gpus: Array.from({ length: count }, (_, index) => gpu(index)).reverse() });
    assert.equal((markup.match(/class="gpu-card"/g) ?? []).length, count);
    for (let index = 1; index < count; index++) assert.ok(markup.indexOf(`GPU ${index - 1}</strong>`) < markup.indexOf(`GPU ${index}</strong>`));
  }
});
test('distinguishes unknown, assigned and unassigned GPUs', () => {
  const markup = render({ gpus: [gpu(0, null), gpu(1, [3]), gpu(2)] });
  assert.match(markup, /Assignment unknown/);
  assert.match(markup, /Training · rank 3/);
  assert.match(markup, /Not assigned to this run/);
});
test('supports legacy snapshots and distinguishes no devices from missing telemetry', () => {
  assert.match(render({ gpu_util: 70, gpu_mem_used: 1024, gpu_mem_total: 2048 }), /GPU 0/);
  assert.match(render({ gpus: [] }), /No GPUs detected/);
  assert.match(render({ gpus: null }), /GPU telemetry unavailable/);
  assert.match(render({}), /GPU telemetry unavailable/);
});
test('partial measurements remain visibly unavailable', () => {
  const markup = render({ gpus: [{ ...gpu(0), utilization: null, memory_used: null }] });
  assert.equal((markup.match(/class="is-unavailable"/g) ?? []).length, 2);
  assert.match(markup, /—/);
});
test('groups machines without duplicating job progress and retains stale readings', () => {
  const markup = render_markup(create_element(metric_cards, {progress, system: {}, nodes: [
    {node_id: '0', hostname: 'primary', ranks: [0,1], connected: true, received_at: 1, system: {gpus: [gpu(0,[0]), gpu(1,[1])]}},
    {node_id: '1', hostname: 'worker', ranks: [2,3], connected: false, received_at: 1, system: {gpus: [gpu(0,[2]), gpu(1,[3])]}},
  ]}));
  assert.equal((markup.match(/class="gpu-card"/g) ?? []).length, 4);
  assert.equal((markup.match(/>Elapsed</g) ?? []).length, 1);
  assert.match(markup, /Machine 0 · primary/);
  assert.match(markup, /Machine 1 · worker/);
  assert.match(markup, /Telemetry inactive \/ stale/);
  assert.match(markup, /rank 3/);
});
