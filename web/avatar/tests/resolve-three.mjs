// Node ESM hook: resolve the bare specifier "three" to the vendored copy,
// mirroring the <script type="importmap"> used in the browser.
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const THREE = pathToFileURL(
  path.resolve(import.meta.dirname, '../../vendor/three/three.module.js')
).href;

export async function resolve(specifier, context, nextResolve) {
  if (specifier === 'three') return { url: THREE, shortCircuit: true };
  return nextResolve(specifier, context);
}
