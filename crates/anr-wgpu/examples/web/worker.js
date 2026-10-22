import init, { DFN3 } from './pkg/anr_wgpu.js';

let reducer = null;
let queue = Promise.resolve();

async function handle(request) {
  if (!request || (typeof request.id !== 'string' && typeof request.id !== 'number')) {
    throw new TypeError('A request must have a string or number id');
  }
  if (request.type === 'init') {
    if (reducer !== null) throw new Error('DFN3 is already initialized');
    if (!(request.model instanceof ArrayBuffer)) {
      throw new TypeError('init requires a model ArrayBuffer');
    }
    await init();
    reducer = await DFN3.fromBytes(new Uint8Array(request.model));
    self.postMessage({ id: request.id, ok: true, factor: reducer.factor() });
    return;
  }
  if (reducer === null) throw new Error('DFN3 is not initialized');
  switch (request.type) {
    case 'run': {
      let input = request.input;
      if (input instanceof ArrayBuffer) input = new Float32Array(input);
      if (!(input instanceof Float32Array)) {
        throw new TypeError('run requires a Float32Array or ArrayBuffer input');
      }
      const output = await reducer.run(request.sampleRate, input);
      self.postMessage({ id: request.id, ok: true, output }, [output.buffer]);
      break;
    }
    case 'setFactor':
      reducer.setFactor(request.factor);
      self.postMessage({ id: request.id, ok: true, factor: reducer.factor() });
      break;
    case 'reset':
      reducer.reset();
      self.postMessage({ id: request.id, ok: true, factor: reducer.factor() });
      break;
    case 'dispose':
      reducer.free();
      reducer = null;
      self.postMessage({ id: request.id, ok: true });
      break;
    default:
      throw new Error(`Unknown request type: ${request.type}`);
  }
}

self.onmessage = ({ data }) => {
  // The chain retains only the submitted request, not an extra audio copy.
  // Catch each request separately so an error cannot poison later messages.
  queue = queue.then(() => handle(data)).catch((error) => {
    self.postMessage({
      id: data?.id,
      ok: false,
      error: error instanceof Error ? error : new Error(String(error)),
    });
  });
};
