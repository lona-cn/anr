function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function rejects(operation, message) {
  try {
    await operation();
  } catch (error) {
    assert(error instanceof Error, `${message}: expected a JavaScript Error`);
    return { name: error.name, message: error.message };
  }
  throw new Error(message);
}

async function rejectsPromise(operation, message) {
  const promise = operation();
  assert(promise instanceof Promise, `${message}: run must return a Promise`);
  return rejects(() => promise, message);
}

async function binary(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

function compare(actual, expected) {
  assert(actual.length === expected.length, 'Output length differs from input');
  let maximum = 0;
  let squared = 0;
  for (let i = 0; i < expected.length; i++) {
    assert(Number.isFinite(actual[i]), `Non-finite output at sample ${i}`);
    const error = Math.abs(actual[i] - expected[i]);
    maximum = Math.max(maximum, error);
    squared += error * error;
  }
  const rms = Math.sqrt(squared / expected.length);
  assert(maximum < 2e-5 && rms < 2e-6, `Reference mismatch: max=${maximum}, RMS=${rms}`);
  return { maximum, rms, samples: actual.length };
}

const SAMPLE_RATE = 48000;
const FRAME_SIZE = 480;

async function referenceStream(input, expected, setFactor, run, invalidCalls) {
  const actual = new Float32Array(input.length);
  const sections = [
    [0, 12 * FRAME_SIZE, 80],
    [12 * FRAME_SIZE, 24 * FRAME_SIZE, 0],
    [24 * FRAME_SIZE, 36 * FRAME_SIZE, 5],
    [36 * FRAME_SIZE, input.length, 100],
  ];
  for (const [start, end, factor] of sections) {
    await setFactor(factor);
    for (let cursor = start; cursor < end; cursor += 5 * FRAME_SIZE) {
      if (cursor === 5 * FRAME_SIZE) await invalidCalls();
      const next = Math.min(cursor + 5 * FRAME_SIZE, end);
      const output = await run(input.slice(cursor, next));
      assert(output instanceof Float32Array, 'run did not return a Float32Array');
      assert(output.length === next - cursor, 'run changed the chunk length');
      actual.set(output, cursor);
    }
  }
  return compare(actual, expected);
}

function workerClient(workerUrl) {
  const worker = new Worker(workerUrl, { type: 'module' });
  let nextId = 1;
  const pending = new Map();
  const completed = [];
  const fail = (error) => {
    for (const entry of pending.values()) {
      clearTimeout(entry.timeout);
      entry.reject(error);
    }
    pending.clear();
  };
  worker.onmessage = ({ data }) => {
    const entry = pending.get(data.id);
    if (!entry) {
      fail(new Error(`Unexpected Worker response id ${data.id}`));
      return;
    }
    pending.delete(data.id);
    clearTimeout(entry.timeout);
    completed.push(data.id);
    if (data.ok) entry.resolve(data);
    else entry.reject(data.error);
  };
  worker.onerror = (event) => {
    fail(new Error(`Worker failed: ${event.message}`));
  };
  worker.onmessageerror = () => fail(new Error('Worker response could not be deserialized'));
  return {
    completed,
    request(type, fields = {}, transfer = []) {
      const id = nextId++;
      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          pending.delete(id);
          reject(new Error(`Worker ${type} request ${id} timed out`));
        }, 120000);
        pending.set(id, { resolve, reject, timeout });
        try {
          worker.postMessage({ id, type, ...fields }, transfer);
        } catch (error) {
          clearTimeout(timeout);
          pending.delete(id);
          reject(error);
        }
      });
    },
    close() {
      worker.terminate();
      fail(new Error('Worker terminated'));
    },
  };
}

async function workerReference(workerUrl, modelBytes, input, expected) {
  const client = workerClient(workerUrl);
  try {
    // Both requests are submitted before initialization has finished. An invalid
    // model must fail its own request, not poison the subsequent valid init.
    const badModel = new Uint8Array([1, 2, 3]).buffer;
    const invalidInit = rejects(
      () => client.request('init', { model: badModel }, [badModel]),
      'Worker accepted an unrelated model',
    );
    const transferredModel = modelBytes.slice(0);
    const initialized = client.request('init', { model: transferredModel }, [transferredModel]);
    assert(transferredModel.byteLength === 0, 'Worker model was not transferred');
    await Promise.all([invalidInit, initialized]);
    await rejects(() => client.request('init', { model: new ArrayBuffer(0) }), 'Worker accepted duplicate init');

    const setFactor = (factor) => client.request('setFactor', { factor });
    const run = async (samples, asBuffer = false) => {
      const response = client.request('run', {
        sampleRate: SAMPLE_RATE,
        input: asBuffer ? samples.buffer : samples,
      }, [samples.buffer]);
      assert(samples.byteLength === 0, 'Worker input was not transferred');
      return (await response).output;
    };
    const invalidCalls = async () => {
      await rejects(
        () => client.request('run', { sampleRate: 44100, input: new Float32Array(FRAME_SIZE) }),
        'Worker accepted an invalid sample rate',
      );
      await rejects(
        () => client.request('run', { sampleRate: SAMPLE_RATE, input: new Float32Array(0) }),
        'Worker accepted empty input',
      );
    };
    const first = await referenceStream(input, expected, setFactor, run, invalidCalls);
    await setFactor(37);
    const reset = await client.request('reset');
    assert(reset.factor === 37, 'Worker reset changed strength');

    // Queue an entire real stream without awaiting responses. Ordering includes
    // factor mutations and a rejected request between two valid audio chunks.
    const orderStart = client.completed.length;
    const queued = [];
    const actual = new Float32Array(input.length);
    const enqueue = (promise) => queued.push(promise);
    for (const [start, end, factor] of [
      [0, 12 * FRAME_SIZE, 80],
      [12 * FRAME_SIZE, 24 * FRAME_SIZE, 0],
      [24 * FRAME_SIZE, 36 * FRAME_SIZE, 5],
      [36 * FRAME_SIZE, input.length, 100],
    ]) {
      enqueue(setFactor(factor));
      for (let cursor = start; cursor < end; cursor += 5 * FRAME_SIZE) {
        if (cursor === 5 * FRAME_SIZE) {
          enqueue(rejects(() => client.request('unknown'), 'Worker accepted an unknown request'));
        }
        const next = Math.min(cursor + 5 * FRAME_SIZE, end);
        enqueue(run(input.slice(cursor, next), true).then((output) => {
          assert(output instanceof Float32Array, 'Worker did not return a Float32Array');
          assert(output.length === next - cursor, 'Worker changed the chunk length');
          actual.set(output, cursor);
        }));
      }
    }
    await Promise.all(queued);
    const completionOrder = client.completed.slice(orderStart);
    assert(completionOrder.every((id, i) => i === 0 || id === completionOrder[i - 1] + 1),
      'Worker responses were not serialized in request order');
    const queuedReference = compare(actual, expected);
    await client.request('dispose');
    await rejects(
      () => client.request('run', { sampleRate: SAMPLE_RATE, input: new Float32Array(FRAME_SIZE) }),
      'Disposed Worker accepted a run',
    );
    return { first, queuedReference, queuedRequests: queued.length, errorRecovery: true, disposed: true };
  } finally {
    client.close();
  }
}

export async function runReference(packageUrl, modelUrl, workerUrl) {
  assert(workerUrl, 'A module Worker URL is required');
  const pkg = await import(packageUrl);
  await pkg.default();
  const [modelBytes, inputBytes, expectedBytes] = await Promise.all([
    binary(modelUrl),
    binary(new URL('../fixtures/reference_input.f32le', import.meta.url)),
    binary(new URL('../fixtures/reference_output.f32le', import.meta.url)),
  ]);
  const input = new Float32Array(inputBytes);
  const expected = new Float32Array(expectedBytes);
  assert(input.length % FRAME_SIZE === 13, 'Fixture must contain a partial final frame');
  const invalidModel = await rejects(
    () => pkg.DFN3.fromBytes(new Uint8Array([1, 2, 3])), 'Unrelated model was accepted',
  );
  const altered = new Uint8Array(modelBytes.slice(0));
  altered[altered.length - 1] ^= 1;
  await rejects(() => pkg.DFN3.fromBytes(altered), 'Altered model was accepted');
  const reducer = await pkg.DFN3.fromBytes(new Uint8Array(modelBytes));
  let direct;
  try {
    reducer.setFactor(-1);
    assert(reducer.factor() === 0, 'Negative strength was not clamped');
    reducer.setFactor(101);
    assert(reducer.factor() === 100, 'Excessive strength was not clamped');
    reducer.setFactor(NaN);
    assert(reducer.factor() === 100, 'NaN changed strength');
    const invalidCalls = async () => {
      await rejects(() => reducer.run(44100, new Float32Array(FRAME_SIZE)), 'Invalid sample rate was accepted');
      await rejects(() => reducer.run(SAMPLE_RATE, new Float32Array(0)), 'Empty input was accepted');
    };
    await invalidCalls();
    const run = () => referenceStream(input, expected,
      (factor) => reducer.setFactor(factor),
      (samples) => reducer.run(SAMPLE_RATE, samples),
      invalidCalls);
    const first = await run();
    reducer.setFactor(37);
    reducer.reset();
    assert(reducer.factor() === 37, 'Reset changed strength');
    const afterReset = await run();

    // The stream must acquire its run permit before returning its Promise.
    // Immediate access and access after the first poll must both reject without
    // cancelling or corrupting the original operation.
    reducer.reset();
    reducer.setFactor(80);
    const active = reducer.run(SAMPLE_RATE, input.slice(0, 5 * FRAME_SIZE));
    const immediate = [
      rejects(() => reducer.setFactor(0), 'Immediate setFactor raced an in-flight run'),
      rejects(() => reducer.reset(), 'Immediate reset raced an in-flight run'),
      rejects(() => reducer.factor(), 'Immediate factor read an in-flight stream'),
      rejectsPromise(() => reducer.run(SAMPLE_RATE, input.slice(0, FRAME_SIZE)), 'Immediate concurrent run was accepted'),
    ];
    await Promise.resolve();
    const operations = await Promise.allSettled([
      active,
      ...immediate,
      rejects(() => reducer.setFactor(0), 'setFactor raced an in-flight run'),
      rejects(() => reducer.reset(), 'reset raced an in-flight run'),
      rejects(() => reducer.factor(), 'factor read an in-flight stream'),
      rejectsPromise(() => reducer.run(SAMPLE_RATE, input.slice(0, FRAME_SIZE)), 'Concurrent run was accepted'),
    ]);
    for (const result of operations) {
      if (result.status === 'rejected') throw result.reason;
    }
    const activeOutput = operations[0].value;
    const concurrent = operations.slice(1).map((result) => result.value);
    const inFlight = compare(activeOutput, expected.subarray(0, 5 * FRAME_SIZE));
    reducer.setFactor(0);
    reducer.reset();
    assert(reducer.factor() === 0, 'Mutations did not recover after the active run');
    const afterRejectedConcurrency = await run();

    // The short final call must advance a complete zero-padded model frame.
    // Its next frame must match an explicitly padded replay, not just truncate
    // the right number of samples from the independent reference fixture.
    const continuation = input.slice(3 * FRAME_SIZE, 4 * FRAME_SIZE);
    const afterPartial = await reducer.run(SAMPLE_RATE, continuation);
    reducer.reset();
    const explicitlyPadded = await referenceStream(input, expected,
      (factor) => reducer.setFactor(factor),
      async (samples) => {
        if (samples.length % FRAME_SIZE === 0) return reducer.run(SAMPLE_RATE, samples);
        const padded = new Float32Array(Math.ceil(samples.length / FRAME_SIZE) * FRAME_SIZE);
        padded.set(samples);
        return (await reducer.run(SAMPLE_RATE, padded)).subarray(0, samples.length);
      },
      invalidCalls);
    const paddingState = compare(await reducer.run(SAMPLE_RATE, continuation), afterPartial);
    direct = {
      first, afterReset, inFlight, afterRejectedConcurrency,
      explicitlyPadded, paddingState, concurrent, invalidModel,
    };
  } finally {
    reducer.free();
  }
  const released = await pkg.DFN3.fromBytes(new Uint8Array(modelBytes));
  released.setFactor(80);
  const afterImmediateFree = released.run(SAMPLE_RATE, input.slice(0, 5 * FRAME_SIZE));
  // No microtask may run between starting this operation and freeing its JS
  // wrapper: the Promise must already own the live stream, not a raw pointer.
  released.free();
  direct.afterImmediateFree = compare(
    await afterImmediateFree, expected.subarray(0, 5 * FRAME_SIZE),
  );
  const worker = await workerReference(workerUrl, modelBytes, input, expected);
  return { direct, worker, sampleRate: SAMPLE_RATE, samples: input.length, partialFrameSamples: 13 };
}
