// Minimal test framework for WebGPU compute pipelines
export function assertEqual(actual, expected, message) {
	if (actual !== expected) {
		throw new Error(`Assertion failed: ${message} (expected ${expected}, got ${actual})`);
	}
}

export function assertArrayEqual(actual, expected, message) {
	if (actual.length !== expected.length) {
		throw new Error(`Assertion failed: ${message} (array length)`);
	}
	for (let i = 0; i < actual.length; ++i) {
		if (actual[i] !== expected[i]) {
			throw new Error(`Assertion failed: ${message} (at index ${i}, expected ${expected[i]}, got ${actual[i]})`);
		}
	}
}

export function test(name, fn) {
	// Support async and sync tests
	try {
		const result = fn();
		if (result && typeof result.then === 'function') {
			// Async test
			result.then(() => {
				console.log(`PASS: ${name}`);
			}).catch(e => {
				console.error(`FAIL: ${name}\n  ${e && e.message ? e.message : e}`);
			});
		} else {
			// Sync test
			console.log(`PASS: ${name}`);
		}
	} catch (e) {
		console.error(`FAIL: ${name}\n  ${e && e.message ? e.message : e}`);
	}
}
