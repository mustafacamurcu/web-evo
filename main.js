import {
	makeShaderDataDefinitions,
	makeStructuredView,
	getSizeAndAlignmentOfUnsizedArrayElement
} from 'https://greggman.github.io/webgpu-utils/dist/2.x/webgpu-utils.module.js';

// also imports structs.wgsl by default
async function fetchShaderCode(url) {
	const structs = await fetch('structs.wgsl');
	const response = await fetch(url);
	if (!response.ok) throw new Error('Failed to load shader');
	const code = await response.text();
	const structsCode = await structs.text();
	return structsCode + '\n' + code;
}

function createRenderPipeline(device, code, context) {
	const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
	context.configure({
		device,
		format: presentationFormat,
	});
	const renderModule = device.createShaderModule({ code });

	const pipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: renderModule,
			buffers: [
				{
					arrayStride: 32, // position + direction = 16 bytes
					stepMode: 'instance',
					attributes: [
						{
							// bots position
							shaderLocation: 0,
							offset: 0,
							format: 'float32x2',
						},
						{
							// bot direction
							shaderLocation: 1,
							offset: 8,
							format: 'float32x2',
						},
						{
							// bot die_stay_breed
							shaderLocation: 3,
							offset: 16,
							format: 'uint32',
						},
						{
							// bot age
							shaderLocation: 4,
							offset: 20,
							format: 'uint32',
						},
					],
				},
				{
					// vertex buffer
					arrayStride: 8,
					stepMode: 'vertex',
					attributes: [
						{
							// vertex positions
							shaderLocation: 2,
							offset: 0,
							format: 'float32x2',
						},
					],
				},
			],
		},
		fragment: {
			module: renderModule,
			targets: [
				{
					format: presentationFormat,
				},
			],
		},
		primitive: {
			topology: 'triangle-list',
		},
	});

	const renderPassDescriptor = {
		colorAttachments: [
			{
				// view: <- to be filled out when we render
				clearValue: [0.3, 0.3, 0.3, 1],
				loadOp: 'clear',
				storeOp: 'store',
			},
		],
	};

	const vertexBufferData = new Float32Array([ // long triangle
		-0.002, -0.004, 0.002,
		-0.004, 0.00, 0.004,
	]);

	const longTriangleVertexBuffer = device.createBuffer({
		size: vertexBufferData.byteLength,
		usage: GPUBufferUsage.VERTEX,
		mappedAtCreation: true,
	});
	new Float32Array(longTriangleVertexBuffer.getMappedRange()).set(vertexBufferData);
	longTriangleVertexBuffer.unmap();

	return { pipeline, renderPassDescriptor, longTriangleVertexBuffer };
}

function createStorageBuffer(device, storageDefinition, objectCount, usage, data) {
	const objectSize = getSizeAndAlignmentOfUnsizedArrayElement(storageDefinition).size;
	const totalSize = objectSize * objectCount;
	const structuredView = makeStructuredView(storageDefinition, new ArrayBuffer(totalSize));
	const buffer = device.createBuffer({
		size: totalSize,
		usage: GPUBufferUsage.STORAGE | usage,
	});

	if (data) {
		structuredView.set(data);
		device.queue.writeBuffer(buffer, 0, structuredView.arrayBuffer);
	}
	return buffer;
}

function createBotStepperPipeline(device, code) {
	const module = device.createShaderModule({
		label: 'bot stepper compute module',
		code,
	});

	const pipeline = device.createComputePipeline({
		label: 'bot stepper compute pipeline',
		layout: 'auto',
		compute: {
			module,
		},
	});

	const defs = makeShaderDataDefinitions(code);
	let bot_data = [];
	const bot_count = 100000;
	for (let i = 0; i < MAX_BOTS; ++i) {
		if (i < bot_count * 2) {
			let bot = {
				position: [2 * (Math.random() - 0.5), 2 * (Math.random() - 0.5)],
				velocity: [0.00002 * (Math.random() - 0.5), 0.00002 * (Math.random() - 0.5)],
				energy: 0,
				// die_stay_breed: Math.floor(Math.random() * 3), // alive
				die_stay_breed: ((i + 1) % 2), // alive
				id: i,
				age: 0,
			}
			bot_data.push(bot);
		}
		else {
			let bot = {
				position: [10, 10], // offscreen
				velocity: [0, 0],
				energy: 0,
				die_stay_breed: 0, // dead
				id: i,
				age: 0,
			}
			bot_data.push(bot);
		}
	}
	// Create Buffers
	const botsBuffer = createStorageBuffer(device, defs.storages.bots, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX, bot_data);

	const scratchBuffer = createStorageBuffer(device, defs.storages.scratchBuffer, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, null);

	// Bind Groups
	const bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: botsBuffer,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: scratchBuffer,
				},
			},
		],
	});

	return { pipeline, bindGroup, botsBuffer, scratchBuffer };
}

function createPrefixSumPipeline(device, code, scratchBuffer) {
	const module = device.createShaderModule({ code, });
	const pipeline1 = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'l1',
		},
	});
	const pipeline2 = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'l2',
		},
	});

	const pipeline3 = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module,
			entryPoint: 'l3',
		},
	});

	const defs = makeShaderDataDefinitions(code);

	const L1Buffer = createStorageBuffer(device, defs.storages.l1Buffer, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		null);

	const L2Buffer = createStorageBuffer(device, defs.storages.l2Buffer, MAX_BOTS / 64 + 1,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		null);

	const L3Buffer = createStorageBuffer(device, defs.storages.l3Buffer, MAX_BOTS / 64 / 64 + 1,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
		null);

	// Bind Groups
	const L1BindGroup = device.createBindGroup({
		layout: pipeline1.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: scratchBuffer,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: L1Buffer,
				},
			},
		],
	});

	const L2BindGroup = device.createBindGroup({
		layout: pipeline2.getBindGroupLayout(0),
		entries: [
			{
				binding: 1,
				resource: {
					buffer: L1Buffer,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: L2Buffer,
				},
			},
		],
	});

	const L3BindGroup = device.createBindGroup({
		layout: pipeline3.getBindGroupLayout(0),
		entries: [
			{
				binding: 2,
				resource: {
					buffer: L2Buffer,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: L3Buffer,
				},
			},
		],
	});

	return { pipeline1, pipeline2, pipeline3, L1BindGroup, L2BindGroup, L3BindGroup, L1Buffer, L2Buffer, L3Buffer };
}

function createReaperPipeline(device, code, scratchBuffer, botsBuffer, L1Buffer, L2Buffer, L3Buffer) {
	const module = device.createShaderModule({ code, });
	const pipeline = device.createComputePipeline({
		layout: 'auto',
		compute: {
			module,
		},
	});

	const bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: scratchBuffer,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: botsBuffer,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: L1Buffer,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: L2Buffer,
				},
			},
			{
				binding: 4,
				resource: {
					buffer: L3Buffer,
				},
			},
		],
	});

	return { pipeline, bindGroup };
}

class Bot {
	constructor() {
		this.position = [0, 0];
		this.velocity = [0, 0];
		this.die_stay_breed = 0;
		this.age = 0;
		this.energy = 0;
		this.id = 0;
	}
}

async function dumpBuffer(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);

	const numBots = data.byteLength / 32;
	const bots = [];

	for (let i = 0; i < numBots; i++) {
		const byteOffset = i * 32;
		const bot = new Bot();

		bot.position[0] = view.getFloat32(byteOffset + 0, true);
		bot.position[1] = view.getFloat32(byteOffset + 4, true);
		bot.velocity[0] = view.getFloat32(byteOffset + 8, true);
		bot.velocity[1] = view.getFloat32(byteOffset + 12, true);
		bot.die_stay_breed = view.getUint32(byteOffset + 16, true);
		bot.age = view.getUint32(byteOffset + 20, true);
		bot.energy = view.getUint32(byteOffset + 24, true);
		bot.id = view.getUint32(byteOffset + 28, true);

		bots.push(bot);
	}
	return bots;
}

async function dumpBufferInt(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);

	const u32Array = [];
	for (let i = 0; i < view.byteLength / 4; i++) {
		u32Array.push(view.getUint32(i * 4, true)); // The `true` is for little-endian
	}
	return u32Array;
}


// Constants
const MAX_BOTS = 64 * 64 * 64; // increase by adding a level to prefix sum
async function main() {
	// WebGPU setup
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice({
		requiredFeatures: ['timestamp-query'],
	});
	const canvas = document.querySelector('canvas');
	const context = canvas.getContext('webgpu');

	// Bot Stepper Compute Pipeline
	const computeShaderCode = await fetchShaderCode('bot_step.comp');
	const botStepperPipeline = createBotStepperPipeline(device, computeShaderCode);

	// Prefix Sum Pipeline
	const prefixSumShaderCode = await fetchShaderCode('prefix_sum.comp');
	const prefixSumPipeline = createPrefixSumPipeline(device, prefixSumShaderCode, botStepperPipeline.scratchBuffer);

	// Reaper Pipeline
	const reaperShaderCode = await fetchShaderCode('reaper.comp');
	const reaperPipeline = createReaperPipeline(device, reaperShaderCode, botStepperPipeline.scratchBuffer, botStepperPipeline.botsBuffer,
		prefixSumPipeline.L1Buffer, prefixSumPipeline.L2Buffer, prefixSumPipeline.L3Buffer);

	// Render Pipeline
	const renderShaderCode = await fetchShaderCode('shader.render');
	const renderPipeline = createRenderPipeline(device, renderShaderCode, context);


	// --- FPS & IPS display logic ---
	let paused = false;
	let lastIPSUpdate = performance.now();
	let frameCount = 0;
	let iterationCount = 0;
	let ips = 0;
	let fps = 0;

	const fpsElem = document.getElementById('fps');
	const ipsElem = document.getElementById('ms');
	// Use the renderTime span from HTML
	const renderTimeElem = document.getElementById('renderTime');
	const prefix1TimeElem = document.getElementById('prefix1Time');
	const prefix2TimeElem = document.getElementById('prefix2Time');
	const prefix3TimeElem = document.getElementById('prefix3Time');
	const reaperTimeElem = document.getElementById('reaperTime');
	const stepTimeElem = document.getElementById('stepTime');
	const repeatCountElem = document.getElementById('repeatCount');
	const computeTimeElem = document.getElementById('computeTime');
	const totalTimeElem = document.getElementById('totalTime');


	const PASSES = 5; // step, prefix1, prefix2, prefix3, reaper
	const QUERIES_PER_PASS = 2;
	const repeat = 50; // or your chosen repeat value
	const timestampSampleCount = 100;
	const TOTAL_QUERIES = PASSES * QUERIES_PER_PASS * timestampSampleCount + 2; // +2 for render pass

	// WebGPU timestamp query setup
	let timestampQuerySet = null;
	let timestampResolveBuffer = null;
	let timestampStagingBuffer = null;
	let timestampPeriod = 1;
	if (device.features && device.features.has('timestamp-query')) {
		timestampQuerySet = device.createQuerySet({
			type: 'timestamp',
			count: TOTAL_QUERIES
		});
		timestampResolveBuffer = device.createBuffer({
			size: TOTAL_QUERIES * 8,
			usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
		});
		timestampStagingBuffer = device.createBuffer({
			size: TOTAL_QUERIES * 8,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		if (device.limits && device.limits.timestampPeriod) {
			timestampPeriod = device.limits.timestampPeriod;
		}
	}

	window.addEventListener('keydown', async (e) => {
		if (e.code === 'Space') {
			paused = !paused;
			if (!paused) requestAnimationFrame(frame);
			else {
				var bots = await dumpBuffer(device, botStepperPipeline.botsBuffer);
				// var l1buffer = await dumpBufferInt(device, prefixSumPipeline.L1Buffer);
				// var l2buffer = await dumpBufferInt(device, prefixSumPipeline.L2Buffer);
				// var l3buffer = await dumpBufferInt(device, prefixSumPipeline.L3Buffer);
				console.log(bots.slice(0, 100));
				// console.log('L1 buffer:', l1buffer);
				// console.log('L2 buffer:', l2buffer);
				// console.log('L3 buffer:', l3buffer);
			}
		}
	});
	async function frame() {
		if (paused) return;

		const bots1 = await dumpBuffer(device, botStepperPipeline.botsBuffer);
		// Step the computation many times
		const encoder = device.createCommandEncoder({});
		for (let i = 0; i < repeat; ++i) {
			if (i < timestampSampleCount) {
				let base = i * PASSES * 2;
				// Step bots in botsBuffer (with timestamp)
				const stepPass = encoder.beginComputePass({
					timestampWrites: {
						querySet: timestampQuerySet,
						beginningOfPassWriteIndex: base + 0,
						endOfPassWriteIndex: base + 1,
					}
				});
				stepPass.setPipeline(botStepperPipeline.pipeline);
				stepPass.setBindGroup(0, botStepperPipeline.bindGroup);
				stepPass.dispatchWorkgroups(MAX_BOTS / 64);
				stepPass.end();
				// Prefix1
				const prefixSumPass1 = encoder.beginComputePass({
					timestampWrites: {
						querySet: timestampQuerySet,
						beginningOfPassWriteIndex: base + 2,
						endOfPassWriteIndex: base + 3,
					}
				});
				prefixSumPass1.setPipeline(prefixSumPipeline.pipeline1);
				prefixSumPass1.setBindGroup(0, prefixSumPipeline.L1BindGroup);
				prefixSumPass1.dispatchWorkgroups(MAX_BOTS / 64 / 64);
				prefixSumPass1.end();

				// Prefix2
				const prefixSumPass2 = encoder.beginComputePass({
					timestampWrites: {
						querySet: timestampQuerySet,
						beginningOfPassWriteIndex: base + 4,
						endOfPassWriteIndex: base + 5,
					}
				});
				prefixSumPass2.setPipeline(prefixSumPipeline.pipeline2);
				prefixSumPass2.setBindGroup(0, prefixSumPipeline.L2BindGroup);
				prefixSumPass2.dispatchWorkgroups(MAX_BOTS / 64 / 64 / 64);
				prefixSumPass2.end();

				// Prefix3
				const prefixSumPass3 = encoder.beginComputePass({
					timestampWrites: {
						querySet: timestampQuerySet,
						beginningOfPassWriteIndex: base + 6,
						endOfPassWriteIndex: base + 7,
					}
				});
				prefixSumPass3.setPipeline(prefixSumPipeline.pipeline3);
				prefixSumPass3.setBindGroup(0, prefixSumPipeline.L3BindGroup);
				prefixSumPass3.dispatchWorkgroups(1);
				prefixSumPass3.end();

				// Reaper
				const reaperPass = encoder.beginComputePass({
					timestampWrites: {
						querySet: timestampQuerySet,
						beginningOfPassWriteIndex: base + 8,
						endOfPassWriteIndex: base + 9,
					}
				});
				reaperPass.setPipeline(reaperPipeline.pipeline);
				reaperPass.setBindGroup(0, reaperPipeline.bindGroup);
				reaperPass.dispatchWorkgroups(MAX_BOTS / 64);
				reaperPass.end();
			} else {
				const stepPass = encoder.beginComputePass({});
				stepPass.setPipeline(botStepperPipeline.pipeline);
				stepPass.setBindGroup(0, botStepperPipeline.bindGroup);
				stepPass.dispatchWorkgroups(MAX_BOTS / 64);
				stepPass.end();
				// Prefix1
				const prefixSumPass1 = encoder.beginComputePass({});
				prefixSumPass1.setPipeline(prefixSumPipeline.pipeline1);
				prefixSumPass1.setBindGroup(0, prefixSumPipeline.L1BindGroup);
				prefixSumPass1.dispatchWorkgroups(MAX_BOTS / 64 / 64);
				prefixSumPass1.end();

				// Prefix2
				const prefixSumPass2 = encoder.beginComputePass({});
				prefixSumPass2.setPipeline(prefixSumPipeline.pipeline2);
				prefixSumPass2.setBindGroup(0, prefixSumPipeline.L2BindGroup);
				prefixSumPass2.dispatchWorkgroups(MAX_BOTS / 64 / 64 / 64);
				prefixSumPass2.end();

				// Prefix3
				const prefixSumPass3 = encoder.beginComputePass({});
				prefixSumPass3.setPipeline(prefixSumPipeline.pipeline3);
				prefixSumPass3.setBindGroup(0, prefixSumPipeline.L3BindGroup);
				prefixSumPass3.dispatchWorkgroups(1);
				prefixSumPass3.end();

				// Reaper
				const reaperPass = encoder.beginComputePass({});
				reaperPass.setPipeline(reaperPipeline.pipeline);
				reaperPass.setBindGroup(0, reaperPipeline.bindGroup);
				reaperPass.dispatchWorkgroups(MAX_BOTS / 64);
				reaperPass.end();
			}
		}

		// render
		renderPipeline.renderPassDescriptor.colorAttachments[0].view = context
			.getCurrentTexture()
			.createView();
		renderPipeline.renderPassDescriptor.timestampWrites = {
			querySet: timestampQuerySet,
			beginningOfPassWriteIndex: PASSES * 2 * timestampSampleCount,
			endOfPassWriteIndex: PASSES * 2 * timestampSampleCount + 1,
		};

		// --- WebGPU timestamp queries for render pass ---
		let useTimestamp = timestampQuerySet !== null;

		const renderPass = encoder.beginRenderPass(renderPipeline.renderPassDescriptor);
		renderPass.setPipeline(renderPipeline.pipeline);
		renderPass.setVertexBuffer(0, botStepperPipeline.botsBuffer);
		renderPass.setVertexBuffer(1, renderPipeline.longTriangleVertexBuffer);
		renderPass.draw(3, MAX_BOTS);
		renderPass.end();

		// Resolve timestamp queries if available
		if (timestampQuerySet) {
			encoder.resolveQuerySet(timestampQuerySet, 0, TOTAL_QUERIES, timestampResolveBuffer, 0);
			encoder.copyBufferToBuffer(timestampResolveBuffer, 0, timestampStagingBuffer, 0, TOTAL_QUERIES * 8);
		}

		const commandBuffer = encoder.finish();
		device.queue.submit([commandBuffer]);

		// Read and display average pass times if available
		await timestampStagingBuffer.mapAsync(GPUMapMode.READ);
		const array = new BigUint64Array(timestampStagingBuffer.getMappedRange());
		// For each pass, average over all iterations
		const passNames = [
			{ elem: stepTimeElem, label: 'Step', begin: 0, end: 1 },
			{ elem: prefix1TimeElem, label: 'Prefix1', begin: 2, end: 3 },
			{ elem: prefix2TimeElem, label: 'Prefix2', begin: 4, end: 5 },
			{ elem: prefix3TimeElem, label: 'Prefix3', begin: 6, end: 7 },
			{ elem: reaperTimeElem, label: 'Reaper', begin: 8, end: 9 },
			{ elem: totalTimeElem, label: 'ComputeTime', begin: 0, end: 9 },
		];
		for (let p = 0; p < passNames.length; ++p) {
			let max = -Infinity;
			let sum = 0;
			for (let i = 0; i < timestampSampleCount; ++i) {
				let base = i * PASSES * 2;
				const t0 = array[base + passNames[p].begin];
				const t1 = array[base + passNames[p].end];
				const time = Number(t1 - t0) * timestampPeriod * 1e-6; // convert nanoseconds to milliseconds
				sum += time;
				if (time > max) max = time;
			}
			let avg = sum / timestampSampleCount;
			let adjustedSum = sum * repeat / timestampSampleCount;
			passNames[p].elem.textContent =
				`${passNames[p].label}: avg=${avg.toFixed(2)}, tot=${adjustedSum.toFixed(2)}, max=${max.toFixed(2)} ms}`;
		}
		const tbegin = array[0];
		const tend = array[PASSES * 2 * timestampSampleCount + 1];
		const totalTime = Number(tend - tbegin) * timestampPeriod * 1e-6;
		totalTimeElem.textContent =
			`Total Time: ${totalTime.toFixed(2)} ms`;
		const t0 = array[PASSES * 2 * timestampSampleCount];
		const t1 = array[PASSES * 2 * timestampSampleCount + 1];
		const renderTime = Number(t1 - t0) * timestampPeriod * 1e-6; // convert nanoseconds to milliseconds
		renderTimeElem.textContent =
			`Render Time: ${renderTime.toFixed(2)} ms`;


		timestampStagingBuffer.unmap();
		// --- FPS & IPS update ---
		iterationCount += repeat;
		frameCount++;
		const now = performance.now();
		if (now - lastIPSUpdate > 500) {
			ips = Math.round(iterationCount * 1000 / (now - lastIPSUpdate));
			fps = Math.round(frameCount * 1000 / (now - lastIPSUpdate));
			if (fpsElem) fpsElem.textContent = `FPS: ${fps}`;
			if (ipsElem) ipsElem.textContent = `IPS: ${ips}`;
			if (repeatCountElem) repeatCountElem.textContent = `Repeat: ${repeat}`;
			lastIPSUpdate = now;
			iterationCount = 0;
			frameCount = 0;
		}

		requestAnimationFrame(frame);
	}

	requestAnimationFrame(frame);
}

main();
