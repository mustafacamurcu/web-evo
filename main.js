import {
	makeShaderDataDefinitions,
	makeStructuredView,
	getSizeAndAlignmentOfUnsizedArrayElement
} from 'https://greggman.github.io/webgpu-utils/dist/2.x/webgpu-utils.module.js';
import { createRenderPipeline } from './pipelines/render_pipeline.js';
import { createBotStepperPipeline } from './pipelines/bot_step_pipeline.js';
import { createBotDecidePipeline } from './pipelines/bot_decide_pipeline.js';
import { createPrefixSumPipeline } from './pipelines/prefix_sum_pipeline.js';
import { createReaperPipeline } from './pipelines/reaper_pipeline.js';
import { createBotSensePipeline } from './pipelines/bot_sense_pipeline.js';
import { createBotVerticesPipeline } from './pipelines/bot_vertices_pipeline.js';
import { dumpBuffer, dumpBufferInt, dumpBufferSense } from './utils.js';

// also imports structs.wgsl by default
async function fetchShaderCode(url) {
	const structs = await fetch('shaders/structs.wgsl');
	const response = await fetch('shaders/' + url);
	if (!response.ok) throw new Error('Failed to load shader');
	const code = await response.text();
	const structsCode = await structs.text();
	return structsCode + '\n' + code;
}

function createStorageBuffer(device, storageDefinitionName, code, objectCount, usage, data = null) {
	const defs = makeShaderDataDefinitions(code);
	const storageDefinition = defs.storages[storageDefinitionName];
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

// Constants
const MAX_BOTS = 64 * 64 * 1; // increase by adding a level to prefix sum
const INITIAL_BOT_COUNT = 50;
const PASSES = 7; // sense, decide, step, prefix1, prefix2, prefix3, reaper
const QUERIES_PER_PASS = 2;
const REPEAT = 20;
const TOTAL_QUERIES = PASSES * QUERIES_PER_PASS * REPEAT + 2 + 2; // +2 for vertices, +2 for render

async function main() {
	// WebGPU setup
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice({
		requiredFeatures: ['timestamp-query'],
	});
	const canvas = document.querySelector('canvas');
	const context = canvas.getContext('webgpu');

	const depthTexture = device.createTexture({
		size: [canvas.width, canvas.height, 1], // Or presentationSize for the swap chain
		format: "depth24plus", // Or other suitable depth format
		usage: GPUTextureUsage.RENDER_ATTACHMENT
	});

	// Initial data setup
	let bot_data = [];
	for (let i = 1; i < MAX_BOTS; ++i) {
		if (i <= INITIAL_BOT_COUNT) {
			let bot = {
				color: [Math.random(), Math.random(), Math.random(), 1.0],
				position: [2 * (Math.random() - 0.5), 2 * (Math.random() - 0.5)],
				velocity: [0.00002 * (Math.random() - 0.5), 0.00002 * (Math.random() - 0.5)],
				die_stay_breed: 1, // alive
				energy: 0,
				id: i,
				age: 0
			};
			bot_data.push(bot);
		}
		else {
			let bot = {
				color: [Math.random(), Math.random(), Math.random(), 0.0],
				position: [10, 10], // offscreen
				velocity: [0, 0],
				die_stay_breed: 0, // dead
				energy: 0,
				id: i,
				age: 0,
			}
			bot_data.push(bot);
		}
	}

	// brain data
	let brain_data = [];
	for (let i = 0; i < MAX_BOTS; ++i) {
		let brain = {
			w1: new Float32Array(16 * 32).map(() => (Math.random() * 2 - 1) * 0.1),
			b1: new Float32Array(32).map(() => (Math.random() * 2 - 1) * 0.1),
			w2: new Float32Array(32 * 16).map(() => (Math.random() * 2 - 1) * 0.1),
			b2: new Float32Array(16).map(() => (Math.random() * 2 - 1) * 0.1),
		}
		brain_data.push(brain);
	}

	// Load shader codes
	const botSenseShaderCode = await fetchShaderCode('bot_sense.comp');
	const botDecideShaderCode = await fetchShaderCode('bot_decide.comp');
	const botStepShaderCode = await fetchShaderCode('bot_step.comp');
	const prefixSumShaderCode = await fetchShaderCode('prefix_sum.comp');
	const reaperShaderCode = await fetchShaderCode('reaper.comp');
	const botVerticesShaderCode = await fetchShaderCode('bot_vertices.comp');
	const renderShaderCode = await fetchShaderCode('bot.render');

	// Create buffers
	const botSensesBuffer = createStorageBuffer(device, 'bot_senses', botSenseShaderCode, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const botBrainsBuffer = createStorageBuffer(device, 'bot_brains', botDecideShaderCode, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, brain_data);
	const botsBuffer = createStorageBuffer(device, 'bots', botStepShaderCode, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, bot_data);
	const scratchBuffer = createStorageBuffer(device, 'scratchBuffer', botStepShaderCode, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const L1Buffer = createStorageBuffer(device, 'l1Buffer', prefixSumShaderCode, MAX_BOTS,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const L2Buffer = createStorageBuffer(device, 'l2Buffer', prefixSumShaderCode, MAX_BOTS / 64 + 1,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const L3Buffer = createStorageBuffer(device, 'l3Buffer', prefixSumShaderCode, MAX_BOTS / 64 / 64 + 1,
		GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const verticesBuffer = createStorageBuffer(device, "vertex_datas", botVerticesShaderCode, MAX_BOTS,
		GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

	const numBotsBuffer = device.createBuffer({
		size: 3 * 4, // 3 u32 values * 4 bytes/u32 = 12 bytes
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	device.queue.writeBuffer(numBotsBuffer, 0, new Uint32Array([INITIAL_BOT_COUNT, 1, 1]));

	const verticesBufferStride = getSizeAndAlignmentOfUnsizedArrayElement(
		makeShaderDataDefinitions(botVerticesShaderCode).storages["vertex_datas"]
	).size;


	// Create pipelines
	const botSensesPipeline = createBotSensePipeline(device, botSenseShaderCode, botsBuffer, botSensesBuffer);
	const botDecidePipeline = createBotDecidePipeline(device, botDecideShaderCode, botsBuffer, botBrainsBuffer, botSensesBuffer);
	const botStepperPipeline = createBotStepperPipeline(device, botStepShaderCode, botsBuffer, scratchBuffer);
	const prefixSumPipeline = createPrefixSumPipeline(device, prefixSumShaderCode, L1Buffer, L2Buffer, L3Buffer, scratchBuffer);
	const reaperPipeline = createReaperPipeline(device, reaperShaderCode, scratchBuffer, botsBuffer, L1Buffer, L2Buffer, L3Buffer, numBotsBuffer);
	const botVerticesPipeline = createBotVerticesPipeline(device, botVerticesShaderCode, botsBuffer, botSensesBuffer, verticesBuffer);
	const renderPipeline = createRenderPipeline(device, renderShaderCode, context, verticesBufferStride, depthTexture);


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
	const prefix1TimeElem = document.getElementById('prefix1Time');
	const prefix2TimeElem = document.getElementById('prefix2Time');
	const prefix3TimeElem = document.getElementById('prefix3Time');
	const reaperTimeElem = document.getElementById('reaperTime');
	const senseTimeElem = document.getElementById('senseTime');
	const decideTimeElem = document.getElementById('decideTime');
	const stepTimeElem = document.getElementById('stepTime');
	const repeatCountElem = document.getElementById('repeatCount');
	const botCountElem = document.getElementById('botCount');
	const computeTimeElem = document.getElementById('computeTime');
	const vertexTimeElem = document.getElementById('vertexTime');
	const renderTimeElem = document.getElementById('renderTime');
	const totalTimeElem = document.getElementById('totalTime');

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

	let timestamp_enabled = true;
	function tswrites(start, end) {
		if (!timestamp_enabled) return {};
		return {
			timestampWrites: {
				querySet: timestampQuerySet,
				beginningOfPassWriteIndex: start,
				endOfPassWriteIndex: end,
			}
		};
	}

	async function frame() {
		if (paused) return;

		// Step the computation many times
		const encoder = device.createCommandEncoder({});
		for (let i = 0; i < REPEAT; ++i) {
			let base = i * PASSES * 2;
			// Calculate bot senses
			const sensePass = encoder.beginComputePass(tswrites(base + 0, base + 1));
			sensePass.setPipeline(botSensesPipeline.pipeline);
			sensePass.setBindGroup(0, botSensesPipeline.bindGroup);
			sensePass.dispatchWorkgroupsIndirect(numBotsBuffer, 0);
			sensePass.end();

			// Decide bot actions
			const decidePass = encoder.beginComputePass(tswrites(base + 2, base + 3));
			decidePass.setPipeline(botDecidePipeline.pipeline);
			decidePass.setBindGroup(0, botDecidePipeline.bindGroup);
			decidePass.dispatchWorkgroupsIndirect(numBotsBuffer, 0);
			decidePass.end();

			// Step bots in botsBuffer (with timestamp)
			const stepPass = encoder.beginComputePass(tswrites(base + 4, base + 5));
			stepPass.setPipeline(botStepperPipeline.pipeline);
			stepPass.setBindGroup(0, botStepperPipeline.bindGroup);
			stepPass.dispatchWorkgroups(MAX_BOTS / 64);
			stepPass.end();

			// Prefix1
			const prefixSumPass1 = encoder.beginComputePass(tswrites(base + 6, base + 7));
			prefixSumPass1.setPipeline(prefixSumPipeline.pipeline1);
			prefixSumPass1.setBindGroup(0, prefixSumPipeline.L1BindGroup);
			prefixSumPass1.dispatchWorkgroups(MAX_BOTS / 64);
			prefixSumPass1.end();

			// Prefix2
			const prefixSumPass2 = encoder.beginComputePass(tswrites(base + 8, base + 9));
			prefixSumPass2.setPipeline(prefixSumPipeline.pipeline2);
			prefixSumPass2.setBindGroup(0, prefixSumPipeline.L2BindGroup);
			prefixSumPass2.dispatchWorkgroups(MAX_BOTS / 64 / 64);
			prefixSumPass2.end();

			// Prefix3
			const prefixSumPass3 = encoder.beginComputePass(tswrites(base + 10, base + 11));
			prefixSumPass3.setPipeline(prefixSumPipeline.pipeline3);
			prefixSumPass3.setBindGroup(0, prefixSumPipeline.L3BindGroup);
			prefixSumPass3.dispatchWorkgroups(1);
			prefixSumPass3.end();

			// Reaper
			const reaperPass = encoder.beginComputePass(tswrites(base + 12, base + 13));
			reaperPass.setPipeline(reaperPipeline.pipeline);
			reaperPass.setBindGroup(0, reaperPipeline.bindGroup);
			reaperPass.dispatchWorkgroups(MAX_BOTS / 64);
			reaperPass.end();
		}

		// Update bot vertices for rendering
		const botVerticesPass = encoder.beginComputePass(tswrites(REPEAT * PASSES * 2, REPEAT * PASSES * 2 + 1));
		botVerticesPass.setPipeline(botVerticesPipeline.pipeline);
		botVerticesPass.setBindGroup(0, botVerticesPipeline.bindGroup);
		botVerticesPass.dispatchWorkgroups(MAX_BOTS / 64);
		botVerticesPass.end();

		// render
		renderPipeline.renderPassDescriptor.colorAttachments[0].view = context
			.getCurrentTexture()
			.createView();
		renderPipeline.renderPassDescriptor.timestampWrites = tswrites(REPEAT * PASSES * 2 + 2, REPEAT * PASSES * 2 + 3).timestampWrites;

		const renderPass = encoder.beginRenderPass(renderPipeline.renderPassDescriptor);
		renderPass.setPipeline(renderPipeline.pipeline);
		renderPass.setVertexBuffer(0, verticesBuffer);
		renderPass.setVertexBuffer(1, renderPipeline.botModelBuffer);
		renderPass.draw(3 * 9, MAX_BOTS);
		renderPass.end();

		// Resolve timestamp queries if available
		if (timestamp_enabled) {
			encoder.resolveQuerySet(timestampQuerySet, 0, TOTAL_QUERIES, timestampResolveBuffer, 0);
			encoder.copyBufferToBuffer(timestampResolveBuffer, 0, timestampStagingBuffer, 0, TOTAL_QUERIES * 8);
		}

		const commandBuffer = encoder.finish();
		device.queue.submit([commandBuffer]);

		if (timestamp_enabled) {
			// Read and display average pass times if available
			await timestampStagingBuffer.mapAsync(GPUMapMode.READ);
			const array = new BigUint64Array(timestampStagingBuffer.getMappedRange());
			// For each pass, average over all iterations
			const passNames = [
				{ elem: senseTimeElem, label: 'Sense', begin: 0, end: 1 },
				{ elem: decideTimeElem, label: 'Decide', begin: 2, end: 3 },
				{ elem: stepTimeElem, label: 'Step', begin: 4, end: 5 },
				{ elem: prefix1TimeElem, label: 'Prefix1', begin: 6, end: 7 },
				{ elem: prefix2TimeElem, label: 'Prefix2', begin: 8, end: 9 },
				{ elem: prefix3TimeElem, label: 'Prefix3', begin: 10, end: 11 },
				{ elem: reaperTimeElem, label: 'Reaper', begin: 12, end: 13 },
				{ elem: totalTimeElem, label: 'ComputeTime', begin: 0, end: 13 },
			];
			for (let p = 0; p < passNames.length; ++p) {
				let max = -Infinity;
				let sum = 0;
				for (let i = 0; i < REPEAT; ++i) {
					let base = i * PASSES * 2;
					const t0 = array[base + passNames[p].begin];
					const t1 = array[base + passNames[p].end];
					const time = Number(t1 - t0) * timestampPeriod * 1e-6; // convert nanoseconds to milliseconds
					sum += time;
					if (time > max) max = time;
				}
				let avg = sum / REPEAT;
				passNames[p].elem.textContent =
					`${passNames[p].label}: avg=${avg.toFixed(2)}, tot=${sum.toFixed(2)}, max=${max.toFixed(2)} ms}`;
			}
			const computeBegin = array[0];
			const computeEnd = array[PASSES * 2 * REPEAT + 3];
			const computeTime = Number(computeEnd - computeBegin) * timestampPeriod * 1e-6;
			computeTimeElem.textContent =
				`Compute Time: ${computeTime.toFixed(2)} ms`;
			const tbegin = array[0];
			const tend = array[PASSES * 2 * REPEAT + 3];
			const totalTime = Number(tend - tbegin) * timestampPeriod * 1e-6;
			totalTimeElem.textContent =
				`Total Time: ${totalTime.toFixed(2)} ms`;
			const vertexBegin = array[PASSES * 2 * REPEAT];
			const vertexEnd = array[PASSES * 2 * REPEAT + 1];
			const vertexTime = Number(vertexEnd - vertexBegin) * timestampPeriod * 1e-6; // convert nanoseconds to milliseconds
			vertexTimeElem.textContent =
				`Vertex Time: ${vertexTime.toFixed(2)} ms`;
			const t0 = array[PASSES * 2 * REPEAT + 2];
			const t1 = array[PASSES * 2 * REPEAT + 3];
			const renderTime = Number(t1 - t0) * timestampPeriod * 1e-6; // convert nanoseconds to milliseconds
			renderTimeElem.textContent =
				`Render Time: ${renderTime.toFixed(2)} ms`;


			timestampStagingBuffer.unmap();
		}
		// --- FPS & IPS update ---
		iterationCount += REPEAT;
		frameCount++;
		const now = performance.now();
		if (now - lastIPSUpdate > 500) {
			ips = Math.round(iterationCount * 1000 / (now - lastIPSUpdate));
			fps = Math.round(frameCount * 1000 / (now - lastIPSUpdate));
			if (fpsElem) fpsElem.textContent = `FPS: ${fps}`;
			if (ipsElem) ipsElem.textContent = `IPS: ${ips}`;
			if (repeatCountElem) repeatCountElem.textContent = `Repeat: ${REPEAT}`;
			if (repeatCountElem) botCountElem.textContent = `Bot Count: ${INITIAL_BOT_COUNT}`;
			lastIPSUpdate = now;
			iterationCount = 0;
			frameCount = 0;
		}

		requestAnimationFrame(frame);
	}

	window.addEventListener('keydown', async (e) => {
		if (e.code === 'Space') {
			paused = !paused;
			if (!paused) requestAnimationFrame(frame);
			else {
				var bots = await dumpBuffer(device, botsBuffer);
				var senses = await dumpBufferSense(device, botSensesBuffer);
				var numBots = await dumpBufferInt(device, numBotsBuffer);
				console.log('Num bots: ' + numBots[0]);
				var l3 = await dumpBufferInt(device, L3Buffer);
				console.log('L3: ' + l3);
				console.log(bots.slice(0, Math.min(20, INITIAL_BOT_COUNT)));
				console.log(senses.slice(0, Math.min(20, INITIAL_BOT_COUNT)));
				for (var j = 0; j < Math.min(20, INITIAL_BOT_COUNT); j++) {
					var bot = senses[j];
					for (let i = 0; i < 8; i++) {
						if (bot[i * 2] != 0) {
							console.log(`me: ${j + 1}, sense[${i}]: dist=${bot[i * 2 + 1]}, target=${bot[i * 2]}`);
						}
					}
				}
			}
		}
		if (e.code === 'KeyT') {
			timestamp_enabled = !timestamp_enabled;
		}
		const statsElem = document.getElementById('stats');
		window.addEventListener('keydown', (e) => {
			if (e.code === 'KeyH') {
				if (statsElem) {
					statsElem.style.display = (statsElem.style.display === 'none') ? '' : 'none';
				}
			}
		});
	});

	requestAnimationFrame(frame);
}

main();
