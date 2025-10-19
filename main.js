import {
	makeShaderDataDefinitions,
	makeStructuredView,
	getSizeAndAlignmentOfUnsizedArrayElement
} from 'https://greggman.github.io/webgpu-utils/dist/2.x/webgpu-utils.module.js';
import { createRenderPipeline } from './pipelines/render_pipeline.js';
import { createFoodRenderPipeline } from './pipelines/food_render_pipeline.js';
import { dumpBotsBuffer, dumpBufferFloat, dumpBufferInt, dumpBufferSense, fetchShaderCode } from './utils.js';
import { setupSimulation } from './simulation_setup.js';

async function main() {
	// WebGPU setup
	const adapter = await navigator.gpu?.requestAdapter();
	const defaultLimits = adapter.limits;
	const device = await adapter.requestDevice({
		requiredLimits: {
			maxStorageBufferBindingSize: defaultLimits.maxStorageBufferBindingSize,
			maxStorageBuffersPerShaderStage: defaultLimits.maxStorageBuffersPerShaderStage,
		},
		requiredFeatures: ['timestamp-query'],
	});

	const canvas = document.querySelector('canvas');
	const context = canvas.getContext('webgpu');

	// Compute Pipelines and Buffers setup
	const { buffers, pipelines, constants } = await setupSimulation(device);
	const { verticesBuffer, foodsBuffer, numBotsBuffer } = buffers;
	const { botStepperPipeline,
		foodStepperPipeline,
		botDecidePipeline,
		prefixSumPipeline,
		reaperPipeline,
		botSensesPipeline,
		botVerticesPipeline,
		numBotsPipeline,
		repopulatePipeline
	} = pipelines;


	const botStepShaderCode = await fetchShaderCode('bot_step.comp');
	const defs = makeShaderDataDefinitions(botStepShaderCode);
	const botsStructStorageDef = defs.storages['bots'];


	// Render Pipelines setup
	const depthTexture = device.createTexture({
		size: [canvas.width, canvas.height, 1], // Or presentationSize for the swap chain
		format: "depth24plus", // Or other suitable depth format
		usage: GPUTextureUsage.RENDER_ATTACHMENT,
		sampleCount: 4, // Match the sample count of the MSAA texture
	});

	const msaaTexture = device.createTexture({
		size: [canvas.width, canvas.height],
		sampleCount: 4, // Enable multisampling with 4 samples
		format: navigator.gpu.getPreferredCanvasFormat(),
		usage: GPUTextureUsage.RENDER_ATTACHMENT,
	});

	const renderShaderCode = await fetchShaderCode('bot.render');
	const foodRenderShaderCode = await fetchShaderCode('food.render');
	const renderPipeline = createRenderPipeline(device, renderShaderCode, context, msaaTexture, 48, depthTexture);
	const foodRenderPipeline = createFoodRenderPipeline(device, foodRenderShaderCode, context, msaaTexture, depthTexture);

	// Constants
	const { MAX_BOTS, MAX_FOOD } = constants;
	const PASSES = 7; // sense, decide, step, prefix1, prefix2, prefix3, reaper
	const QUERIES_PER_PASS = 2;
	const REPEAT = 100;
	const TOTAL_QUERIES = PASSES * QUERIES_PER_PASS * REPEAT + 2 + 2; // +2 for vertices, +2 for render

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

		// COMPUTE PIPELINE
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

			// Step food in foodsBuffer (with timestamp)
			const foodStepPass = encoder.beginComputePass();
			foodStepPass.setPipeline(foodStepperPipeline.pipeline);
			foodStepPass.setBindGroup(0, foodStepperPipeline.bindGroup);
			foodStepPass.dispatchWorkgroups(MAX_FOOD / 64 + 1);
			foodStepPass.end();

			// Prefix1
			const prefixSumPass1 = encoder.beginComputePass(tswrites(base + 6, base + 7));
			prefixSumPass1.setPipeline(prefixSumPipeline.pipeline1);
			prefixSumPass1.setBindGroup(0, prefixSumPipeline.bindGroup1);
			prefixSumPass1.dispatchWorkgroups(MAX_BOTS / 64);
			prefixSumPass1.end();

			// Prefix2
			const prefixSumPass2 = encoder.beginComputePass(tswrites(base + 8, base + 9));
			prefixSumPass2.setPipeline(prefixSumPipeline.pipeline2);
			prefixSumPass2.setBindGroup(0, prefixSumPipeline.bindGroup2);
			prefixSumPass2.dispatchWorkgroups(MAX_BOTS / 64 / 64);
			prefixSumPass2.end();

			// Prefix3
			const prefixSumPass3 = encoder.beginComputePass(tswrites(base + 10, base + 11));
			prefixSumPass3.setPipeline(prefixSumPipeline.pipeline3);
			prefixSumPass3.setBindGroup(0, prefixSumPipeline.bindGroup3);
			prefixSumPass3.dispatchWorkgroups(1);
			prefixSumPass3.end();

			// Reaper
			const reaperPass = encoder.beginComputePass(tswrites(base + 12, base + 13));
			reaperPass.setPipeline(reaperPipeline.pipeline);
			reaperPass.setBindGroup(0, reaperPipeline.bindGroup);
			reaperPass.dispatchWorkgroupsIndirect(numBotsBuffer, 0);
			reaperPass.end();

			// Update num bots
			const numBotsPass = encoder.beginComputePass();
			numBotsPass.setPipeline(numBotsPipeline.pipeline);
			numBotsPass.setBindGroup(0, numBotsPipeline.bindGroup);
			numBotsPass.dispatchWorkgroups(1);
			numBotsPass.end();

			// Repopulate bots if needed
			const repopulatePass = encoder.beginComputePass();
			repopulatePass.setPipeline(repopulatePipeline.pipeline);
			repopulatePass.setBindGroup(0, repopulatePipeline.bindGroup);
			repopulatePass.dispatchWorkgroups(1);
			repopulatePass.end();
		}

		// Update bot vertices for rendering
		const botVerticesPass = encoder.beginComputePass(tswrites(REPEAT * PASSES * 2, REPEAT * PASSES * 2 + 1));
		botVerticesPass.setPipeline(botVerticesPipeline.pipeline);
		botVerticesPass.setBindGroup(0, botVerticesPipeline.bindGroup);
		botVerticesPass.dispatchWorkgroups(MAX_BOTS / 64);
		botVerticesPass.end();

		// render
		renderPipeline.renderPassDescriptor.colorAttachments[0].resolveTarget = context
			.getCurrentTexture()
			.createView();
		renderPipeline.renderPassDescriptor.timestampWrites = tswrites(REPEAT * PASSES * 2 + 2, REPEAT * PASSES * 2 + 3).timestampWrites;

		const renderPass = encoder.beginRenderPass(renderPipeline.renderPassDescriptor);
		renderPass.setPipeline(renderPipeline.pipeline);
		renderPass.setVertexBuffer(0, verticesBuffer);
		renderPass.setVertexBuffer(1, renderPipeline.botModelBuffer);
		renderPass.drawIndirect(numBotsBuffer, 4 * 4);
		renderPass.end();

		// render food
		const foodRenderPassDescriptor = foodRenderPipeline.renderPassDescriptor;
		foodRenderPassDescriptor.colorAttachments[0].resolveTarget = context
			.getCurrentTexture()
			.createView();

		const foodRenderPass = encoder.beginRenderPass(foodRenderPassDescriptor);
		foodRenderPass.setPipeline(foodRenderPipeline.pipeline);
		foodRenderPass.setVertexBuffer(0, foodsBuffer);
		foodRenderPass.setVertexBuffer(1, foodRenderPipeline.foodModelBuffer);
		foodRenderPass.draw(3 * 2, MAX_FOOD);
		foodRenderPass.end();

		// Resolve timestamp queries if available
		if (timestamp_enabled) {
			encoder.resolveQuerySet(timestampQuerySet, 0, TOTAL_QUERIES, timestampResolveBuffer, 0);
			encoder.copyBufferToBuffer(timestampResolveBuffer, 0, timestampStagingBuffer, 0, TOTAL_QUERIES * 8);
		}

		const commandBuffer = encoder.finish();
		device.queue.submit([commandBuffer]);

		await device.queue.onSubmittedWorkDone();

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
		var numBots = await dumpBufferInt(device, numBotsBuffer);
		iterationCount += REPEAT;
		frameCount++;
		const now = performance.now();
		if (now - lastIPSUpdate > 500) {
			ips = Math.round(iterationCount * 1000 / (now - lastIPSUpdate));
			fps = Math.round(frameCount * 1000 / (now - lastIPSUpdate));
			if (fpsElem) fpsElem.textContent = `FPS: ${fps}`;
			if (ipsElem) ipsElem.textContent = `IPS: ${ips}`;
			if (repeatCountElem) repeatCountElem.textContent = `Repeat: ${REPEAT}`;
			if (repeatCountElem) botCountElem.textContent = `Bot Count: ${numBots[0]}`;
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
		}
		if (e.code === 'KeyT') {
			timestamp_enabled = !timestamp_enabled;
		}
		const statsElem = document.getElementById('stats');
		if (e.code === 'KeyH') {
			if (statsElem) {
				statsElem.style.display = (statsElem.style.display === 'none') ? '' : 'none';
			}
		}
		if (e.code === 'KeyB') {
			// Dump some bot data
			let numBots = await dumpBufferInt(device, numBotsBuffer);
			const bots = await dumpBotsBuffer(device, buffers.botsBuffer);
			console.log('--- Bot Data ---');
			for (let i = 0; i < numBots[0]; ++i) {
				console.log(`Bot ${i}: Pos(${bots[i].position[0].toFixed(3)}, ${bots[i].position[1].toFixed(3)}) Energy: ${bots[i].energy.toFixed(2)} Age: ${bots[i].age} DSB: ${bots[i].die_stay_breed} ID: ${bots[i].id} decision: ${bots[i].decision}`);
			}
		}
	});

	requestAnimationFrame(frame);
}

main();
