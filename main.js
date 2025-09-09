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

	const renderPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: renderModule,
			buffers: [
				{
					arrayStride: 16, // position + direction = 16 bytes
					stepMode: 'instance',
					attributes: [
						{
							// bot position
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
		-0.01, -0.02, 0.01,
		-0.02, 0.0, 0.02,
	]);

	const spriteVertexBuffer = device.createBuffer({
		size: vertexBufferData.byteLength,
		usage: GPUBufferUsage.VERTEX,
		mappedAtCreation: true,
	});
	new Float32Array(spriteVertexBuffer.getMappedRange()).set(vertexBufferData);
	spriteVertexBuffer.unmap();

	return { renderPipeline, renderPassDescriptor, spriteVertexBuffer };
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


function createBotsBuffers(device, defs) {
	let bot_data = [];
	for (let i = 0; i < bot_count; ++i) {
		let bot = {
			position: [2 * (Math.random() - 0.5), 2 * (Math.random() - 0.5)],
			velocity: [0.00002 * (Math.random() - 0.5), 0.00002 * (Math.random() - 0.5)],
			energy: 0,
		}
		bot_data.push(bot);
	}

	var buffers = new Array(2);
	for (let i = 0; i < 2; ++i) {
		buffers[i] = createStorageBuffer(device, defs.storages.in_bots, bot_count,
			GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
			bot_data);
	}

	return buffers;
}

function createFoodsBuffer(device, defs) {
	let food_data = [];
	for (let i = 0; i < food_count; ++i) {
		food_data.push(2 * (Math.random() - 0.5));
		food_data.push(2 * (Math.random() - 0.5));
	}
	return createStorageBuffer(device, defs.storages.foods, food_count,
		GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
		food_data);
}

function createSpikesBuffer(device, defs) {
	let spike_data = [];
	for (let i = 0; i < spike_count; ++i) {
		spike_data.push(2 * (Math.random() - 0.5));
		spike_data.push(2 * (Math.random() - 0.5));
	}
	return createStorageBuffer(device, defs.storages.spikes, spike_count,
		GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
		spike_data);
}

function createComputePipeline(device, code) {
	const botStepperModule = device.createShaderModule({
		label: 'bot stepper compute module',
		code,
	});

	const botStepperPipeline = device.createComputePipeline({
		label: 'bot stepper compute pipeline',
		layout: 'auto',
		compute: {
			module: botStepperModule,
		},
	});


	const defs = makeShaderDataDefinitions(code);

	// Create Buffers
	const botsBuffers = createBotsBuffers(device, defs);
	const foodsBuffer = createFoodsBuffer(device, defs);
	const spikesBuffer = createSpikesBuffer(device, defs);

	// Bind Groups
	const botsBindGroups = new Array(2);
	for (let i = 0; i < 2; ++i) {
		botsBindGroups[i] = device.createBindGroup({
			layout: botStepperPipeline.getBindGroupLayout(0),
			entries: [
				{
					binding: 0,
					resource: {
						buffer: botsBuffers[i],
					},
				},
				{
					binding: 1,
					resource: {
						buffer: botsBuffers[(i + 1) % 2],
					},
				},
				{
					binding: 2,
					resource: {
						buffer: foodsBuffer,
					},
				},
				{
					binding: 3,
					resource: {
						buffer: spikesBuffer,
					},
				},
			],
		});
	}

	return { botStepperPipeline, botsBuffers, foodsBuffer, spikesBuffer, botsBindGroups };
}

function createVerticesPipeline(device, code, botsBuffer, foodsBuffer, spikesBuffer) {
	const verticesModule = device.createShaderModule({
		label: 'make vertices compute module',
		code,
	});
	const verticesPipeline = device.createComputePipeline({
		label: 'make vertices compute pipeline',
		layout: 'auto',
		compute: {
			module: verticesModule,
		},
	});


	const defs = makeShaderDataDefinitions(code);

	const verticesBuffer = createStorageBuffer(device, defs.storages.orientations, bot_count,
		GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		null);

	// Bind Groups
	const verticesBindGroup = device.createBindGroup({
		layout: verticesPipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: botsBuffer,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: verticesBuffer,
				},
			}
		],
	});

	return { verticesPipeline, verticesBuffer, verticesBindGroup };
}


// Constants
const bot_count = 64 * 10;
const food_count = 100;
const spike_count = 100;

async function main() {
	// WebGPU setup
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();
	const canvas = document.querySelector('canvas');
	const context = canvas.getContext('webgpu');

	// Bot Stepper Compute Pipeline
	const computeShaderCode = await fetchShaderCode('bot_step.comp');
	const { botStepperPipeline, botsBuffers, foodsBuffer, spikesBuffer, botsBindGroups } = createComputePipeline(device, computeShaderCode);

	// Vertex Creation Compute Pipeline
	const verticesShaderCode = await fetchShaderCode('make_vertices.comp');
	const { verticesPipeline, verticesBuffer, verticesBindGroup } = createVerticesPipeline(device, verticesShaderCode, botsBuffers[0], foodsBuffer, spikesBuffer);

	// Render Pipeline
	const renderShaderCode = await fetchShaderCode('shader.render');
	const { renderPipeline, renderPassDescriptor, spriteVertexBuffer } = createRenderPipeline(device, renderShaderCode, context);

	let paused = false;
	window.addEventListener('keydown', (e) => {
		if (e.code === 'Space') {
			paused = !paused;
			if (!paused) requestAnimationFrame(frame);
		}
	});

	const fpsElem = document.getElementById('fps');
	const msElem = document.getElementById('ms');

	let lastStatsUpdate = 0;
	// Has to be at least 2 to ping-pong
	const repeat = 1000;
	async function frame() {
		if (paused) return;
		let startTime = performance.now();

		// Step the computation many times
		const encoder = device.createCommandEncoder({});
		for (let i = 0; i < repeat; ++i) {
			const pass = encoder.beginComputePass({});
			pass.setPipeline(botStepperPipeline);
			pass.setBindGroup(0, botsBindGroups[i % 2]);
			pass.dispatchWorkgroups(bot_count / 64);
			pass.end();
		}

		let computeCommandBuffer = encoder.finish();
		device.queue.submit([computeCommandBuffer]);
		await device.queue.onSubmittedWorkDone();
		let endTime = performance.now();

		const ms_per_step = (endTime - startTime) / repeat;
		const sec_per_step = ms_per_step / 1000.0;
		const ips = 1 / sec_per_step;

		// Only update stats once per second
		if (endTime - lastStatsUpdate > 1000) {
			fpsElem.textContent = `IPS: ${ips.toFixed(1)}`;
			msElem.textContent = `ms: ${ms_per_step.toFixed(1)}`;
			lastStatsUpdate = endTime;
		}

		// make vertices
		const renderEncoder = device.createCommandEncoder({});
		const verticesPass = renderEncoder.beginComputePass({});
		verticesPass.setPipeline(verticesPipeline);
		verticesPass.setBindGroup(0, verticesBindGroup);
		verticesPass.dispatchWorkgroups(bot_count);
		verticesPass.end();

		// render
		renderPassDescriptor.colorAttachments[0].view = context
			.getCurrentTexture()
			.createView();
		const renderPass = renderEncoder.beginRenderPass(renderPassDescriptor);
		renderPass.setPipeline(renderPipeline);
		renderPass.setVertexBuffer(0, verticesBuffer);
		renderPass.setVertexBuffer(1, spriteVertexBuffer);
		renderPass.draw(3, bot_count);
		renderPass.end();

		const renderCommandBuffer = renderEncoder.finish();
		device.queue.submit([renderCommandBuffer]);
		requestAnimationFrame(frame);
	}

	requestAnimationFrame(frame);
}

main();
