import {
	makeShaderDataDefinitions,
	makeStructuredView,
} from 'https://greggman.github.io/webgpu-utils/dist/2.x/webgpu-utils.module.js';


async function fetchShaderCode(url) {
	const response = await fetch(url);
	if (!response.ok) throw new Error('Failed to load shader');
	return await response.text();
}

async function main() {
	const computeShaderCode = await fetchShaderCode('shader.comp');
	const renderShaderCode = await fetchShaderCode('shader.render');
	const adapter = await navigator.gpu?.requestAdapter();
	const device = await adapter?.requestDevice();
	if (!device) {
		fail('need a browser that supports WebGPU');
		return;
	}
	const canvas = document.querySelector('canvas');
	const context = canvas.getContext('webgpu');
	const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
	context.configure({
		device,
		format: presentationFormat,
	});



	// Initial bot data
	const bot_count = 100;
	let bot_data = [];
	for (let i = 0; i < bot_count; ++i) {
		let bot = {
			position: [2 * (Math.random() - 0.5), 2 * (Math.random() - 0.5), 0, 0],
			velocity: [0.02 * (Math.random() - 0.5), 0.02 * (Math.random() - 0.5), 0, 0],
			score: 0,
		}
		bot_data.push(bot);
	}
	const defs = makeShaderDataDefinitions(computeShaderCode);
	const botsView = makeStructuredView(defs.storages.in_bots);
	const botsViewSize = botsView.arrayBuffer.byteLength;
	console.log('botsViewSize', botsViewSize);

	botsView.set(bot_data);


	const renderModule = device.createShaderModule({
		label: 'our hardcoded red triangle shaders',
		code: renderShaderCode,
	});

	const renderPipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: renderModule,
			buffers: [
				{
					// instanced particles buffer
					arrayStride: botsViewSize / bot_count,
					stepMode: 'instance',
					attributes: [
						{
							// instance position
							shaderLocation: 0,
							offset: 0,
							format: 'float32x2',
						},
						{
							// instance velocity
							shaderLocation: 1,
							offset: 4 * 4,
							format: 'float32x2',
						},
					],
				},
				{
					// vertex buffer
					arrayStride: 2 * 4,
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
		label: 'our basic canvas renderPass',
		colorAttachments: [
			{
				// view: <- to be filled out when we render
				clearValue: [0.3, 0.3, 0.3, 1],
				loadOp: 'clear',
				storeOp: 'store',
			},
		],
	};

	const vertexBufferData = new Float32Array([
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

	const computeModule = device.createShaderModule({
		label: 'doubling compute module',
		code: computeShaderCode,
	});

	const computePipeline = device.createComputePipeline({
		label: 'doubling compute pipeline',
		layout: 'auto',
		compute: {
			module: computeModule,
		},
	});

	// ping-pong buffers for bot data
	const botsBuffers = new Array(2);
	for (let i = 0; i < 2; ++i) {
		botsBuffers[i] = device.createBuffer({
			label: 'work buffer',
			size: botsViewSize,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX,
		});

	}

	device.queue.writeBuffer(botsBuffers[0], 0, botsView.arrayBuffer);

	// Setup a bindGroup to tell the shader which
	// buffer to use for the computation
	const botsBindGroups = new Array(2);
	for (let i = 0; i < 2; ++i) {
		botsBindGroups[i] = device.createBindGroup({
			layout: computePipeline.getBindGroupLayout(0),
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
			],
		});
	}

	let step = 0;

	let paused = false;
	window.addEventListener('keydown', (e) => {
		if (e.code === 'Space') {
			paused = !paused;
			if (!paused) requestAnimationFrame(frame);
		}
	});

	function frame() {
		if (paused) return;
		let startTime = performance.now();
		renderPassDescriptor.colorAttachments[0].view = context
			.getCurrentTexture()
			.createView();
		// Encode commands to do the computation
		const encoder = device.createCommandEncoder({
			label: 'doubling encoder',
		});
		const pass = encoder.beginComputePass({
			label: 'doubling compute pass',
		});
		pass.setPipeline(computePipeline);
		pass.setBindGroup(0, botsBindGroups[step % 2]);
		pass.dispatchWorkgroups(100);
		pass.end();

		const passEncoder = encoder.beginRenderPass(renderPassDescriptor);
		passEncoder.setPipeline(renderPipeline);
		passEncoder.setVertexBuffer(0, botsBuffers[(step + 1) % 2]);
		passEncoder.setVertexBuffer(1, spriteVertexBuffer);
		passEncoder.draw(3, bot_count);
		passEncoder.end();

		const commandBuffer = encoder.finish();

		device.queue.submit([commandBuffer]);
		step++;

		const endTime = performance.now();

		const durationMilliseconds = endTime - startTime;
		const durationSeconds = durationMilliseconds / 1000;
		requestAnimationFrame(frame);
	}

	requestAnimationFrame(frame);

	await device.queue.onSubmittedWorkDone();
}

main();
