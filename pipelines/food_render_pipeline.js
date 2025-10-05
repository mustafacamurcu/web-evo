
export function createFoodRenderPipeline(device, code, context, msaaTexture, depthTexture) {
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
					arrayStride: 16,
					stepMode: 'instance',
					attributes: [
						{
							// food position
							shaderLocation: 0,
							offset: 0,
							format: 'float32x2',
						},
						{
							// food energy (as color)
							shaderLocation: 2,
							offset: 8,
							format: 'float32',
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
							shaderLocation: 1,
							offset: 0,
							format: 'float32x2',
						},
					],
				}
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
		multisample: {
			count: 4, // Use 4x MSAA
		},
		depthStencil: {
			format: "depth24plus", // Must match the depth texture format
			depthWriteEnabled: true, // Enable writing to the depth buffer
			depthCompare: "less" // Or other comparison function (e.g., "less-equal")
		},
	});

	const renderPassDescriptor = {
		colorAttachments: [
			{
				// resolveTarget: <- to be filled out when we render
				view: msaaTexture,
				clearValue: [0.3, 0.3, 0.3, 1],
				loadOp: 'load',
				storeOp: 'store',
			},
		],
		depthStencilAttachment: {
			view: depthTexture.createView(),
			depthClearValue: 1.0,
			depthLoadOp: 'load',
			depthStoreOp: 'store',
		}
	};

	const scale = 0.005;

	// create square for food
	const halfSize = scale / 2;
	const vertices = new Float32Array([
		-halfSize, -halfSize,
		halfSize, -halfSize,
		halfSize, halfSize,
		-halfSize, -halfSize,
		halfSize, halfSize,
		-halfSize, halfSize,
	]);
	const foodModelBuffer = device.createBuffer({
		size: vertices.byteLength,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
		mappedAtCreation: true, // You can also use this to write data at creation time
	});
	new Float32Array(foodModelBuffer.getMappedRange()).set(vertices);
	foodModelBuffer.unmap();

	return { pipeline, renderPassDescriptor, foodModelBuffer };
}