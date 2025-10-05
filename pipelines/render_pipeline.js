
export function createRenderPipeline(device, code, context, msaaTexture, verticesBufferStride, depthTexture) {
	const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
	context.configure({
		device,
		format: presentationFormat,
	});
	const renderModule = device.createShaderModule({ code });
	// @location(0) color : vec4f,
	// @location(1) position : vec2f,
	// @location(2) direction : vec2f,
	// @location(3) senses : u32,
	// @location(4) vertex_pos : vec2f,
	// @location(5) sense_id : u32,
	const pipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
			module: renderModule,
			buffers: [
				{
					arrayStride: verticesBufferStride,
					stepMode: 'instance',
					attributes: [
						{
							// bot color
							shaderLocation: 0,
							offset: 0,
							format: 'float32x4',
						},
						{
							// bots position
							shaderLocation: 1,
							offset: 16,
							format: 'float32x2',
						},
						{
							// bot direction
							shaderLocation: 2,
							offset: 24,
							format: 'float32x2',
						},
						{
							// bot senses
							shaderLocation: 4,
							offset: 32,
							format: 'uint32',
						},
					],
				},
				{
					// vertex buffer
					arrayStride: 12,
					stepMode: 'vertex',
					attributes: [
						{
							// vertex positions
							shaderLocation: 3,
							offset: 0,
							format: 'float32x2',
						},
						{
							// vertex sense_id
							shaderLocation: 5,
							offset: 8,
							format: 'uint32',
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
				view: msaaTexture.createView(),
				clearValue: [0.3, 0.3, 0.3, 1],
				loadOp: 'clear',
				storeOp: 'store',
			},
		],
		depthStencilAttachment: {
			view: depthTexture.createView(),
			depthClearValue: 1.0,
			depthLoadOp: 'clear',
			depthStoreOp: 'store',
		}
	};

	const scale = 0.05;
	const rayLen = 0.1;
	const numRays = 8;
	const numTriangles = numRays + 1; // 1 for the body
	const numVertices = 3 * numTriangles;
	const vertexBufferSize = numVertices * (8 + 4); // 8 bytes for float32x2 + 4 bytes for uint32
	const buffer = new ArrayBuffer(vertexBufferSize);
	const view = new DataView(buffer);

	var ray_points = [
		0.7071, 0.7071,
		0.5556, 0.8315,
		0.3827, 0.9239,
		0.1951, 0.9808,
		0.0000, 1.0000,
		-0.1951, 0.9808,
		-0.3827, 0.9239,
		-0.5556, 0.8315,
		-0.7071, 0.7071,
	];

	for (let i = 0; i < ray_points.length; i++) {
		ray_points[i] *= rayLen;
	}

	let offset = 0;
	for (let i = 0; i < numRays; i++) {
		// position
		view.setFloat32(offset, ray_points[i * 2], true); // x-coordinate
		view.setFloat32(offset + 4, ray_points[i * 2 + 1], true); // y-coordinate
		offset += 8;
		// Write the uint32 sense_id
		view.setUint32(offset, i, true);
		offset += 4;
		// position
		view.setFloat32(offset, ray_points[i * 2 + 2], true); // x-coordinate
		view.setFloat32(offset + 4, ray_points[i * 2 + 3], true); // y-coordinate
		offset += 8;
		// Write the uint32 sense_id
		view.setUint32(offset, i, true);
		offset += 4;
		// position
		view.setFloat32(offset, 0.0, true); // x-coordinate
		view.setFloat32(offset + 4, 0.0, true); // y-coordinate
		offset += 8;
		// Write the uint32 sense_id
		view.setUint32(offset, i, true);
		offset += 4;
	}
	view.setFloat32(offset, -0.2 * scale, true); // x-coordinate
	view.setFloat32(offset + 4, -0.4 * scale, true); // y-coordinate
	offset += 8;
	// Write the uint32 sense_id
	view.setUint32(offset, 8, true);
	offset += 4;
	// position
	view.setFloat32(offset, 0.2 * scale, true); // x-coordinate
	view.setFloat32(offset + 4, -0.4 * scale, true); // y-coordinate
	offset += 8;
	// Write the uint32 sense_id
	view.setUint32(offset, 8, true);
	offset += 4;
	// position
	view.setFloat32(offset, 0.0 * scale, true); // x-coordinate
	view.setFloat32(offset + 4, 0.4 * scale, true); // y-coordinate
	offset += 8;
	// Write the uint32 sense_id
	view.setUint32(offset, 8, true);
	offset += 4;

	// Create the WebGPU buffer and write the data
	const botModelBuffer = device.createBuffer({
		size: buffer.byteLength,
		usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
	});

	device.queue.writeBuffer(botModelBuffer, 0, buffer);

	return { pipeline, renderPassDescriptor, botModelBuffer };
}