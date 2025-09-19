export function createPrefixSumPipeline(device, code, L1Buffer, L2Buffer, L3Buffer, scratchBuffer, numBotsBuffer) {
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
			{
				binding: 4,
				resource: {
					buffer: numBotsBuffer,
				},
			},
		],
	});

	return { pipeline1, pipeline2, pipeline3, L1BindGroup, L2BindGroup, L3BindGroup };
}