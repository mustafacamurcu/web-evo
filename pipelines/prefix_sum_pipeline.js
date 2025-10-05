export function createPrefixSumPipeline(device, code, prefixBuffer, scratchBuffer) {
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
	const bindGroup1 = device.createBindGroup({
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
					buffer: prefixBuffer,
				},
			},
		],
	});

	const bindGroup2 = device.createBindGroup({
		layout: pipeline2.getBindGroupLayout(0),
		entries: [
			{
				binding: 1,
				resource: {
					buffer: prefixBuffer,
				},
			},
		],
	});

	const bindGroup3 = device.createBindGroup({
		layout: pipeline3.getBindGroupLayout(0),
		entries: [
			{
				binding: 1,
				resource: {
					buffer: prefixBuffer,
				},
			},
		],
	});

	return { pipeline1, pipeline2, pipeline3, bindGroup1, bindGroup2, bindGroup3 };
}