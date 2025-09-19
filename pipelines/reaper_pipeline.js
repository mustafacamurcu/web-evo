export function createReaperPipeline(device, code, scratchBuffer, botsBuffer, L1Buffer, L2Buffer, L3Buffer,
	botBrainsBuffer, brainFreeListBuffer, brainFreeListCounterBuffer) {
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
			{
				binding: 5,
				resource: {
					buffer: botBrainsBuffer,
				},
			},
			{
				binding: 6,
				resource: {
					buffer: brainFreeListBuffer,
				}
			},
			{
				binding: 7,
				resource: {
					buffer: brainFreeListCounterBuffer,
				}
			},
		],
	});

	return { pipeline, bindGroup };
}