export function creatRepopulatePipeline(device, code, botsBuffer, numBotsBuffer,
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
					buffer: numBotsBuffer,
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
					buffer: botBrainsBuffer,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: brainFreeListBuffer,
				}
			},
			{
				binding: 4,
				resource: {
					buffer: brainFreeListCounterBuffer,
				}
			}
		],
	});

	return { pipeline, bindGroup };
}