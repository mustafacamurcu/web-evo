export function createNumBotsPipeline(device, code, prefixBuffer, numBotsBuffer) {
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
					buffer: prefixBuffer,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: numBotsBuffer,
				}
			}
		],
	});

	return { pipeline, bindGroup };
}