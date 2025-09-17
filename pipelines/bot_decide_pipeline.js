export function createBotDecidePipeline(device, code, botsBuffer, brainsBuffer, sensesBuffer) {
	const module = device.createShaderModule({
		label: 'bot decide compute module',
		code,
	});

	const pipeline = device.createComputePipeline({
		label: 'bot decide compute pipeline',
		layout: 'auto',
		compute: {
			module,
		},
	});

	// Bind Groups
	const bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {
					buffer: brainsBuffer,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: sensesBuffer,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: botsBuffer,
				},
			}
		],
	});

	return { pipeline, bindGroup };
}