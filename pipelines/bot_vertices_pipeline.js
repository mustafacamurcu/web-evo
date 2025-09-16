export function createBotVerticesPipeline(device, code, botsBuffer, botSensesBuffer, botVerticesBuffer) {
	const module = device.createShaderModule({
		label: 'make vertices compute module',
		code,
	});
	const pipeline = device.createComputePipeline({
		label: 'make vertices compute pipeline',
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
					buffer: botsBuffer,
				},
			},
			{
				binding: 1,
				resource: {
					buffer: botSensesBuffer,
				},
			},
			{
				binding: 2,
				resource: {
					buffer: botVerticesBuffer,
				},
			}
		],
	});

	return { pipeline, bindGroup };
}