export function createBotSensePipeline(device, code, botsBuffer, botSensesBuffer) {
	const module = device.createShaderModule({
		label: 'bot sense compute module',
		code,
	});

	const pipeline = device.createComputePipeline({
		label: 'bot sense compute pipeline',
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
		],
	});

	return { pipeline, bindGroup };
}