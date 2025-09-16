export function createBotStepperPipeline(device, code, botsBuffer, scratchBuffer) {
	const module = device.createShaderModule({
		label: 'bot stepper compute module',
		code,
	});

	const pipeline = device.createComputePipeline({
		label: 'bot stepper compute pipeline',
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
					buffer: scratchBuffer,
				},
			}
		],
	});

	return { pipeline, bindGroup };
}