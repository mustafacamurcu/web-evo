export function createBotStepperPipeline(device, code, botsBuffer, scratchBuffer, foodSlotsBuffer, foodNextSlotBuffer, botSensesBuffer) {
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
			},
			{
				binding: 2,
				resource: {
					buffer: foodSlotsBuffer,
				},
			},
			{
				binding: 3,
				resource: {
					buffer: foodNextSlotBuffer,
				},
			},
			{
				binding: 4,
				resource: {
					buffer: botSensesBuffer,
				},
			},
		],
	});

	return { pipeline, bindGroup };
}