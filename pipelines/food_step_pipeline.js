export function createFoodStepPipeline(device, code, botsBuffer, foodsBuffer, foodSlotsBuffer, foodNextSlotBuffer) {
	const module = device.createShaderModule({
		label: 'food stepper compute module',
		code,
	});

	const pipeline = device.createComputePipeline({
		label: 'food stepper compute pipeline',
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
					buffer: foodsBuffer,
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
		],
	});

	return { pipeline, bindGroup };
}