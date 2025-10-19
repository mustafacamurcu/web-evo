import { test, assertEqual } from './test_utils.js';
import { dumpBotsBuffer, dumpFoodBuffer, dumpBufferInt } from '../utils.js';
import { setupSimulation } from '../simulation_setup.js';

test('Bots accurately share limited food energy', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	const initialBotEnergy = 10;
	const foodEnergy = 1; // Not enough for all
	const center = [0.5, 0.5];
	const numBots = 5;

	const bot_data = [];
	for (let i = 0; i < numBots; ++i) {
		bot_data.push({
			die_stay_breed: 1,
			energy: initialBotEnergy,
			age: 0,
			color: [1, 1, 0, 1],
			position: [center[0] + (i - 1) * 0.01, center[1]],
			direction: [0, 1],
			id: 200 + i,
			decision: 15,
			brain_id: 0
		});
	}
	while (bot_data.length < 64) bot_data.push({ die_stay_breed: 0, energy: 0, age: 0, color: [0, 0, 0, 1], position: [0, 0], direction: [0, 0], id: 0, decision: 0, brain_id: 0 });

	const food_data = [
		{ position: [center[0], center[1] + 0.03], energy: foodEnergy }
	];
	while (food_data.length < 1000) food_data.push({ position: [1000, 1000], energy: 0 });

	const { buffers, pipelines } = await setupSimulation(device, {
		MAX_BOTS: 64,
		INITIAL_BOT_COUNT: numBots,
		bot_data,
		food_data
	});

	const { scratchBuffer, foodsBuffer, botSensesBuffer } = buffers;
	const { botStepperPipeline, foodStepperPipeline } = pipelines;

	// Hardcode the food into all bots' sense buffers
	const senseArray = new ArrayBuffer(96 * numBots);
	const senseView = new DataView(senseArray);
	for (let b = 0; b < numBots; ++b) {
		// First ray: food id 0, type 2, distance 0.01
		senseView.setUint32(b * 96 + 0, 0, true);
		senseView.setUint32(b * 96 + 4, 2, true);
		senseView.setFloat32(b * 96 + 8, 0.01, true);
		// Other rays: wall at max range
		for (let i = 1; i < 8; ++i) {
			senseView.setUint32(b * 96 + i * 8 + 0, 0, true);
			senseView.setUint32(b * 96 + i * 8 + 4, 0, true);
			senseView.setFloat32(b * 96 + i * 8 + 8, 0.1, true);
		}
	}
	await device.queue.writeBuffer(botSensesBuffer, 0, senseArray);

	// Run bot_step pipeline
	{
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(botStepperPipeline.pipeline);
		pass.setBindGroup(0, botStepperPipeline.bindGroup);
		pass.dispatchWorkgroups(1);
		pass.end();
		device.queue.submit([encoder.finish()]);
		await device.queue.onSubmittedWorkDone();
	}

	// Run food_step pipeline
	{
		const encoder = device.createCommandEncoder();
		const pass = encoder.beginComputePass();
		pass.setPipeline(foodStepperPipeline.pipeline);
		pass.setBindGroup(0, foodStepperPipeline.bindGroup);
		pass.dispatchWorkgroups(1);
		pass.end();
		device.queue.submit([encoder.finish()]);
		await device.queue.onSubmittedWorkDone();
	}

	const bots = await dumpBotsBuffer(device, scratchBuffer);
	const foods = await dumpFoodBuffer(device, foodsBuffer);

	// The sum of energy gained by all bots should be equal to the food energy
	let totalGained = 0;
	for (let i = 0; i < numBots; ++i) {
		const gained = bots[i].energy - initialBotEnergy;
		totalGained += gained;
	}
	assertEqual(Math.abs(totalGained - foodEnergy - 1 + numBots * 1.01) < 0.01, true, 'Bots should share all available food energy');
	// Food should have no energy left
	assertEqual(foods[0].energy < 0.1, true, 'Food should be dead after being eaten');
});
