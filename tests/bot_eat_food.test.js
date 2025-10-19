import { test, assertEqual } from './test_utils.js';
import { dumpBotsBuffer, dumpFoodBuffer, dumpBufferInt } from '../utils.js';
import { setupSimulation } from '../simulation_setup.js';

test('Single bot eats single food', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	const initialBotEnergy = 10;
	const foodEnergy = 20;
	const center = [0.5, 0.5];

	const bot_data = [
		{
			die_stay_breed: 1,
			energy: initialBotEnergy,
			age: 0,
			color: [1, 1, 0, 1],
			position: [center[0], center[1]],
			direction: [0, 1], // Facing up
			id: 42, // Unique nonzero id
			decision: 15, // EAT
			brain_id: 0
		}
	];
	while (bot_data.length < 64) bot_data.push({ die_stay_breed: 0, energy: 0, age: 0, color: [0, 0, 0, 1], position: [0, 0], direction: [0, 0], id: 0, decision: 0, brain_id: 0 });

	// Place food slightly in front of the bot, within vision range (e.g., 0.03 units ahead)
	const food_data = [
		{ position: [center[0], center[1] + 0.03], energy: foodEnergy }
	];
	while (food_data.length < 1000) food_data.push({ position: [1000, 1000], energy: 0 });

	const { buffers, pipelines } = await setupSimulation(device, {
		MAX_BOTS: 64,
		INITIAL_BOT_COUNT: 1,
		bot_data,
		food_data
	});


	const { scratchBuffer, foodsBuffer, botSensesBuffer } = buffers;
	const { botStepperPipeline, foodStepperPipeline } = pipelines;

	// Hardcode the food into the bot's sense buffer (simulate that the bot sees the food)
	// Each bot has 8 rays, each ray: u32 object_id, u32 type_id, f32 distance (16 bytes per sense)
	// We'll set the first sense to point to food id 0, type 2 (food), distance 0.01
	const senseArray = new ArrayBuffer(128); // 8 senses * 8 bytes (u32, u32, f32)
	const senseView = new DataView(senseArray);
	// Set first sense to food id 0, type 2, distance 0.01
	senseView.setUint32(0, 0, true); // object_id
	senseView.setUint32(4, 2, true); // type_id (2 = food)
	senseView.setFloat32(8, 0.01, true); // distance

	// Fill other rays with wall at max range (object_id=0, type_id=0, distance=0.1)
	for (let i = 1; i < 8; ++i) {
		senseView.setUint32(i * 16 + 0, 0, true); // object_id
		senseView.setUint32(i * 16 + 4, 0, true); // type_id (0 = wall)
		senseView.setFloat32(i * 16 + 8, 1, true); // distance (max range)
	}
	await device.queue.writeBuffer(botSensesBuffer, 0, senseArray);

	// Declare foodNextSlotBuffer once
	const foodNextSlotBuffer = buffers.foodNextSlotCounterBuffer;

	// Check foodNextSlot buffer before bot_step (should be 0)
	const stagingNextSlotBefore = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	{
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(foodNextSlotBuffer, 0, stagingNextSlotBefore, 0, 4);
		const commands = encoder.finish();
		device.queue.submit([commands]);
		await device.queue.onSubmittedWorkDone();
	}
	await stagingNextSlotBefore.mapAsync(GPUMapMode.READ);
	const nextSlotViewBefore = new DataView(stagingNextSlotBefore.getMappedRange());
	const nextSlotBefore = nextSlotViewBefore.getUint32(0, true);
	stagingNextSlotBefore.destroy();
	assertEqual(nextSlotBefore, 0, 'foodNextSlot should be 0 before bot queues to eat');

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

	// Check foodNextSlot buffer (should be 1 for food 0 if one bot queued)
	const stagingNextSlot = device.createBuffer({
		size: 4,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	{
		const encoder = device.createCommandEncoder();
		encoder.copyBufferToBuffer(foodNextSlotBuffer, 0, stagingNextSlot, 0, 4);
		const commands = encoder.finish();
		device.queue.submit([commands]);
		await device.queue.onSubmittedWorkDone();
	}
	await stagingNextSlot.mapAsync(GPUMapMode.READ);
	const nextSlotView = new DataView(stagingNextSlot.getMappedRange());
	const nextSlot = nextSlotView.getUint32(0, true);
	stagingNextSlot.destroy();
	assertEqual(nextSlot, 1, 'foodNextSlot should be 1 after one bot queues to eat');

	var foodSlots = await dumpBufferInt(device, buffers.foodSlotsBuffer);
	assertEqual(foodSlots[0], 0, 'Bot should be queued in the food slot after bot_step');


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

	// Bot should have gained energy
	assertEqual(bots[0].energy > initialBotEnergy, true, 'Bot should have gained energy');
	// Food should have no energy left
	assertEqual(foods[0].energy < foodEnergy, true, 'Food should have gained and lost same energy');
});
