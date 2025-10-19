import { test, assertEqual } from './test_utils.js';
import { dumpBotsBuffer, dumpBufferInt } from '../utils.js';
import { setupSimulation } from '../simulation_setup.js';

test('Bot repopulation creates new bot with copied brain', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	// Create a single parent bot
	const bot_data = [{
		die_stay_breed: 1,
		energy: 60,
		age: 5001,
		color: [1, 0, 0, 1],
		position: [0.5, 0.5],
		velocity: [1, 0],
		id: 123,
		decision: 0,
		brain_id: 0
	}];
	while (bot_data.length < 64) bot_data.push({ die_stay_breed: 0, energy: 0, age: 0, color: [0, 0, 0, 1], position: [0, 0], velocity: [0, 0], id: 0, decision: 0, brain_id: 0 });

	// Setup with initial bot count of 1
	const { buffers, pipelines } = await setupSimulation(device, {
		MAX_BOTS: 64,
		INITIAL_BOT_COUNT: 1,
		bot_data
	});

	const { botsBuffer, botBrainsBuffer, brainFreeListBuffer, brainFreeListCounterBuffer, numBotsBuffer } = buffers;
	const { repopulatePipeline } = pipelines;

	// Check initial state
	const botsBefore = await dumpBotsBuffer(device, botsBuffer);
	const numBotsBefore = await dumpBufferInt(device, numBotsBuffer);
	const brainCounterBefore = await dumpBufferInt(device, brainFreeListCounterBuffer);

	assertEqual(numBotsBefore[0], 1, 'Should start with 1 bot');
	assertEqual(botsBefore[0].id, 123, 'Parent bot should have correct ID');

	// Run repopulate pipeline
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(repopulatePipeline.pipeline);
	pass.setBindGroup(0, repopulatePipeline.bindGroup);
	pass.dispatchWorkgroups(1);
	pass.end();
	device.queue.submit([encoder.finish()]);
	await device.queue.onSubmittedWorkDone();

	// Check results
	const botsAfter = await dumpBotsBuffer(device, botsBuffer);
	const numBotsAfter = await dumpBufferInt(device, numBotsBuffer);
	const brainCounterAfter = await dumpBufferInt(device, brainFreeListCounterBuffer);

	// Check bot count increased
	assertEqual(numBotsAfter[0], 2, 'Bot count should increase by 1');

	// Check new bot properties
	const newBot = botsAfter[1];
	assertEqual(newBot.die_stay_breed, 1, 'New bot should be alive');
	assertEqual(newBot.age, 0, 'New bot should have age 0');
	assertEqual(newBot.energy, 50, 'New bot should have 50 energy');
	assertEqual(newBot.id, 123 + 1000000, 'New bot should have parent ID + 1000000');

	// Check brain counter decreased (brain allocated)
	assertEqual(brainCounterAfter[0], brainCounterBefore[0] - 1, 'Brain counter should decrease by 1');

	// Check that new bot has different brain ID from parent
	assertEqual(newBot.brain_id !== botsBefore[0].brain_id, true, 'New bot should have different brain ID');
	console.log('new bot:', newBot)
});

test('Repopulation should still work when 0 bots', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	// Create an array of empty bots
	const bot_data = [];
	while (bot_data.length < 64) bot_data.push({
		die_stay_breed: 0,
		energy: 0,
		age: 0,
		color: [0, 0, 0, 1],
		position: [0, 0],
		velocity: [0, 0],
		id: 0,
		decision: 0,
		brain_id: 0
	});

	// Setup with no bots
	const { buffers, pipelines } = await setupSimulation(device, {
		MAX_BOTS: 64,
		INITIAL_BOT_COUNT: 0,
		bot_data
	});

	const { botsBuffer, numBotsBuffer } = buffers;
	const { repopulatePipeline } = pipelines;

	// Check initial state
	const numBotsBefore = await dumpBufferInt(device, numBotsBuffer);
	assertEqual(numBotsBefore[0], 0, 'Should start with 0 bots');

	// Run repopulate pipeline
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(repopulatePipeline.pipeline);
	pass.setBindGroup(0, repopulatePipeline.bindGroup);
	pass.dispatchWorkgroups(1);
	pass.end();
	device.queue.submit([encoder.finish()]);
	await device.queue.onSubmittedWorkDone();

	// Check results
	const numBotsAfter = await dumpBufferInt(device, numBotsBuffer);
	const botsAfter = await dumpBotsBuffer(device, botsBuffer);
	assertEqual(numBotsAfter[0], 1, 'Bot count should increase to 1');
	assertEqual(botsAfter[0].die_stay_breed, 1, 'New bot should be alive');
	assertEqual(botsAfter[0].energy, 50, 'New bot should have 50 energy');
	console.log('New bot:', botsAfter[0]);
});

test('Repopulation does not occur when at max population', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	// Create bot data for near-max population (32 bots = half of 64)
	const bot_data = [];
	for (let i = 0; i < 32; i++) {
		bot_data.push({
			die_stay_breed: 1,
			energy: 60,
			age: 5001,
			color: [1, 0, 0, 1],
			position: [0.5, 0.5],
			velocity: [1, 0],
			id: i,
			decision: 0,
			brain_id: 0
		});
	}
	while (bot_data.length < 64) bot_data.push({ die_stay_breed: 0, energy: 0, age: 0, color: [0, 0, 0, 1], position: [0, 0], velocity: [0, 0], id: 0, decision: 0, brain_id: 0 });

	// Setup with max population
	const { buffers, pipelines } = await setupSimulation(device, {
		MAX_BOTS: 64,
		INITIAL_BOT_COUNT: 32,
		bot_data
	});

	const { numBotsBuffer } = buffers;
	const { repopulatePipeline } = pipelines;

	// Check initial state
	const numBotsBefore = await dumpBufferInt(device, numBotsBuffer);
	assertEqual(numBotsBefore[0], 32, 'Should start with 32 bots (max population)');

	// Run repopulate pipeline
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(repopulatePipeline.pipeline);
	pass.setBindGroup(0, repopulatePipeline.bindGroup);
	pass.dispatchWorkgroups(1);
	pass.end();
	device.queue.submit([encoder.finish()]);
	await device.queue.onSubmittedWorkDone();

	// Check results
	const numBotsAfter = await dumpBufferInt(device, numBotsBuffer);
	assertEqual(numBotsAfter[0], 32, 'Bot count should remain unchanged at max population');
});
