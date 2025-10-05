

import { test, assertEqual } from './test_utils.js';
import { dumpBotsBuffer } from '../utils.js';
import { setupSimulation } from '../simulation_setup.js';

test('basic pipeline scenario: all bots die_stay_breed=1, decision=0', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	// All bots: die_stay_breed=1, decision=0
	const bot_data = [];
	for (let i = 0; i < 8; ++i) {
		bot_data.push({
			die_stay_breed: 1,
			energy: 10,
			age: 0,
			color: [0, 0, 0, 1],
			position: [i, i],
			direction: [0, 0],
			id: i,
			decision: 0,
			brain_id: 0
		});
	}
	while (bot_data.length < 64) bot_data.push({ die_stay_breed: 0, energy: 0, age: 0, color: [0, 0, 0, 1], position: [0, 0], direction: [0, 0], id: 0, decision: 0, brain_id: 0 });

	const { buffers, pipelines } = await setupSimulation(device, { MAX_BOTS: 64, INITIAL_BOT_COUNT: 8, bot_data });
	const { botsBuffer, scratchBuffer } = buffers;
	const { botStepperPipeline } = pipelines;
	const botsBefore = await dumpBotsBuffer(device, botsBuffer);

	// Run pipeline once for all bots
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(botStepperPipeline.pipeline);
	pass.setBindGroup(0, botStepperPipeline.bindGroup);
	pass.dispatchWorkgroups(1);
	pass.end();
	device.queue.submit([encoder.finish()]);
	await device.queue.onSubmittedWorkDone();
	const botsAfter = await dumpBotsBuffer(device, scratchBuffer);

	for (let i = 0; i < 8; ++i) {
		assertEqual(botsAfter[i].die_stay_breed, 1, `Bot ${i} should stay alive`);
		assertEqual(botsAfter[i].decision, 0, `Bot ${i} decision should remain 0`);
		assertEqual(botsAfter[i].age, botsBefore[i].age + 1, `Bot ${i} age should increment`);
		assertEqual(botsAfter[i].position[0], botsBefore[i].position[0], `Bot ${i} position x should not change`);
		assertEqual(botsAfter[i].position[1], botsBefore[i].position[1], `Bot ${i} position y should not change`);
	}
});
