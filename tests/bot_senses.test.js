import { test, assertEqual } from './test_utils.js';
import { dumpBufferSense } from '../utils.js';
import { setupSimulation } from '../simulation_setup.js';

test('bot_sense: bot senses another bot in front, not food behind', async () => {
	if (!navigator.gpu) return;
	const adapter = await navigator.gpu.requestAdapter();
	const device = await adapter.requestDevice();

	// Bot 0 at (0,0) facing right, Bot 1 at (0.05,0) (within vision), food at (-0.05,0) (behind)
	const bot_data = [
		{ die_stay_breed: 1, energy: 10, age: 0, color: [1, 0, 0, 1], position: [0, 0], direction: [1, 0], id: 0, decision: 0, brain_id: 0 },
		{ die_stay_breed: 1, energy: 10, age: 0, color: [0, 1, 0, 1], position: [0.05, 0], direction: [0, 1], id: 1, decision: 0, brain_id: 1 }
	];
	while (bot_data.length < 64) bot_data.push({ die_stay_breed: 0, energy: 0, age: 0, color: [0, 0, 0, 1], position: [0, 0], direction: [0, 0], id: 0, decision: 0, brain_id: 0 });

	const food_data = [
		{ position: [-0.05, 0], energy: 20 }
	];
	while (food_data.length < 1000) food_data.push({ position: [1000, 1000], energy: 0 });

	const { buffers, pipelines } = await setupSimulation(device, {
		MAX_BOTS: 64,
		INITIAL_BOT_COUNT: 2,
		bot_data,
		food_data
	});
	const { botsBuffer, botSensesBuffer } = buffers;
	const { botSensesPipeline } = pipelines;

	// Run bot_sense pipeline using indirect dispatch
	const { numBotsBuffer } = buffers;
	const encoder = device.createCommandEncoder();
	const pass = encoder.beginComputePass();
	pass.setPipeline(botSensesPipeline.pipeline);
	pass.setBindGroup(0, botSensesPipeline.bindGroup);
	pass.dispatchWorkgroupsIndirect(numBotsBuffer, 0);
	pass.end();
	device.queue.submit([encoder.finish()]);
	await device.queue.onSubmittedWorkDone();

	const senses = await dumpBufferSense(device, botSensesBuffer);

	// For bot 0, at least one ray should have id=1 (bot 1 detected), and no ray should have id=-1 (food is not a bot)
	const bot0_senses = senses[0];
	let foundBot1 = false;
	for (let i = 0; i < 8; ++i) {
		if (bot0_senses[i * 2] === 1) foundBot1 = true;
		assertEqual(bot0_senses[i * 2] !== -1 || bot0_senses[i * 2 + 1] === 0.1, true, 'No food should be detected as bot');
	}
	assertEqual(foundBot1, true, 'Bot 0 should sense Bot 1 in front');
});
