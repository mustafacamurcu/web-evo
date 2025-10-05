// Shared simulation setup for main.js and tests
// Exports a function to create all buffers and pipelines as in the app

import { makeShaderDataDefinitions, makeStructuredView, getSizeAndAlignmentOfUnsizedArrayElement } from 'https://greggman.github.io/webgpu-utils/dist/2.x/webgpu-utils.module.js';
import { createBotStepperPipeline } from './pipelines/bot_step_pipeline.js';
import { createFoodStepPipeline } from './pipelines/food_step_pipeline.js';
import { createBotDecidePipeline } from './pipelines/bot_decide_pipeline.js';
import { createPrefixSumPipeline } from './pipelines/prefix_sum_pipeline.js';
import { createReaperPipeline } from './pipelines/reaper_pipeline.js';
import { createNumBotsPipeline } from './pipelines/update_num_bots_pipeline.js';
import { createBotSensePipeline } from './pipelines/bot_sense_pipeline.js';
import { createBotVerticesPipeline } from './pipelines/bot_vertices_pipeline.js';
import { creatRepopulatePipeline } from './pipelines/repopulate_pipeline.js';

// Helper to fetch and combine WGSL code
async function fetchShaderCode(url) {
	const structs = await fetch('shaders/structs.wgsl');
	const response = await fetch('shaders/' + url);
	if (!response.ok) throw new Error('Failed to load shader');
	const code = await response.text();
	const structsCode = await structs.text();
	return structsCode + '\n' + code;
}

function createStorageBuffer(device, storageDefinitionName, code, objectCount, usage, data = null) {
	const defs = makeShaderDataDefinitions(code);
	const storageDefinition = defs.storages[storageDefinitionName];
	const objectSize = getSizeAndAlignmentOfUnsizedArrayElement(storageDefinition).size;
	const totalSize = objectSize * objectCount;
	const structuredView = makeStructuredView(storageDefinition, new ArrayBuffer(totalSize));
	const buffer = device.createBuffer({
		size: totalSize,
		usage: GPUBufferUsage.STORAGE | usage,
	});
	if (data) {
		structuredView.set(data);
		device.queue.writeBuffer(buffer, 0, structuredView.arrayBuffer);
	}
	return buffer;
}

// Main setup function

export async function setupSimulation(device, options = {}) {

	// Constants (can be parameterized)
	const MAX_FOOD = options.MAX_FOOD || 1000;
	const FOOD_SLOTS = options.FOOD_SLOTS || 100;
	const MAX_BOTS = options.MAX_BOTS || 64 * 64 * 1;
	const INITIAL_BOT_COUNT = options.INITIAL_BOT_COUNT || 700;
	const BOT_INITIAL_ENERGY = options.BOT_INITIAL_ENERGY || 100;

	// Allow custom data for tests, fallback to defaults if not provided
	let bot_data = options.bot_data;
	if (!bot_data) {
		bot_data = [];
		for (let i = 0; i < MAX_BOTS; ++i) {
			if (i < INITIAL_BOT_COUNT) {
				let bot = {
					color: [Math.random(), Math.random(), Math.random(), 1.0],
					position: [2 * (Math.random() - 0.5), 2 * (Math.random() - 0.5)],
					direction: [0.00002 * (Math.random() - 0.5), 0.00002 * (Math.random() - 0.5)],
					die_stay_breed: 1,
					energy: BOT_INITIAL_ENERGY,
					id: i,
					age: 0,
					decision: 0,
					brain_id: i
				};
				bot_data.push(bot);
			} else {
				let bot = {
					color: [0.0, 0.0, 0.0, 1.0],
					position: [1000.0, 1000.0],
					direction: [1000.0, 1000.0],
					die_stay_breed: 0,
					energy: 0,
					id: 0,
					age: 0,
					decision: 0,
					brain_id: 0
				};
			}
		}
	}
	let brain_data = options.brain_data;
	if (!brain_data) {
		brain_data = [];
		for (let i = 0; i < MAX_BOTS; ++i) {
			let brain = {
				W1: Array.from({ length: 16 * 32 }, () => Math.random() * 2 - 1),
				B1: Array.from({ length: 32 }, () => Math.random() * 2 - 1),
				W2: Array.from({ length: 32 * 16 }, () => Math.random() * 2 - 1),
				B2: Array.from({ length: 16 }, () => Math.random() * 2 - 1)
			}
			brain_data.push(brain);
		}
	}
	let brain_free_list_data = options.brain_free_list_data;
	if (!brain_free_list_data) {
		brain_free_list_data = [];
		for (let i = INITIAL_BOT_COUNT + 1; i < MAX_BOTS; ++i) {
			brain_free_list_data.push(i);
		}
	}
	let food_data = options.food_data;
	if (!food_data) {
		food_data = [];
		for (let i = 0; i < MAX_FOOD; ++i) {
			let food = {
				position: [2 * (Math.random() - 0.5), 2 * (Math.random() - 0.5)],
				energy: 20,
			};
			food_data.push(food);
		}
	}

	// Load shader codes
	const botSenseShaderCode = await fetchShaderCode('bot_sense.comp');
	const botDecideShaderCode = await fetchShaderCode('bot_decide.comp');
	const botStepShaderCode = await fetchShaderCode('bot_step.comp');
	const foodStepShaderCode = await fetchShaderCode('food_step.comp');
	const prefixSumShaderCode = await fetchShaderCode('prefix_sum.comp');
	const reaperShaderCode = await fetchShaderCode('reaper.comp');
	const numBotsShaderCode = await fetchShaderCode('update_num_bots.comp');
	const botVerticesShaderCode = await fetchShaderCode('bot_vertices.comp');
	const repopulateShaderCode = await fetchShaderCode('repopulate.comp');

	// Create buffers
	const botSensesBuffer = createStorageBuffer(device, 'bot_senses', botDecideShaderCode, MAX_BOTS, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const botBrainsBuffer = createStorageBuffer(device, 'bot_brains', botDecideShaderCode, MAX_BOTS, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, brain_data);
	const botsBuffer = createStorageBuffer(device, 'bots', botStepShaderCode, MAX_BOTS, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, bot_data);
	const scratchBuffer = createStorageBuffer(device, 'scratchBuffer', botStepShaderCode, MAX_BOTS, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const prefixBuffer = createStorageBuffer(device, 'prefixBuffer', prefixSumShaderCode, MAX_BOTS + MAX_BOTS / 64 + 64, GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
	const verticesBuffer = createStorageBuffer(device, "vertex_datas", botVerticesShaderCode, MAX_BOTS, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
	const brainFreeListBuffer = createStorageBuffer(device, "brain_free_list", reaperShaderCode, MAX_BOTS, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, brain_free_list_data);
	const foodsBuffer = createStorageBuffer(device, "foods", botSenseShaderCode, MAX_FOOD, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX, food_data);
	const foodSlotsBuffer = createStorageBuffer(device, "food_slots", botStepShaderCode, MAX_FOOD, GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);

	const foodNextSlotCounterBuffer = device.createBuffer({
		label: "food next slot counter",
		size: 4 * FOOD_SLOTS,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	const brainFreeListCounterBuffer = device.createBuffer({
		label: "free list counter",
		size: 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	device.queue.writeBuffer(brainFreeListCounterBuffer, 0, new Uint32Array([MAX_BOTS - INITIAL_BOT_COUNT]));
	const numBotsBuffer = device.createBuffer({
		label: "num_bots",
		size: 4 * 4 + 4 * 4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
	});
	device.queue.writeBuffer(numBotsBuffer, 0, new Uint32Array([INITIAL_BOT_COUNT, 1, 1]));

	// Create pipelines
	const botSensesPipeline = createBotSensePipeline(device, botSenseShaderCode, botsBuffer, botSensesBuffer, foodsBuffer);
	const botDecidePipeline = createBotDecidePipeline(device, botDecideShaderCode, botsBuffer, botBrainsBuffer, botSensesBuffer);
	const botStepperPipeline = createBotStepperPipeline(device, botStepShaderCode, botsBuffer, scratchBuffer, foodSlotsBuffer, foodNextSlotCounterBuffer, botSensesBuffer);
	const foodStepperPipeline = createFoodStepPipeline(device, foodStepShaderCode, scratchBuffer, foodsBuffer, foodSlotsBuffer, foodNextSlotCounterBuffer);
	const prefixSumPipeline = createPrefixSumPipeline(device, prefixSumShaderCode, prefixBuffer, scratchBuffer);
	const reaperPipeline = createReaperPipeline(device, reaperShaderCode, scratchBuffer, botsBuffer, prefixBuffer, botBrainsBuffer, brainFreeListBuffer, brainFreeListCounterBuffer);
	const numBotsPipeline = createNumBotsPipeline(device, numBotsShaderCode, prefixBuffer, numBotsBuffer);
	const botVerticesPipeline = createBotVerticesPipeline(device, botVerticesShaderCode, botsBuffer, botSensesBuffer, verticesBuffer);
	const repopulatePipeline = creatRepopulatePipeline(device, repopulateShaderCode, botsBuffer, numBotsBuffer, botBrainsBuffer, brainFreeListBuffer, brainFreeListCounterBuffer);

	// Return all relevant objects for tests
	return {
		buffers: {
			botsBuffer,
			botSensesBuffer,
			botBrainsBuffer,
			scratchBuffer,
			prefixBuffer,
			verticesBuffer,
			brainFreeListBuffer,
			foodsBuffer,
			foodSlotsBuffer,
			foodNextSlotCounterBuffer,
			brainFreeListCounterBuffer,
			numBotsBuffer,
		},
		pipelines: {
			botSensesPipeline,
			botDecidePipeline,
			botStepperPipeline,
			foodStepperPipeline,
			prefixSumPipeline,
			reaperPipeline,
			numBotsPipeline,
			botVerticesPipeline,
			repopulatePipeline
		},
		constants: {
			MAX_BOTS,
			INITIAL_BOT_COUNT,
			MAX_FOOD,
			FOOD_SLOTS,
		}
	};
}
