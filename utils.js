
class Bot {
	constructor() {
		this.color = [1, 1, 1, 1];
		this.position = [0, 0];
		this.direction = [0, 0];
		this.die_stay_breed = 0;
		this.age = 0;
		this.energy = 0;
		this.id = 0;
		this.decision = 0;
	}
}

export async function dumpBotsBuffer(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);

	const numBots = data.byteLength / 64;
	const bots = [];

	for (let i = 0; i < numBots; i++) {
		const byteOffset = i * 64;
		const bot = new Bot();

		bot.color[0] = view.getFloat32(byteOffset + 0, true);
		bot.color[1] = view.getFloat32(byteOffset + 4, true);
		bot.color[2] = view.getFloat32(byteOffset + 8, true);
		bot.color[3] = view.getFloat32(byteOffset + 12, true);
		bot.position[0] = view.getFloat32(byteOffset + 16, true);
		bot.position[1] = view.getFloat32(byteOffset + 20, true);
		bot.direction[0] = view.getFloat32(byteOffset + 24, true);
		bot.direction[1] = view.getFloat32(byteOffset + 28, true);
		bot.die_stay_breed = view.getUint32(byteOffset + 32, true);
		bot.age = view.getUint32(byteOffset + 36, true);
		bot.energy = view.getFloat32(byteOffset + 40, true);
		bot.id = view.getUint32(byteOffset + 44, true);
		bot.decision = view.getUint32(byteOffset + 48, true);

		bots.push(bot);
	}
	stagingBuffer.destroy();
	return bots;
}

export async function dumpBufferSense(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);

	const numBots = data.byteLength / 64;
	const bots = [];

	for (let i = 0; i < numBots; i++) {
		var sense = [];
		for (let j = 0; j < 8; j++) {
			const byteOffset = (i * 64) + (j * 8);
			sense.push(view.getInt32(byteOffset, true));
			sense.push(view.getFloat32(byteOffset + 4, true));
		}
		bots.push(sense);
	}
	stagingBuffer.destroy();
	return bots;
}

export async function dumpBufferInt(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);

	const u32Array = [];
	for (let i = 0; i < view.byteLength / 4; i++) {
		u32Array.push(view.getUint32(i * 4, true)); // The `true` is for little-endian
	}
	stagingBuffer.destroy();
	return u32Array;
}

export async function dumpBufferFloat(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);

	const u32Array = [];
	for (let i = 0; i < view.byteLength / 4; i++) {
		u32Array.push(view.getFloat32(i * 4, true)); // The `true` is for little-endian
	}
	stagingBuffer.destroy();
	return u32Array;
}

export async function dumpFoodBuffer(device, buffer) {
	const stagingBuffer = device.createBuffer({
		size: buffer.size, // Size must match the source buffer
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});
	const encoder = device.createCommandEncoder();
	encoder.copyBufferToBuffer(
		buffer, // Source buffer
		0, // Source offset
		stagingBuffer, // Destination buffer
		0, // Destination offset
		stagingBuffer.size // Size of data to copy
	);
	const commands = encoder.finish();
	device.queue.submit([commands]);
	await stagingBuffer.mapAsync(GPUMapMode.READ);
	const data = stagingBuffer.getMappedRange();
	const view = new DataView(data);
	const numFood = data.byteLength / 16;
	const foods = [];
	for (let i = 0; i < numFood; i++) {
		const byteOffset = i * 16;
		const f = {};
		f.position = [view.getFloat32(byteOffset + 0, true), view.getFloat32(byteOffset + 4, true)];
		f.energy = view.getFloat32(byteOffset + 8, true);
		foods.push(f);
	}
	stagingBuffer.destroy();
	return foods;
}
