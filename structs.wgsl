struct Bot {
  position: vec2f,
  velocity: vec2f,
	die_stay_breed: u32, // 0 = kill, 1 = nothing, 2 = duplicate 
  age: u32,
  energy: f32,
  id: u32,
};

struct Uniforms {
    deltaTime: f32,
};

const INPUT_LAYER_SIZE = 16;
const MIDDLE_LAYER_SIZE = 32;
const OUTPUT_LAYER_SIZE = 16;

struct Brain {
	W1: array<f32, INPUT_LAYER_SIZE * MIDDLE_LAYER_SIZE>,
	B1: array<f32, MIDDLE_LAYER_SIZE>,
	W2: array<f32, MIDDLE_LAYER_SIZE * OUTPUT_LAYER_SIZE>,
	B2: array<f32, OUTPUT_LAYER_SIZE>,
}

struct BotSense {
  type_id: u32,
  distance: f32,
}

struct BotSenses {
  senses: array<BotSense, INPUT_LAYER_SIZE/2>
}