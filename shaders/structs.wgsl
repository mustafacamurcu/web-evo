struct Bot {
  color: vec4f,
  position: vec2f,
  direction: vec2f,
	die_stay_breed: u32, // 0 = kill, 1 = nothing, 2 = duplicate 
  age: u32,
  energy: f32,
  id: u32,
  decision: u32, // 0-15
  brain_id: u32,
}; // 52 bytes size, padded to 64 bytes

struct Food {
  position: vec2f,
  energy: f32,
}

struct FoodSlots {
  slots: array<u32, 100>,
}

struct BotVertexData {
  color: vec4f,
  position: vec2f,
  direction: vec2f,
  active_senses: u32,
} // 48 bytes

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
  object_id: u32,
  type_id: u32,
  distance: f32,
}

struct BotSenses {
  senses: array<BotSense, INPUT_LAYER_SIZE/2>
}

fn rf(st: vec2<f32>) -> f32 {
    return fract(sin(dot(st, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn rv2(st: vec2<f32>) -> vec2<f32> {
    return vec2f(rf(st + vec2f(1.0, 0.0)), rf(st + vec2f(0.0, 1.0)));
}