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
