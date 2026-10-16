#version 450 core

layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output {
  float data[];
} uOutput;
layout(set = 0, binding = 1) buffer restrict readonly Input {
  float data[];
} uInput;
layout(set = 0, binding = 2) uniform Params {
  uint count;
  uint operation;
} uParams;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= uParams.count) {
    return;
  }
  float x = uInput.data[index];
  float s = 1.0 / (1.0 + exp(-x));
  uOutput.data[index] = uParams.operation == 5u ? x * s : s;
}
