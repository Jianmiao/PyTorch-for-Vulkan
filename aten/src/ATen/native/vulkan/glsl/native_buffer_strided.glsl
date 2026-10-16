#version 450 core
layout(std430) buffer;
layout(set = 0, binding = 0) buffer restrict writeonly Output { float data[]; } uOutput;
layout(set = 0, binding = 1) buffer restrict readonly Input { float data[]; } uInput;
layout(set = 0, binding = 2) uniform Params {
  uvec4 sizes;
  uvec4 input_strides;
  uvec4 output_strides;
  uvec4 info; // count, input offset, output offset, operation (0 copy, 5 silu, 7 sigmoid)
} p;
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
void main() {
  uint index = gl_GlobalInvocationID.x;
  if (index >= p.info.x) return;
  uint input_index = p.info.y;
  uint output_index = p.info.z;
  // Metadata order is width, height, channel, batch, with leading ones.
  for (int d = 0; d < 4; ++d) {
    uint coordinate = index % p.sizes[d];
    index /= p.sizes[d];
    input_index += coordinate * p.input_strides[d];
    output_index += coordinate * p.output_strides[d];
  }
  float value = uInput.data[input_index];
  if (p.info.w != 0u) {
    float sigmoid_value = 1.0 / (1.0 + exp(-value));
    value = p.info.w == 5u ? value * sigmoid_value : sigmoid_value;
  }
  uOutput.data[output_index] = value;
}
