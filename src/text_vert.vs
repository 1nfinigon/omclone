#version 100
attribute vec2 local_pos;
attribute vec2 uv;
uniform vec2 offset;
uniform vec2 world_offset;
uniform vec2 scale;
varying lowp vec2 tex_uv;
void main() {
    gl_Position = vec4((local_pos+offset+world_offset)*scale, 0, 1);
    tex_uv = uv;
}