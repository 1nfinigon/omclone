#version 100
attribute vec2 local_pos;
attribute vec2 uv;
uniform vec2 offset;
uniform vec2 world_offset;
uniform float angle;
uniform float scale;
varying lowp vec2 tex_uv;
void main() {
    //TODO: Check if row-major or column-major
    mat2 rotation = mat2(cos(angle),-sin(angle),
                    sin(angle),cos(angle));
    gl_Position = vec4(((rotation*local_pos)+offset+world_offset)*scale, 0, 1);
    tex_uv = uv;
}