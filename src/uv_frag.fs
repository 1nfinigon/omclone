#version 100
varying lowp vec2 tex_uv;
uniform sampler2D tex;
void main() {
    gl_FragColor = texture2D(tex, tex_uv);
}