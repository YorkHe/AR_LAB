#version 330 core

in vec2 TexCoords;

out vec4 color;

uniform sampler2D bgImage;

void main()
{
	color = vec4(texture(bgImage, TexCoords));
}