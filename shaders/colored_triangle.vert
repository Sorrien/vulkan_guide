#version 450

layout (location = 0) out vec4 outColor;

void main() 
{
	//const array of positions for the triangle
	const vec3 positions[3] = vec3[3](
		vec3(1.f,1.f, 0.0f),
		vec3(-1.f,1.f, 0.0f),
		vec3(0.f,-1.f, 0.0f)
	);

	//const array of colors for the triangle
	const vec4 colors[3] = vec4[3](
		vec4(1.0f, 0.0f, 0.0f, 1.0f), //red
		vec4(0.0f, 1.0f, 0.0f, 1.0f), //green
		vec4(00.f, 0.0f, 1.0f, 1.0f)  //blue
	);

	//output the position of each vertex
	gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
	outColor = colors[gl_VertexIndex];
}