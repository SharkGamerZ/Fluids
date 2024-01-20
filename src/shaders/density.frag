#version 330 core

// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
in float colorAlpha;
out vec4 FragColor;

void main()
{
    FragColor = vec4(colorAlpha, colorAlpha, colorAlpha, colorAlpha);
}
