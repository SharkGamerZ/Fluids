#version 330 core

// Fragment Shader
//     Prende in input i frammenti (pixel) e
//     restituisce in output il colore dei pixel (in questo caso arancione)
in float colorAlpha;
out vec4 FragColor;

void main()
{
    float val1 = 0.0;
    float val2 = 0.0;
    if (colorAlpha < 0.0) val1 = -colorAlpha;
    else val2 = colorAlpha;

    FragColor = vec4(0.0, val1, val2, 1.0);
}
