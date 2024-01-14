#include <stdio.h>
#include <cstdlib>
#define CHANNELS 3


// Function to load the image from file
unsigned char* loadPPM(const char* path, int* width, int* height)
{
	FILE* file = fopen(path, "rb");

	if (!file) {
		fprintf(stderr, "Failed to open file\n");
		return NULL;
	}

	char header[3];
	fscanf(file, "%2s", header);
	if (header[0] != 'P' || header[1] != '6') {
		fprintf(stderr, "Invalid PPM file\n");
		return NULL;
	}

	fscanf(file, "%d %d", width, height);

	int maxColor;
	fscanf(file, "%d", &maxColor);

	fgetc(file);  // Skip single whitespace character

	unsigned char* img = (unsigned char*) malloc((*width) * (*height) * CHANNELS);
	if (!img) {
		fprintf(stderr, "Failed to allocate memory\n");
		return NULL;
	}

	fread(img, CHANNELS, *width * *height, file);

	fclose(file);

	return img;
}

// Function to write the matrix image to file
void writePPM(const char* path, unsigned char* img, int width, int height)
{
    FILE* file = fopen(path, "wb");

    if (!file) {
        fprintf(stderr, "Failed to open file\n");
        return;
    }

    fprintf(file, "P6\n%d %d\n255\n", width, height);

    fwrite(img, CHANNELS, width * height, file);

    fclose(file);
}
