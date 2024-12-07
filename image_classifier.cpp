#include "jetson-inference/imageNet.h"
#include "jetson-inference/cudaMappedMemory.h"
#include "jetson-inference/loadImage.h"

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <model.engine> <labels.txt> <input_image.jpg>\n", argv[0]);
        return 1;
    }

    // Create the imageNet object
    imageNet* net = imageNet::Create();
    if (!net->LoadNetwork(argv[1], argv[2])) {
        printf("Failed to load network\n");
        return 1;
    }

    // Load the input image
    float* imgCUDA = NULL;
    int imgWidth, imgHeight;
    if (!loadImage(argv[3], &imgCUDA, &imgWidth, &imgHeight)) {
        printf("Failed to load image\n");
        return 1;
    }

    // Prepare image data
    imageNet::ImageData imgData;
    imgData.width = imgWidth;
    imgData.height = imgHeight;
    imgData.input = imgCUDA;

    // Classify the image
    float confidence = 0.0f;
    int classID = net->Classify(imgData, &confidence);

    if (classID >= 0) {
        printf("Class ID: %d\n", classID);
        printf("Class: %s\n", net->GetClassDesc(classID));
        printf("Confidence: %f\n", confidence);
    } else {
        printf("Failed to classify the image\n");
    }

    // Free CUDA memory and delete network object
    cudaFree(imgCUDA);
    delete net;

    return 0;
}
