#define LAYER_COUNT 4
#define MAX_NEURONS_PER_LAYER 64

struct NNData
{
	uint frameNumber;
	uint outputWidth;
	uint outputHeight;
	float learningRate;

	float rcpBatchSize;
	uint batchSize;
	float adamBeta1;
	float adamBeta2;

	float adamEpsilon;
	float adamBeta1T;
	float adamBeta2T;
	uint frequencies;

    uint32_t layerCount;
};

// 32-bit Xorshift random number generator
uint xorshift(inout uint rngState)
{
    rngState ^= rngState << 13;
    rngState ^= rngState >> 17;
    rngState ^= rngState << 5;
    return rngState;
}

// Jenkins's "one at a time" hash function
uint jenkinsHash(uint x)
{
    x += x << 10;
    x ^= x >> 6;
    x += x << 3;
    x ^= x >> 11;
    x += x << 15;
    return x;
}

// Converts unsigned integer into float int range <0; 1) by using 23 most significant bits for mantissa
float uintToFloat(uint x)
{
    return asfloat(0x3f800000 | (x >> 9)) - 1.0f;
}

// Initialize RNG for given pixel, and frame number (Xorshift-based version)
uint initRNG(uint2 pixelCoords, uint2 resolution, uint frameNumber)
{
    uint seed = dot(pixelCoords, uint2(1, resolution.x)) ^ jenkinsHash(frameNumber);
    return jenkinsHash(seed);
}

// Return random float in <0; 1) range (Xorshift-based version)
float rand(inout uint rngState)
{
    return uintToFloat(xorshift(rngState));
}

float randInRange(inout uint rng, float range)
{
    return (rand(rng) * 2.0f - 1.0f) * range;
}

float xavierUniformScale(inout uint rng, const uint nInputs, const uint nOutputs)
{
    return sqrt(6.0f / (nInputs + nOutputs));
}


// =========================================================================
//   Activation functions
// =========================================================================


#define LEAKY_RELU_SLOPE 0.01f

float relu(float x)
{
    return max(0.0f, x);
}

float reluDeriv(float x)
{
    return (x <= 0.0f) ? (0.0f) : (1.0f);
}

float leakyRelu(float x)
{
    return (x >= 0.0f) ? x : (x * LEAKY_RELU_SLOPE);
}

float leakyReluDeriv(float x)
{
    return (x <= 0.0f) ? (LEAKY_RELU_SLOPE) : (1.0f);
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float sigmoidDeriv(float x)
{
    return x * (1.0f - x);
}

#define FLOAT_PACKING_CONSTANT 1000000.0f
int packFloat(float x)
{
    return int(x * FLOAT_PACKING_CONSTANT);
}

float unpackFloat(int x)
{
    return float(x) / FLOAT_PACKING_CONSTANT;
}
