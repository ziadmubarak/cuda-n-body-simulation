#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ const float G = 1.0f;
__device__ const float epsilon2 = 0.1f * 0.1f;
__device__ const float dt = 0.03f;
const int N = 1024;
const int p = 16;
const int numSteps = 800;
const float PI = 3.14159265f;

struct Body {
    float4 posMass;
    float3 velocity;
    float3 acceleration;
};

__device__ void body_body_force_calc(Body *BodyI, Body BodyJ, float3 *myAccel){
    float3 r;

    r.x = BodyJ.posMass.x - BodyI->posMass.x;
    r.y = BodyJ.posMass.y - BodyI->posMass.y;
    r.z = BodyJ.posMass.z - BodyI->posMass.z;

    float r_squared_plus_epsilon = (r.x * r.x) + (r.y * r.y) + (r.z * r.z) + epsilon2;
    float scalar_factor = G * BodyJ.posMass.w / (r_squared_plus_epsilon * sqrtf(r_squared_plus_epsilon));
    myAccel->x += (r.x * scalar_factor);
    myAccel->y += (r.y * scalar_factor);
    myAccel->z += (r.z * scalar_factor);
}

__device__ void tile_calculation(Body *myBody, Body *sharedBodies, float3 *myAccel, int p, int id, int k){
    for(int i = 0; i <= p - 1; i++) {
        if((k * p + i) != id){
            body_body_force_calc(myBody, sharedBodies[i], myAccel);
        }
    }
}

__global__ void calculate_forces(Body *devBodies){

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ Body sharedBodies[p];

    float3 myAccel = {0.0f, 0.0f, 0.0f};

    if(id < N) {
        for(int k = 0; k <= (N / p) - 1; k++) {
            sharedBodies[threadIdx.x] = devBodies[k * p + threadIdx.x];
            __syncthreads();
            tile_calculation(&devBodies[id], sharedBodies, &myAccel, p, id, k);
            __syncthreads();
        }
        devBodies[id].acceleration = myAccel;

        devBodies[id].velocity.x = (devBodies[id].velocity.x + (myAccel.x)*dt);
        devBodies[id].velocity.y = (devBodies[id].velocity.y + (myAccel.y)*dt);
        devBodies[id].velocity.z = (devBodies[id].velocity.z + (myAccel.z)*dt);

        devBodies[id].posMass.x = (devBodies[id].posMass.x + (devBodies[id].velocity.x)*dt);
        devBodies[id].posMass.y = (devBodies[id].posMass.y + (devBodies[id].velocity.y)*dt);
        devBodies[id].posMass.z = (devBodies[id].posMass.z + (devBodies[id].velocity.z)*dt);
    }
}

int main(){
    int threadsPerBlock = p;
    int blocksPerGrid= ((N + p - 1) / p);

    Body* bodies = (Body*)malloc(N*sizeof(Body));

    for(int i = 0; i <= N - 1; i++){
        float angle = ((float)rand() / RAND_MAX) * 2.0f * PI;  // random angle, 0 to 2π
        float radius = ((float)rand() / RAND_MAX) * 500.0f; // random distance from center, 0 to 500

        bodies[i].posMass.x = radius * cosf(angle);
        bodies[i].posMass.y = radius * sinf(angle);
        bodies[i].posMass.z = 0.0f;
        bodies[i].posMass.w = 1.0f;

        float speed = 15.0f;  
        bodies[i].velocity.x = -speed * sinf(angle);
        bodies[i].velocity.y = speed * cosf(angle);
        bodies[i].velocity.z = 0.0f;

        bodies[i].acceleration.x = 0.0f;  
        bodies[i].acceleration.y = 0.0f;
        bodies[i].acceleration.z = 0.0f;
    }

    Body *devBodies;
    cudaMalloc(&devBodies, N*sizeof(Body));

    cudaMemcpy(devBodies, bodies, N*sizeof(Body), cudaMemcpyHostToDevice);

    for(int step = 0; step < numSteps; step++) {
        calculate_forces<<<blocksPerGrid, threadsPerBlock>>>(devBodies);
        if(step % 2 == 0){
            cudaMemcpy(bodies, devBodies, N*sizeof(Body), cudaMemcpyDeviceToHost);
            char filename[50];
            sprintf(filename, "frames/frame_%04d.csv", step);
            FILE *file = fopen(filename, "w");
            for(int i = 0; i <= N-1; i++){
                fprintf(file, "%f, %f, %f\n", bodies[i].posMass.x, bodies[i].posMass.y, bodies[i].posMass.z);
            }
            fclose(file);
        }
    }


    printf("Body 0 acceleration: (%f, %f, %f)\n", bodies[0].acceleration.x, bodies[0].acceleration.y, bodies[0].acceleration.z);
    printf("Body %d acceleration: (%f, %f, %f)\n", N/2, bodies[N/2].acceleration.x, bodies[N/2].acceleration.y, bodies[N/2].acceleration.z);
    printf("Body %d acceleration: (%f, %f, %f)\n", N-1, bodies[N-1].acceleration.x, bodies[N-1].acceleration.y, bodies[N-1].acceleration.z);

    free(bodies);
    cudaFree(devBodies);

    return 0;

}