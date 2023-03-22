#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>

// max source size of the kernel string
#define MAX_SOURCE_SIZE (0x100000)

int main()
{
    // create two input vectors
    int list_size;
    printf("how many elements: ");
    scanf("%d", &list_size);
    int *a = (int *)malloc(sizeof(int) * list_size);
    int *b = (int *)malloc(sizeof(int) * list_size);

    // initialize input vectors
    for (int i = 0; i < list_size; i++)
    {
        a[i] = i;
        b[i] = i + 1;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("vector_add.cl", "r");
    if (!fp)
    {
        printf("failed to load source code");
        exit(0);
    }

    source_str = (char *)malloc(MAX_SOURCE_SIZE);

    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;

    cl_int ret;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, NULL, &ret);

    // Create memory buffers on the device for each vector A, B and C
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, list_size * sizeof(int), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, list_size * sizeof(int), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, list_size * sizeof(int), NULL, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, list_size * sizeof(int), a, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, list_size * sizeof(int), b, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source_str, (const size_t*)&source_size, &ret);

    // build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel object
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_mem_obj);

    // Execute the OpenCL kernel on the array
    size_t global_item_size = list_size;
    size_t local_item_size = 1;

    // Execute the kernel on the device
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    ret = clFinish(command_queue);

    // Read the memory buffer C on the device to the local variable C
    int *c = (int *)malloc(sizeof(int) * list_size);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, list_size * sizeof(int), c, 0, NULL, NULL);

    // Display the result to the screen
    for (int i = 0; i < list_size; i++)
        printf("%d + %d = %d\n", a[i], b[i], c[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(a);
    free(b);
    free(c);

}