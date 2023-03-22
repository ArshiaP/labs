#include <stdio.h>
#include <time.h>
#include <CL/cl.h>
#include <string.h>
#include <stdlib.h>
#define MAX_SOURCE_SIZE (0x100000)
int main(void)
{
    time_t start, end;
    start = clock();
    char tempstr[100];
    // Initialize the input string
    int i, n, ar[100];
    printf("Enter number of elements\n");
    scanf("%d", &n);

    printf("Enter elements\n");
    for (i = 0; i < n; i++)
        scanf("%d", ar + i);

    // Load the kernel source code into the array source_str
    FILE *fpeven, *fpodd;
    char *source_str_even, *source_str_odd;
    fpeven = fopen("q3even.cl", "r");
    fpodd = fopen("q3odd.cl", "r");
    source_str_even = (char *)malloc(MAX_SOURCE_SIZE);
    source_str_odd = (char *)malloc(MAX_SOURCE_SIZE);

    size_t size_even = fread(source_str_even, 1, MAX_SOURCE_SIZE, fpeven);
    size_t size_odd = fread(source_str_odd, 1, MAX_SOURCE_SIZE, fpodd);
    // puts(source_str_even);
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                         &ret_num_devices);
    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context,
                                                          device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
    // Create memory buffers on the device for input and output string
    printf("\nDone\n");
    cl_mem s_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(int), NULL, &ret);

    // cl_mem t_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(int), NULL, &ret);
    // Copy the input string into respective memory buffer
    for (i = 0; i < n; i++)
        printf("%d ", ar[i]);
    printf("\n");
    ret = clEnqueueWriteBuffer(command_queue, s_mem_obj, CL_TRUE, 0, n * sizeof(int), ar, 0, NULL, NULL);

    // Create a program from the kernel source
    cl_program program_even = clCreateProgramWithSource(context, 1, (const char **)&source_str_even, (const size_t *)&size_even, &ret);
    cl_program program_odd = clCreateProgramWithSource(context, 1, (const char **)&source_str_odd, (const size_t *)&size_odd, &ret);

    // Build the program
    ret = clBuildProgram(program_even, 1, &device_id, NULL, NULL, NULL);
    ret = clBuildProgram(program_odd, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel even_kernel = clCreateKernel(program_even, "even_kernel", &ret);
    cl_kernel odd_kernel = clCreateKernel(program_odd, "odd_kernel", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(even_kernel, 0, sizeof(cl_mem), (void *)&s_mem_obj);
    // ret = clSetKernelArg(even_kernel, 1, sizeof(cl_mem), (void *)&t_mem_obj);
    ret = clSetKernelArg(odd_kernel, 0, sizeof(cl_mem), (void *)&s_mem_obj);
    // ret = clSetKernelArg(odd_kernel, 1, sizeof(cl_mem), (void *)&t_mem_obj);
    // Set the global work size as string length
    size_t global_item_size = n / 2; // Process the entire lists
    size_t local_item_size = 1;
    // Execute the OpenCL kernel for entire string in parallel
    cl_event event;
    for (int i = 0; i < n / 2; i++)
    {
        ret = clEnqueueNDRangeKernel(command_queue, even_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
        ret = clEnqueueNDRangeKernel(command_queue, odd_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, &event);
    }
    time_t stime = clock();
    // kernel execution must be finished before calculating time
    ret = clFinish(command_queue);
    cl_ulong time_start, time_end;
    double total_time;
    // Find the kernel execution start time
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(time_start), &time_start, NULL);
    // Find the kernel execution end time
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(time_end), &time_end, NULL);
    total_time = (double)(time_end - time_start);
    // Read the result in memory buffer on the device to the local variable strres
    ret = clEnqueueReadBuffer(command_queue, s_mem_obj, CL_TRUE, 0, n * sizeof(int), ar, 0, NULL, NULL);
    for (i = 0; i < n; i++)
        printf("%d ", ar[i]);
    printf("\n");
    getchar();
    ret = clReleaseKernel(even_kernel);
    ret = clReleaseKernel(odd_kernel);

    ret = clReleaseProgram(program_even);
    ret = clReleaseProgram(program_odd);

    ret = clReleaseMemObject(s_mem_obj);
    // ret = clReleaseMemObject(t_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    end = clock();
    printf("\n\n Time taken to execute the KERNEL in milliseconds = %0.3f msec\n\n", total_time / 1000000);
    printf("\nTime taken to execute the whole program in seconds: %0.3f Seconds\n", (end - start) / (double)CLOCKS_PER_SEC);
    getchar();
    return 0;
}
// strtoggle.cl
