#include"opencl_util.h"

cl_int error;
cl_platform_id platform;
cl_uint device_num;
cl_device_id device_ids[1];
cl_context context;
cl_kernel ycbcr2rgb;
cl_kernel idct_kernel;
cl_command_queue queue;
cl_program program;
size_t max_work_item_size[3];
size_t dct_max_work_item_size[3];

void checkerror(cl_int err)
{
    if (err != CL_SUCCESS)
    {
        cout << "Error:" << CLerrorstring(err) << endl;
        exit(1);
    }
}

cl_int getplatformID()
{
    cl_uint platform_num;
    cl_platform_id *platform_ids;

    //platform count
    cl_int error = clGetPlatformIDs(0, NULL, &platform_num);
    checkerror(error);

    if (platform_num == 0)
    {
        cout << "0 platform found" << endl;
        exit(1);
    }
    //make space
    if ((platform_ids = (cl_platform_id *)malloc(platform_num * sizeof(cl_platform_id))) == NULL)
    {
        cout << "fail to malloc memory for platform" << endl;
        exit(1);
    }
    //get platform
    error = clGetPlatformIDs(platform_num, platform_ids, NULL);
    checkerror(error);
    //Use the first platform
    platform = platform_ids[0];

    free(platform_ids);
    return CL_SUCCESS;
}

cl_uint getDeviceID()
{
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &device_num); //***if you want to decode by GPU change CL_DEVICE_TYPE***
    checkerror(error);
    if (device_num <= 0)
    {
        cout << "No device found" << endl;
        exit(1);
    }
    device_num=1;       //One device to decode is avaliable
    error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, device_num, device_ids, NULL);
    checkerror(error);

    return CL_SUCCESS;
}

size_t set_dct_size(size_t i)
{
    if (i > 8)
    {
        return 8;
    }
    else if (i < 8 && i >= 4)
    {
        return 4;
    }
    else if (i < 4 && i >= 2)
    {
        return 2;
    }
    else
    {
        return 1;
    }
}
void set_dct_worksize()
{
    error = clGetDeviceInfo(device_ids[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_size), max_work_item_size, NULL);
    checkerror(error);
    dct_max_work_item_size[0] = set_dct_size(max_work_item_size[0]);
    dct_max_work_item_size[1] = set_dct_size(max_work_item_size[1]);
    dct_max_work_item_size[2] = set_dct_size(max_work_item_size[2]);
}

void getcontext()
{
    //cl_context_properties contextProp[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = clCreateContext(NULL, device_num, device_ids, NULL, NULL, &error);
    checkerror(error);
}

cl_int loadKernel(const char *filename,cl_kernel * kernel,char *kernel_name)
{
    ifstream src_file(filename);
    if(!src_file.is_open())
    {
        return EXIT_FAILURE;
    }
    string src_prog(istreambuf_iterator<char>(src_file),(istreambuf_iterator<char>()));
    const char *src= src_prog.c_str();
    size_t length = src_prog.length();
    program = clCreateProgramWithSource(context,1,&src,&length,&error);
    checkerror(error);
    const char options[]="-I /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Kernel.framework/Versions/A/Headers";     //根据不同的操作系统修改
    error = clBuildProgram(program,device_num,device_ids,options,NULL,NULL);
    if(error !=CL_SUCCESS)
    {
        cout<<error;
        size_t len;
        char buffer[8*1024];
        clGetProgramBuildInfo(program,device_ids[0],CL_PROGRAM_BUILD_LOG,sizeof(buffer),buffer,&len);
        exit(1);
    }
    (* kernel)=clCreateKernel(program,kernel_name,&error);
    checkerror(error);
    return CL_SUCCESS;
}

void init_OpenCL()
{
    //get platform
    error = getplatformID();
    checkerror(error);
    //get device
    error = getDeviceID();
    checkerror(error);
    //get context
    getcontext();
    //set workgroups
    set_dct_worksize();
    //get queue
    queue = clCreateCommandQueue(context,device_ids[0],0,&error);
    checkerror(error);
    //lord kernel
    error = loadKernel("./ycbcr2rgb.cl",&ycbcr2rgb,"to_rgb");
    checkerror(error);
    error =loadKernel("./idct.cl",&idct_kernel,"idct_gpu");
    checkerror(error);
}

void to_rgb_gpu(int16_t* ycbcr,int16_t* rgb,unsigned int y,unsigned int x,unsigned int channels)
{
    cl_mem src_ycbcr=clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(int16_t)*y*x*channels,NULL,&error);
    checkerror(error);
    cl_mem det_rgb=clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(int16_t)*y*x*channels,NULL,&error);
    checkerror(error);
    if(src_ycbcr==NULL || det_rgb==NULL)
    {
        cout<<"Mem Buffer create error"<<endl;
        exit(0);
    }
    error = clEnqueueWriteBuffer(queue,src_ycbcr,CL_TRUE,0,sizeof(int16_t)*y*x*channels,ycbcr,0,NULL,NULL);
    checkerror(error);
    clSetKernelArg(ycbcr2rgb,0,sizeof(cl_mem),(void *)&src_ycbcr);
    clSetKernelArg(ycbcr2rgb,1,sizeof(cl_mem),(void *)&det_rgb);
    clSetKernelArg(ycbcr2rgb,2,sizeof(cl_uint),&y);
    clSetKernelArg(ycbcr2rgb,3,sizeof(cl_uint),&x);

    size_t max_Local[2];
    error = clGetKernelWorkGroupInfo(ycbcr2rgb,device_ids[0],CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),NULL,max_Local);
    checkerror(error);

    size_t GWS[2], LWS[2];
    LWS[0] = min((unsigned int)max_Local[0],min((unsigned int)max_work_item_size[0], y));
    LWS[1] = min((unsigned int)max_Local[0],min((unsigned int)max_work_item_size[1], x));
    GWS[0] = ceil(((float)y)/((float)LWS[0]))*LWS[0];
    GWS[1] = ceil(((float)x)/((float)LWS[1]))*LWS[1];
    clEnqueueNDRangeKernel(queue,ycbcr2rgb,2,NULL,GWS,LWS,0,NULL,NULL);
    clEnqueueReadBuffer(queue,det_rgb,CL_TRUE,0,sizeof(int16_t)*x*y*channels,rgb,0,NULL,NULL);
    clFinish(queue);
}

void idct_gpu(int16_t *image,int16_t *det,unsigned int blocks)
{
    cl_mem image_data=clCreateBuffer(context,CL_MEM_READ_ONLY,sizeof(int16_t)*blocks,NULL,&error);
    checkerror(error);
    cl_mem det_data=clCreateBuffer(context,CL_MEM_WRITE_ONLY,sizeof(int16_t)*blocks,NULL,&error);
    checkerror(error);
    if(image_data == NULL ||det_data==NULL)
    {
        cout<<"MEM Buffer crater error"<<endl;
        exit(1);
    }
    error = clEnqueueWriteBuffer(queue,image_data,CL_TRUE,0,sizeof(int16_t)*blocks,image,0,NULL,NULL);
    checkerror(error);
    clSetKernelArg(idct_kernel,0,sizeof(cl_mem),(void *)&image_data);
    clSetKernelArg(idct_kernel,1,sizeof(cl_mem),(void *)&det_data);
    clSetKernelArg(idct_kernel,2,sizeof(cl_uint),&blocks);
 
    size_t GWS[1],LWS[1];
    GWS[0] = blocks;
    LWS[0] = 1;
    clEnqueueNDRangeKernel(queue,idct_kernel,1,NULL,GWS,LWS,0,NULL,NULL);
    clEnqueueReadBuffer(queue,det_data,CL_TRUE,0,sizeof(int16_t)*blocks,det,0,NULL,NULL);
    clFinish(queue);
}
const char *CLerrorstring(cl_int err)
{
    switch (err)
    {
    case CL_DEVICE_NOT_FOUND:
        return "Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:
        return "Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
        return "Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
        return "Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
        return "Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
        return "Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
        return "Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
        return "Program build failure";
    case CL_MAP_FAILURE:
        return "Map failure";
    case CL_INVALID_VALUE:
        return "Invalid value";
    case CL_INVALID_DEVICE_TYPE:
        return "Invalid device type";
    case CL_INVALID_PLATFORM:
        return "Invalid platform";
    case CL_INVALID_DEVICE:
        return "Invalid device";
    case CL_INVALID_CONTEXT:
        return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
        return "Invalid command queue";
    case CL_INVALID_HOST_PTR:
        return "Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
        return "Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
        return "Invalid image size";
    case CL_INVALID_SAMPLER:
        return "Invalid sampler";
    case CL_INVALID_BINARY:
        return "Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
        return "Invalid build options";
    case CL_INVALID_PROGRAM:
        return "Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
        return "Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
        return "Invalid kernel definition";
    case CL_INVALID_KERNEL:
        return "Invalid kernel";
    case CL_INVALID_ARG_INDEX:
        return "Invalid argument index";
    case CL_INVALID_ARG_VALUE:
        return "Invalid argument value";
    case CL_INVALID_ARG_SIZE:
        return "Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
        return "Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
        return "Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
        return "Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "Invalid event wait list";
    case CL_INVALID_EVENT:
        return "Invalid event";
    case CL_INVALID_OPERATION:
        return "Invalid operation";
    case CL_INVALID_GL_OBJECT:
        return "Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:
        return "Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
        return "Invalid mip-map level";
    default:
        return "Unknown";
    }
}

void contextCallback(const char *err, const void *private_info, size_t cb, void *usr_data)
{
    cout << "Error in callback:" <<*err<< endl;
    exit(EXIT_FAILURE);
}
