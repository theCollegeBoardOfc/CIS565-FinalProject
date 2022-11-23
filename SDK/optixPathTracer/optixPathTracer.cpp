//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
#define TINYOBJLOADER_IMPLEMENTATION
#include "support/tinyobj/tiny_obj_loader.h"
#include <glad/glad.h>  // Needs to be included before gl_interop

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION        // Implementation in sutil.cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION  //
#if defined( WIN32 )
#    pragma warning( push )
#    pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#    pragma warning( pop )
#endif

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cstring>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Camera.h>
#include <sutil/Exception.h>
#include <sutil/GLDisplay.h>
#include <sutil/Matrix.h>
//#include "SARScene.h"
#include <sutil/Trackball.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>
#include <sutil/Quaternion.h>
#include <optix_stack_size.h>

#include <cuda/whitted.h>
#include <sutil/Aabb.h>
#include <sutil/Camera.h>
#include <sutil/Preprocessor.h>
#include <sutil/sutilapi.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION        // Implementation in sutil.cpp
#define STB_IMAGE_WRITE_IMPLEMENTATION  //
#if defined( WIN32 )
#    pragma warning( push )
#    pragma warning( disable : 4267 )
#endif
// #include <support/tinygltf/tiny_gltf.h>

#include <GLFW/glfw3.h>

#include "optixPathTracer.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string.h>
#include <Windows.h>
#include <commdlg.h>
using namespace sutil;

bool resize_dirty = false;
bool minimized = false;

// Camera state
bool             camera_changed = true;
sutil::Camera    camera;
sutil::Trackball trackball;

// Mouse state
int32_t mouse_button = -1;

int32_t samples_per_launch = 16;

#if TEST
LaunchParams* d_params = nullptr;
LaunchParams   params = {};
int32_t                 width = 768;
int32_t                 height = 768;
#endif

//------------------------------------------------------------------------------
//
// Local types
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

template <typename T>
struct Record
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef Record<RayGenData>   RayGenRecord;
typedef Record<MissData>     MissRecord;
typedef Record<HitGroupData> HitGroupRecord;


struct Vertex
{
    float x, y, z, pad;
};


struct IndexedTriangle
{
    uint32_t v1, v2, v3, pad;
};


struct Instance
{
    float transform[12];
};

#if TEST
#else
struct PathTracerState
{
    OptixDeviceContext context = 0;

    OptixTraversableHandle         gas_handle = 0;  // Traversable handle for triangle AS
    CUdeviceptr                    d_gas_output_buffer = 0;  // Triangle AS memory
    CUdeviceptr                    d_vertices = 0;

    OptixModule                    ptx_module = 0;
    OptixPipelineCompileOptions    pipeline_compile_options = {};
    OptixPipeline                  pipeline = 0;

    OptixProgramGroup              raygen_prog_group = 0;
    OptixProgramGroup              radiance_miss_group = 0;
    OptixProgramGroup              occlusion_miss_group = 0;
    OptixProgramGroup              radiance_hit_group = 0;
    OptixProgramGroup              occlusion_hit_group = 0;

    CUstream                       stream = 0;
    Params                         params;
    Params* d_params;

    OptixShaderBindingTable        sbt = {};
};
#endif


//------------------------------------------------------------------------------
//
// Scene data
//
//------------------------------------------------------------------------------
//// Use std::vector to allow us to dynamically size for objs.
static std::vector<Vertex> sceneMeshPositions;
static std::vector<uint32_t> sceneMeshMaterial;
static std::map<int, Matrix4x4> sceneMeshTransforms;

const int32_t TRIANGLE_COUNT = 12;
const int32_t MAT_COUNT = 4;

const static std::array<Vertex, TRIANGLE_COUNT * 3> g_vertices =
{ {
        // Floor  -- white lambert
        // Plane
        {    -50.0f,    -50.0f,    -50.0f, 0.0f },
        {    -50.0f,    -50.0f,  50.f, 0.0f },
        {  50.0f,    -50.0f,  50.0f, 0.0f },
        {    -50.0f,    -50.0f,    -50.0f, 0.0f },
        {  50.0f,    -50.0f,  50.0f, 0.0f },
        {  50.0f,    -50.0f,    -50.0f, 0.0f },

        // Ceiling -- white lambert
        {    -50.0f,  50.0f,   -50.0f, 0.0f },
        {  50.0f,  50.0f,    -50.0f, 0.0f },
        {  50.0f,  50.0f,  50.0f, 0.0f },

        {    -50.0f,  50.0f,    -50.0f, 0.0f },
        {  50.0f,  50.0f,  50.0f, 0.0f },
        {    -50.0f,  50.0f,  50.0f, 0.0f },

        // Back wall -- white lambert
        {    -50.0f,    -50.0f,  50.0f, 0.0f },
        {    -50.0f,  50.0f,  50.0f, 0.0f },
        {  50.0f,  50.0f,  50.0f, 0.0f },

        {    -50.0f,    -50.0f,  50.0f, 0.0f },
        {  50.0f,  50.0f,  50.0f, 0.0f },
        {  50.0f,    -50.0f,  50.0f, 0.0f },

        // Right wall -- green lambert
        {    -50.0f,    -50.0f,    -50.0f, 0.0f },
        {    -50.0f,  50.0f,    -50.0f, 0.0f },
        {    -50.0f,  50.0f,  50.0f, 0.0f },

        {    -50.0f,    -50.0f,    -50.0f, 0.0f },
        {    -50.0f,  50.0f, 50.0f, 0.0f },
        {    -50.0f,    -50.0f,  50.0f, 0.0f },

        // Left wall -- red lambert
        {  50.0f,    -50.0f,    -50.0f, 0.0f },
        {  50.0f,    -50.0f,  50.0f, 0.0f },
        {  50.0f,  50.8f,  50.0f, 0.0f },

        {  50.0f,    -50.0f,    -50.0f, 0.0f },
        {  50.0f,  50.0f,  50.0f, 0.0f },
        { 50.0f,  50.0f,    -50.0f, 0.0f },

        // Ceiling light -- emmissive
        {  -20.0f,  49.0f,  -20.0f, 0.0f },
        {  -20.f,  49.0f,  20.0f, 0.0f },
        {  20.0f,  49.0f,  20.0f, 0.0f },

        {  -20.0f,  49.0f,  -20.0f, 0.0f },
        { 20.0f,  49.0f,  -20.0f, 0.0f },
        {  20.0f,  49.0f,  20.0f, 0.0f }
    } };

static std::array<uint32_t, TRIANGLE_COUNT> g_mat_indices = { {
    0, 0,                          // Floor         -- white lambert
    0, 0,                          // Ceiling       -- white lambert
    0, 0,                          // Back wall     -- white lambert
    1, 1,                          // Right wall    -- green lambert
    2, 2,                          // Left wall     -- red lambert
    3, 3,                           // Ceiling light -- emmissive
} };


void addHardCodeToDynamicScene() {
    sceneMeshPositions = std::vector<Vertex>(g_vertices.begin(), g_vertices.end());
    sceneMeshMaterial = std::vector<uint32_t>(g_mat_indices.begin(), g_mat_indices.end());
}

bool addObj(std::string filename) {
    std::string inputfile = filename;
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "../materials"; // Path to material files

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        return false;
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();
    int size = 0;

    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx =  attrib.vertices[3 * size_t(idx.vertex_index) + 0];
                tinyobj::real_t vy = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
                tinyobj::real_t vz = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

                /*if (v == 0) {
                    scene->faces[size].p1 = glm::vec3(vx, vy, vz);
                }
                else if (v == 1) {
                    scene->faces[size].p2 = glm::vec3(vx, vy, vz);
                }
                else if (v == 2) {
                    scene->faces[size].p3 = glm::vec3(vx, vy, vz);
                }
                else {
                    return false;
                }*/

                if (v == 0 || v == 1 || v == 2) {
                    sceneMeshPositions.push_back({ vx, vy, vz, 0.0 });
                }
                else {
                    return false;
                }

                // Check if `normal_index` is zero or positive. negative = no normal data
                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                }

                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
            }
            sceneMeshMaterial.push_back(0);
            index_offset += fv;
            size++;

            // per-face material
            shapes[s].mesh.material_ids[f];
        }
    }
    return true;
}


const std::array<float3, MAT_COUNT> g_emission_colors =
{ {
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    {  0.0f,  0.0f,  0.0f },
    { 15.0f, 15.0f,  5.0f }

} };


const std::array<float3, MAT_COUNT> g_diffuse_colors =
{ {
    { 0.80f, 0.80f, 0.80f },
    { 0.05f, 0.80f, 0.05f },
    { 0.80f, 0.05f, 0.05f },
    { 0.50f, 0.00f, 0.00f }
} };

//------------------------------------------------------------------------------
//
// GLFW callbacks
//
//------------------------------------------------------------------------------

static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
    {
        mouse_button = button;
        trackball.startTracking(static_cast<int>(xpos), static_cast<int>(ypos));
    }
    else
    {
        mouse_button = -1;
    }
}

#if TEST
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), width, height);
        camera_changed = true;
    }
}
#else
static void cursorPosCallback(GLFWwindow* window, double xpos, double ypos)
{
    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));

    if (mouse_button == GLFW_MOUSE_BUTTON_LEFT)
    {
        trackball.setViewMode(sutil::Trackball::LookAtFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
    else if (mouse_button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        trackball.setViewMode(sutil::Trackball::EyeFixed);
        trackball.updateTracking(static_cast<int>(xpos), static_cast<int>(ypos), params->width, params->height);
        camera_changed = true;
    }
}
#endif

#if TEST
static void windowSizeCallback_scene(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    width = res_x;
    height = res_y;
    camera_changed = true;
    resize_dirty = true;
}
#else
static void windowSizeCallback(GLFWwindow* window, int32_t res_x, int32_t res_y)
{
    // Keep rendering at the current resolution when the window is minimized.
    if (minimized)
        return;

    // Output dimensions must be at least 1 in both x and y.
    sutil::ensureMinimumSize(res_x, res_y);

    Params* params = static_cast<Params*>(glfwGetWindowUserPointer(window));
    params->width = res_x;
    params->height = res_y;
    camera_changed = true;
    resize_dirty = true;
}
#endif


static void windowIconifyCallback(GLFWwindow* window, int32_t iconified)
{
    minimized = (iconified > 0);
}


static void keyCallback(GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    if (action == GLFW_PRESS)
    {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE)
        {
            glfwSetWindowShouldClose(window, true);
        }
    }
    else if (key == GLFW_KEY_G)
    {
        // toggle UI draw
    }
}


static void scrollCallback(GLFWwindow* window, double xscroll, double yscroll)
{
    if (trackball.wheelEvent((int)yscroll))
        camera_changed = true;
}


//------------------------------------------------------------------------------
//
// Helper functions
// TODO: some of these should move to sutil or optix util header
//
//------------------------------------------------------------------------------

void printUsageAndExit(const char* argv0)
{
    std::cerr << "Usage  : " << argv0 << " [options]\n";
    std::cerr << "Options: --file | -f <filename>      File for image output\n";
    std::cerr << "         --launch-samples | -s       Number of samples per pixel per launch (default 16)\n";
    std::cerr << "         --no-gl-interop             Disable GL interop for display\n";
    std::cerr << "         --dim=<width>x<height>      Set image dimensions; defaults to 768x768\n";
    std::cerr << "         --help | -h                 Print this usage message\n";
    exit(0);
}

#if TEST
void initLaunchParams_scene(const sutil::Scene& scene) {
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width * height * sizeof(float4)
    ));
    params.frame_buffer = nullptr; // Will be set when output buffer is mapped

    params.subframe_index = 0u;

    const float loffset = scene.aabb().maxExtent();

    // TODO: add light support to sutil::Scene
    std::vector<Light> lights(2);
    lights[0].type = Light::Type::POINT;
    lights[0].point.color = { 1.0f, 1.0f, 0.8f };
    lights[0].point.intensity = 5.0f;
    lights[0].point.position = scene.aabb().center() + make_float3(loffset);
    lights[0].point.falloff = Light::Falloff::QUADRATIC;
    lights[1].type = Light::Type::POINT;
    lights[1].point.color = { 0.8f, 0.8f, 1.0f };
    lights[1].point.intensity = 3.0f;
    lights[1].point.position = scene.aabb().center() + make_float3(-loffset, 0.5f * loffset, -0.5f * loffset);
    lights[1].point.falloff = Light::Falloff::QUADRATIC;

    params.lights.count = static_cast<uint32_t>(lights.size());
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.lights.data),
        lights.size() * sizeof(Light)
    ));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(params.lights.data),
        lights.data(),
        lights.size() * sizeof(Light),
        cudaMemcpyHostToDevice
    ));

    params.miss_color = make_float3(0.1f);

    //CUDA_CHECK( cudaStreamCreate( &stream ) );
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(LaunchParams)));

    params.handle = scene.traversableHandle();

    cudaMemset(reinterpret_cast<void*>(params.width), 768, sizeof(int32_t));
    cudaMemset(reinterpret_cast<void*>(params.height), 768, sizeof(int32_t));
}

#else
void initLaunchParams(PathTracerState& state)
{
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&state.params.accum_buffer),
        state.params.width * state.params.height * sizeof(float4)
    ));
    state.params.frame_buffer = nullptr;  // Will be set when output buffer is mapped

    state.params.samples_per_launch = samples_per_launch;
    state.params.subframe_index = 0u;

    state.params.light.emission = make_float3(15.0f, 15.0f, 5.0f);
    state.params.light.corner = make_float3(0.0f, 49.0f, 0.0f);
    state.params.light.v1 = make_float3(0.0f, 0.0f, 10.0f);
    state.params.light.v2 = make_float3(-13.0f, 0.0f, 0.0f);
    state.params.light.normal = normalize(cross(state.params.light.v1, state.params.light.v2));
    state.params.handle = state.gas_handle;

    CUDA_CHECK(cudaStreamCreate(&state.stream));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_params), sizeof(Params)));

}
#endif

#if TEST
void handleCameraUpdate_scene(LaunchParams& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(width) / static_cast<float>(height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
    /*
    std::cerr
        << "Updating camera:\n"
        << "\tU: " << params.U.x << ", " << params.U.y << ", " << params.U.z << std::endl
        << "\tV: " << params.V.x << ", " << params.V.y << ", " << params.V.z << std::endl
        << "\tW: " << params.W.x << ", " << params.W.y << ", " << params.W.z << std::endl;
        */

}

#else
void handleCameraUpdate(Params& params)
{
    if (!camera_changed)
        return;
    camera_changed = false;

    camera.setAspectRatio(static_cast<float>(params.width) / static_cast<float>(params.height));
    params.eye = camera.eye();
    camera.UVWFrame(params.U, params.V, params.W);
}
#endif

#if TEST
void handleResize_scene(sutil::CUDAOutputBuffer<uchar4>& output_buffer)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(width, height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        width * height * sizeof(float4)
    ));
}
#else
void handleResize(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    if (!resize_dirty)
        return;
    resize_dirty = false;

    output_buffer.resize(params.width, params.height);

    // Realloc accumulation buffer
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&params.accum_buffer),
        params.width * params.height * sizeof(float4)
    ));
}
#endif

#if TEST
void updateState_scene(sutil::CUDAOutputBuffer<uchar4>& output_buffer, LaunchParams& params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.subframe_index = 0;

    handleCameraUpdate_scene(params);
    handleResize_scene(output_buffer);
}
#else 
void updateState(sutil::CUDAOutputBuffer<uchar4>& output_buffer, Params& params)
{
    // Update params on device
    if (camera_changed || resize_dirty)
        params.subframe_index = 0;

    handleCameraUpdate(params);
    handleResize(output_buffer, params);
}
#endif

#if TEST
void launchSubframe_scene(sutil::CUDAOutputBuffer<uchar4>& output_buffer, const sutil::Scene& scene)
{

    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_params),
        &params,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice,
        0 // stream
    ));

    OPTIX_CHECK(optixLaunch(
        scene.pipeline(),
        0,             // stream
        reinterpret_cast<CUdeviceptr>(d_params),
        sizeof(LaunchParams),
        scene.sbt(),
        width,  // launch width
        height, // launch height
        1       // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}

#else
void launchSubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, PathTracerState& state)
{
    // Launch
    uchar4* result_buffer_data = output_buffer.map();
    state.params.frame_buffer = result_buffer_data;
    CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(state.d_params),
        &state.params, sizeof(Params),
        cudaMemcpyHostToDevice, state.stream
    ));

    OPTIX_CHECK(optixLaunch(
        state.pipeline,
        state.stream,
        reinterpret_cast<CUdeviceptr>(state.d_params),
        sizeof(Params),
        &state.sbt,
        state.params.width,   // launch width
        state.params.height,  // launch height
        1                     // launch depth
    ));
    output_buffer.unmap();
    CUDA_SYNC_CHECK();
}
#endif

void displaySubframe(sutil::CUDAOutputBuffer<uchar4>& output_buffer, sutil::GLDisplay& gl_display, GLFWwindow* window)
{
    // Display
    int framebuf_res_x = 0;  // The display's resolution (could be HDPI res)
    int framebuf_res_y = 0;  //
    glfwGetFramebufferSize(window, &framebuf_res_x, &framebuf_res_y);
    gl_display.display(
        output_buffer.width(),
        output_buffer.height(),
        framebuf_res_x,
        framebuf_res_y,
        output_buffer.getPBO()
    );
}


static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}

#if TEST
void initCameraState_scene(const sutil::Scene& scene)
{
    camera = scene.camera();
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 1.0f, 0.0f));
    trackball.setGimbalLock(true);
}

void cleanup_scene()
{
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.lights.data)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.width)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(params.height)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
}
#else
void initCameraState()
{
    camera.setEye(make_float3(0.f, 0.f, -190.0f)); //20.0f, 20.0f, -90.0f
    camera.setLookat(make_float3(0.0f, 0.0f, 30.0f)); //20.0f, 20.0f, 30.0f
    camera.setUp(make_float3(0.0f, 1.0f, 0.0f));
    camera.setFovY(35.0f);
    camera_changed = true;

    trackball.setCamera(&camera);
    trackball.setMoveSpeed(10.0f);
    trackball.setReferenceFrame(
        make_float3(1.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 1.0f, 0.0f)
    );
    trackball.setGimbalLock(true);
}
#endif

#if TEST
#else
void createContext(PathTracerState& state)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    OptixDeviceContext context;
    CUcontext          cu_ctx = 0;  // zero means take the current context
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &context_log_cb;
    options.logCallbackLevel = 4;
    OPTIX_CHECK(optixDeviceContextCreate(cu_ctx, &options, &context));

    state.context = context;
}

void buildMeshAccel(PathTracerState& state)
{
    //
    // copy mesh data to device
    //

    //// TODO: Instead of parsing g_vertices.data, send over sceneMeshData instead.
    const size_t vertices_size_in_bytes = sceneMeshPositions.size() * sizeof(Vertex);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_vertices), vertices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(state.d_vertices),
        sceneMeshPositions.data(), vertices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_mat_indices = 0;
    const size_t mat_indices_size_in_bytes = sceneMeshMaterial.size() * sizeof(uint32_t);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mat_indices), mat_indices_size_in_bytes));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_mat_indices),
        sceneMeshMaterial.data(),
        mat_indices_size_in_bytes,
        cudaMemcpyHostToDevice
    ));

    //
    // Build triangle GAS
    //
    uint32_t triangle_input_flags[MAT_COUNT] =  // One per SBT record for this build input
    {
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
        OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT
    };

    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(Vertex);
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(sceneMeshPositions.size());
    triangle_input.triangleArray.vertexBuffers = &state.d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = MAT_COUNT;
    triangle_input.triangleArray.sbtIndexOffsetBuffer = d_mat_indices;
    triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = sizeof(uint32_t);
    triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = sizeof(uint32_t);

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        state.context,
        &accel_options,
        &triangle_input,
        1,  // num_build_inputs
        &gas_buffer_sizes
    ));

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));

    // non-compacted output
    CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
    size_t      compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size),
        compactedSizeOffset + 8
    ));

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result = (CUdeviceptr)((char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,                                  // CUDA stream
        &accel_options,
        &triangle_input,
        1,                                  // num build inputs
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_buffer_temp_output_gas_and_compacted_size,
        gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty,                      // emitted property list
        1                                   // num emitted properties
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_mat_indices)));

    size_t compacted_gas_size;
    CUDA_CHECK(cudaMemcpy(&compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost));

    if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes)
    {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&state.d_gas_output_buffer), compacted_gas_size));

        // use handle as input and output
        OPTIX_CHECK(optixAccelCompact(state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle));

        CUDA_CHECK(cudaFree((void*)d_buffer_temp_output_gas_and_compacted_size));
    }
    else
    {
        state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    }
}

void createModule(PathTracerState& state)
{
    OptixPayloadType payloadTypes[2] = {};
    // radiance prd
    payloadTypes[0].numPayloadValues = sizeof(radiancePayloadSemantics) / sizeof(radiancePayloadSemantics[0]);
    payloadTypes[0].payloadSemantics = radiancePayloadSemantics;
    // occlusion prd
    payloadTypes[1].numPayloadValues = sizeof(occlusionPayloadSemantics) / sizeof(occlusionPayloadSemantics[0]);
    payloadTypes[1].payloadSemantics = occlusionPayloadSemantics;

    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
    module_compile_options.numPayloadTypes = 2;
    module_compile_options.payloadTypes = payloadTypes;

    state.pipeline_compile_options.usesMotionBlur = false;
    state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    state.pipeline_compile_options.numPayloadValues = 0;
    state.pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    size_t      inputSize = 0;
    const char* input = sutil::getInputData(OPTIX_SAMPLE_NAME, OPTIX_SAMPLE_DIR, "optixPathTracer.cu", inputSize);

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
        state.context,
        &module_compile_options,
        &state.pipeline_compile_options,
        input,
        inputSize,
        LOG, &LOG_SIZE,
        &state.ptx_module
    ));
}


void createProgramGroups(PathTracerState& state)
{
    OptixProgramGroupOptions  program_group_options = {};

    {
        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = state.ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";

        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &raygen_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.raygen_prog_group
        ));
    }

    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_miss_group
        ));

        memset(&miss_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = state.ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__occlusion";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context, &miss_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.occlusion_miss_group
        ));
    }

    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.radiance_hit_group
        ));

        memset(&hit_prog_group_desc, 0, sizeof(OptixProgramGroupDesc));
        hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH = state.ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            state.context,
            &hit_prog_group_desc,
            1,  // num program groups
            &program_group_options,
            LOG, &LOG_SIZE,
            &state.occlusion_hit_group
        ));
    }
}


void createPipeline(PathTracerState& state)
{
    OptixProgramGroup program_groups[] =
    {
        state.raygen_prog_group,
        state.radiance_miss_group,
        state.occlusion_miss_group,
        state.radiance_hit_group,
        state.occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OPTIX_CHECK_LOG(optixPipelineCreate(
        state.context,
        &state.pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        LOG, &LOG_SIZE,
        &state.pipeline
    ));

    // We need to specify the max traversal depth.  Calculate the stack sizes, so we can specify all
    // parameters to optixPipelineSetStackSize.
    OptixStackSizes stack_sizes = {};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.raygen_prog_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_miss_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_miss_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.radiance_hit_group, &stack_sizes));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(state.occlusion_hit_group, &stack_sizes));

    uint32_t max_trace_depth = 2;
    uint32_t max_cc_depth = 0;
    uint32_t max_dc_depth = 0;
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        max_trace_depth,
        max_cc_depth,
        max_dc_depth,
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));

    const uint32_t max_traversal_depth = 1;
    OPTIX_CHECK(optixPipelineSetStackSize(
        state.pipeline,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        max_traversal_depth
    ));
}


void createSBT(PathTracerState& state)
{
    CUdeviceptr  d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));

    RayGenRecord rg_sbt = {};
    OPTIX_CHECK(optixSbtRecordPackHeader(state.raygen_prog_group, &rg_sbt));

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_raygen_record),
        &rg_sbt,
        raygen_record_size,
        cudaMemcpyHostToDevice
    ));


    CUdeviceptr  d_miss_records;
    const size_t miss_record_size = sizeof(MissRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_records), miss_record_size * RAY_TYPE_COUNT));

    MissRecord ms_sbt[2];
    OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_miss_group, &ms_sbt[0]));
    ms_sbt[0].data.bg_color = make_float4(0.0f);
    OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_miss_group, &ms_sbt[1]));
    ms_sbt[1].data.bg_color = make_float4(0.0f);

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_miss_records),
        ms_sbt,
        miss_record_size * RAY_TYPE_COUNT,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr  d_hitgroup_records;
    const size_t hitgroup_record_size = sizeof(HitGroupRecord);
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_hitgroup_records),
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT
    ));

    HitGroupRecord hitgroup_records[RAY_TYPE_COUNT * MAT_COUNT];
    for (int i = 0; i < MAT_COUNT; ++i)
    {
        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 0;  // SBT for radiance ray-type for ith material

            OPTIX_CHECK(optixSbtRecordPackHeader(state.radiance_hit_group, &hitgroup_records[sbt_idx]));
            hitgroup_records[sbt_idx].data.emission_color = g_emission_colors[i];
            hitgroup_records[sbt_idx].data.diffuse_color = g_diffuse_colors[i];
            hitgroup_records[sbt_idx].data.vertices = reinterpret_cast<float4*>(state.d_vertices);
        }

        {
            const int sbt_idx = i * RAY_TYPE_COUNT + 1;  // SBT for occlusion ray-type for ith material
            memset(&hitgroup_records[sbt_idx], 0, hitgroup_record_size);

            OPTIX_CHECK(optixSbtRecordPackHeader(state.occlusion_hit_group, &hitgroup_records[sbt_idx]));
        }
    }

    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_records),
        hitgroup_records,
        hitgroup_record_size * RAY_TYPE_COUNT * MAT_COUNT,
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = d_raygen_record;
    state.sbt.missRecordBase = d_miss_records;
    state.sbt.missRecordStrideInBytes = static_cast<uint32_t>(miss_record_size);
    state.sbt.missRecordCount = RAY_TYPE_COUNT;
    state.sbt.hitgroupRecordBase = d_hitgroup_records;
    state.sbt.hitgroupRecordStrideInBytes = static_cast<uint32_t>(hitgroup_record_size);
    state.sbt.hitgroupRecordCount = RAY_TYPE_COUNT * MAT_COUNT;
}


void cleanupState(PathTracerState& state)
{
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(state.raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_miss_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.radiance_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_hit_group));
    OPTIX_CHECK(optixProgramGroupDestroy(state.occlusion_miss_group));
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));


    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.sbt.hitgroupRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_vertices)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_gas_output_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.params.accum_buffer)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(state.d_params)));
}

float3 make_float3_from_double(double x, double y, double z)
{
    return make_float3(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
}

float4 make_float4_from_double(double x, double y, double z, double w)
{
    return make_float4(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), static_cast<float>(w));
}

void processGLTFNode(const tinygltf::Model& model, const tinygltf::Node& gltf_node, const Matrix4x4& parent_matrix)
{
    const Matrix4x4 translation = gltf_node.translation.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::translate(make_float3_from_double(
            gltf_node.translation[0],
            gltf_node.translation[1],
            gltf_node.translation[2]
        ));

    const Matrix4x4 rotation = gltf_node.rotation.empty() ?
        Matrix4x4::identity() :
        Quaternion(
            static_cast<float>(gltf_node.rotation[3]),
            static_cast<float>(gltf_node.rotation[0]),
            static_cast<float>(gltf_node.rotation[1]),
            static_cast<float>(gltf_node.rotation[2])
        ).rotationMatrix();

    const Matrix4x4 scale = gltf_node.scale.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::scale(make_float3_from_double(
            gltf_node.scale[0],
            gltf_node.scale[1],
            gltf_node.scale[2]
        ));

    std::vector<float> gltf_matrix;
    for (double x : gltf_node.matrix) {
        gltf_matrix.push_back(static_cast<float>(x));
    }
    const Matrix4x4 matrix = gltf_node.matrix.empty() ?
        Matrix4x4::identity() :
        Matrix4x4(reinterpret_cast<float*>(gltf_matrix.data())).transpose();

    const Matrix4x4 node_xform = parent_matrix * matrix * translation * rotation * scale;

    if (gltf_node.camera != -1)
    {
        const auto& gltf_camera = model.cameras[gltf_node.camera];
        std::cerr << "Processing camera '" << gltf_camera.name << "'\n"
            << "\ttype: " << gltf_camera.type << "\n";
        if (gltf_camera.type != "perspective")
        {
            std::cerr << "\tskipping non-perpective camera\n";
            return;
        }

        const float3 eye = make_float3(node_xform * make_float4_from_double(0.0f, 0.0f, 0.0f, 1.0f));
        const float3 up = make_float3(node_xform * make_float4_from_double(0.0f, 1.0f, 0.0f, 0.0f));
        const float  yfov = static_cast<float>(gltf_camera.perspective.yfov) * 180.0f / static_cast<float>(M_PI);

        std::cerr << "\teye   : " << eye.x << ", " << eye.y << ", " << eye.z << std::endl;
        std::cerr << "\tup    : " << up.x << ", " << up.y << ", " << up.z << std::endl;
        std::cerr << "\tfov   : " << yfov << std::endl;
        std::cerr << "\taspect: " << gltf_camera.perspective.aspectRatio << std::endl;

        //// set the already existing camera.
        camera.setFovY(yfov);
        camera.setAspectRatio(static_cast<float>(gltf_camera.perspective.aspectRatio));
        camera.setEye(eye);
        camera.setUp(up);
        //scene.addCamera(camera);
    }
    else if (gltf_node.mesh != -1)
    {
        sceneMeshTransforms[gltf_node.mesh] = node_xform;
        std::cout << "matrix for MESH " << gltf_node.mesh << " IS: " << std::endl;

        const float* matrixData = matrix.getData();

        for (int i = 0; i < 4; i++) {
            std::cout << matrixData[i * 4 + 0] << " , " << matrixData[i * 4 + 1] << " , " << matrixData[i * 4 + 2] << " , " << matrixData[i * 4 + 3] << std::endl;
        }

        //std::cout << node_xform.getData().
        //auto instance = std::make_shared<Scene::Instance>();
        //instance->transform = node_xform;
        //instance->mesh_idx = gltf_node.mesh;
        //instance->world_aabb = scene.meshes()[gltf_node.mesh]->object_aabb;
        //instance->world_aabb.transform(node_xform);
        //scene.addInstance(instance);
    }

    if (!gltf_node.children.empty())
    {
        for (int32_t child : gltf_node.children)
        {
            processGLTFNode(model, model.nodes[child], node_xform);
        }
    }
}

void loadGltfModel(std::string &filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret;
    if (filename.size() >= 4 && strncmp(filename.c_str() + filename.size() - 4, ".glb", 4) == 0)
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    else
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty())
        std::cerr << "glTF WARNING: " << warn << std::endl;
    if (!ret)
    {
        std::cerr << "Failed to load GLTF scene '" << filename << "': " << err << std::endl;
        return;
        //throw Exception(err.c_str());
    }

    //
    // Meshes
    //
    ////
    //// Process nodes's transforms
    ////
    std::vector<int32_t> root_nodes(model.nodes.size(), 1);
    for (auto& gltf_node : model.nodes)
        for (int32_t child : gltf_node.children)
            root_nodes[child] = 0;

    for (size_t i = 0; i < root_nodes.size(); ++i)
    {
        if (!root_nodes[i])
            continue;
        auto& gltf_node = model.nodes[i];

        processGLTFNode(model, gltf_node, Matrix4x4::identity());
    }

    // Get all meshes
    

    for (int midx = 0; midx < model.meshes.size(); midx++)
    {
        auto& gltf_mesh = model.meshes[midx];
        std::cerr << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";
        std::cerr << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;

        Matrix4x4 currMeshTransform = sceneMeshTransforms[midx];

        for (const tinygltf::Primitive& gltf_primitive : gltf_mesh.primitives)
        {
            std::cout << "primitive encountered" << std::endl;
            if (gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES) // Ignore non-triangle meshes
            {
                std::cerr << "\tNon-triangle primitive: skipping\n";
                continue;
            }

            std::vector<int> tmpIndices;
            std::vector<float3> tmpNormals, tmpVertices;

            const tinygltf::Accessor& idxAccessor = model.accessors[gltf_primitive.indices];
            const tinygltf::BufferView& idxBufferView = model.bufferViews[idxAccessor.bufferView];
            const tinygltf::Buffer& idxBuf = model.buffers[idxBufferView.buffer];

            // idxBuf.data.data() returns an unsigned char
            const uint8_t* a = idxBuf.data.data() + idxBufferView.byteOffset + idxAccessor.byteOffset;
            const int byteStride = idxAccessor.ByteStride(idxBufferView);
            const size_t count = idxAccessor.count;

            // Debugging purposes
            const float scalingFactor = 3;

            switch (idxAccessor.componentType)
            {
            case TINYGLTF_COMPONENT_TYPE_BYTE: 
                std::cout << "TINYGLTF_COMPONENT_TYPE_BYTE " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((char*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: 
                std::cout << "TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((uint8_t*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_SHORT: 
                std::cout << "TINYGLTF_COMPONENT_TYPE_SHORT " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((short*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: 
                std::cout << "TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((uint16_t*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_INT: 
                std::cout << "TINYGLTF_COMPONENT_TYPE_INT " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((int*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: 
                std::cout << "TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT " << tmpIndices.size() / 3 << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((uint32_t*)a));
                }
                break;
            default: break;
            }

            //const uint16_t* indices = reinterpret_cast<const uint16_t*>(&idxBuf.data[idx_gltf_accessor.byteOffset + idxBufferView.byteOffset]);
            assert(gltf_primitive.attributes.find("POSITION") != gltf_primitive.attributes.end());

            for (const auto& attribute : gltf_primitive.attributes)
            {
                const tinygltf::Accessor attribAccessor = model.accessors[attribute.second];
                const tinygltf::BufferView& bufferView = model.bufferViews[attribAccessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                const uint8_t* a = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
                const int byte_stride = attribAccessor.ByteStride(bufferView);

                const size_t count = attribAccessor.count;
                if (attribute.first == "POSITION")
                {
                    if (attribAccessor.type == TINYGLTF_TYPE_VEC3) {
                        if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                            for (size_t i = 0; i < count; i++, a += byte_stride) {
                                tmpVertices.push_back(*((float3*)a));
                            }
                        }
                        else {
                            std::cerr << "double precision positions not supported in gltf file" << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "unsupported position definition in gltf file" << std::endl;
                    }
                }
            }

            std::cerr << "\t\tNum triangles: " << tmpIndices.size() / 3 << std::endl;
            std::cerr << "\t\tNum verts: " << tmpVertices.size() << std::endl;

            const size_t triangleCount = tmpIndices.size() / 3;
            for (size_t i = 0; i < triangleCount; i++) {
                const uint32_t aIdx = tmpIndices[i * 3 + 0];
                const uint32_t bIdx = tmpIndices[i * 3 + 1];
                const uint32_t cIdx = tmpIndices[i * 3 + 2];

                float3 ta = tmpVertices[aIdx];
                float3 tb = tmpVertices[bIdx];
                float3 tc = tmpVertices[cIdx];

                const float4 aPos = currMeshTransform * float4{ta.x, ta.y, ta.z, 1} * scalingFactor;
                const float4 bPos = currMeshTransform * float4{tb.x, tb.y, tb.z, 1} * scalingFactor;
                const float4 cPos = currMeshTransform * float4{ tc.x, tc.y, tc.z, 1 } * scalingFactor;

                Vertex vertA = Vertex{aPos.x, aPos.y, aPos.z, 0};
                Vertex vertB = Vertex{ bPos.x, bPos.y, bPos.z, 0 };
                Vertex vertC = Vertex{ cPos.x, cPos.y, cPos.z, 0 };

                sceneMeshPositions.push_back(vertA);
                sceneMeshPositions.push_back(vertB);
                sceneMeshPositions.push_back(vertC);

                sceneMeshMaterial.push_back(0);
                sceneMeshMaterial.push_back(0);
                sceneMeshMaterial.push_back(0);

            }

            std::cerr << "\t\tNum triangles: " << sceneMeshPositions.size() / 3 << std::endl;
            // worry about bounding box performance later
            /*if (!pos_gltf_accessor.minValues.empty() && !pos_gltf_accessor.maxValues.empty())
            {
                mesh->object_aabb.include(Aabb(
                    make_float3_from_double(
                        pos_gltf_accessor.minValues[0],
                        pos_gltf_accessor.minValues[1],
                        pos_gltf_accessor.minValues[2]
                    ),
                    make_float3_from_double(
                        pos_gltf_accessor.maxValues[0],
                        pos_gltf_accessor.maxValues[1],
                        pos_gltf_accessor.maxValues[2]
                    )));
            }*/

            //// i have a funny feeling in normal buffer that the normals are calculated dynamically through the cuda.
            //auto normal_accessor_iter = gltf_primitive.attributes.find("NORMAL");
            //if (normal_accessor_iter != gltf_primitive.attributes.end())
            //{
            //    std::cerr << "\t\tHas vertex normals: true\n";
            //    normalBuffer.push_back(bufferViewFromGLTF<float3>(model, normal_accessor_iter->second));
            //}
            //else
            //{
            //    std::cerr << "\t\tHas vertex normals: false\n";
            //    normalBuffer.push_back(bufferViewFromGLTF<float3>(model, -1));
            //}

            /*for (size_t j = 0; j < GeometryData::num_textcoords; ++j)
            {
                char texcoord_str[128];
                snprintf(texcoord_str, 128, "TEXCOORD_%i", (int)j);
                auto texcoord_accessor_iter = gltf_primitive.attributes.find(texcoord_str);
                if (texcoord_accessor_iter != gltf_primitive.attributes.end())
                {
                    std::cerr << "\t\tHas texcoords_" << j << ": true\n";
                    mesh->texcoords[j].push_back(bufferViewFromGLTF<Vec2f>(model, scene, texcoord_accessor_iter->second));
                }
                else
                {
                    std::cerr << "\t\tHas texcoords_" << j << ": false\n";
                    mesh->texcoords[j].push_back(bufferViewFromGLTF<Vec2f>(model, scene, -1));
                }
            }*/

            /*auto color_accessor_iter = gltf_primitive.attributes.find("COLOR_0");
            if (color_accessor_iter != gltf_primitive.attributes.end())
            {
                std::cerr << "\t\tHas color_0: true\n";
                mesh->colors.push_back(bufferViewFromGLTF<Vec4f>(model, scene, color_accessor_iter->second));
            }
            else
            {
                std::cerr << "\t\tHas color_0: false\n";
                mesh->colors.push_back(bufferViewFromGLTF<Vec4f>(model, scene, -1));
            }*/
        }
    }
}
#endif

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char* argv[])
{
#if TEST
    params.width = 768;
    params.height = 768;

#else
    PathTracerState state;
    state.params.width = 768;
    state.params.height = 768;
#endif
    sutil::CUDAOutputBufferType output_buffer_type = sutil::CUDAOutputBufferType::GL_INTEROP;

    //sceneMeshData = std::vector<Vertex>();

    //
    // Parse command line options
    //

    std::string outfile;
    std::string infile = sutil::sampleDataFilePath("DuckDi/Duck.gltf");

    sutil::Scene scene;

    addHardCodeToDynamicScene();

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h")
        {
            printUsageAndExit(argv[0]);
        }
        else if (arg == "--no-gl-interop")
        {
            output_buffer_type = sutil::CUDAOutputBufferType::CUDA_DEVICE;
        }
        else if (arg == "--file" || arg == "-f")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            outfile = argv[++i];
        }
        else if (arg.substr(0, 6) == "--dim=")
        {
            const std::string dims_arg = arg.substr(6);
            int w, h;
            sutil::parseDimensions(dims_arg.c_str(), w, h);
            //state.params.width = w;
            //state.params.height = h;
        }
        else if (arg == "--launch-samples" || arg == "-s")
        {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            samples_per_launch = atoi(argv[++i]);
        }
        else if (arg == "--obj") {
            if (i >= argc - 1)
                printUsageAndExit(argv[0]);
            infile = argv[++i];
        }
        else
        {
            std::cerr << "Unknown option '" << argv[i] << "'\n";
            printUsageAndExit(argv[0]);
        }
    }

    try
    {
        char szFileName[MAX_PATH] = { 0 };
        OPENFILENAME openFileName = { 0 };
        openFileName.lStructSize = sizeof(OPENFILENAME);
        openFileName.nMaxFile = MAX_PATH;  //这个必须设置，不设置的话不会出现打开文件对话框
        openFileName.lpstrFilter = "Obj and gltf Files\0*.gltf;*.obj";
        openFileName.lpstrFile = szFileName;
        openFileName.nFilterIndex = 1;
        openFileName.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;
        if (GetOpenFileName(&openFileName))

        {
            infile =  openFileName.lpstrFile;
            if (infile.find(".gltf") != std::string::npos) {
#if TEST
                sutil::loadScene(infile.c_str(), scene);
                scene.finalize();

                OPTIX_CHECK(optixInit()); // Need to initialize function table
                initCameraState_scene(scene);
                initLaunchParams_scene(scene);
                
#else
                loadGltfModel(infile);
#endif
                std::cout << "gltfokay!" << std::endl;
            }
            else {
                addObj(infile);
            }
        }

#if TEST

#else
        initCameraState();

        //
        // Set up OptiX state
        //
        createContext(state);
        buildMeshAccel(state);
        createModule(state);
        createProgramGroups(state);
        createPipeline(state);
        createSBT(state);
        initLaunchParams(state);
#endif


        if (outfile.empty())
        {
#if TEST
            GLFWwindow* window = sutil::initUI("optixPathTracer", width, height);

#else
            GLFWwindow* window = sutil::initUI("optixPathTracer", state.params.width, state.params.height);
#endif
            glfwSetMouseButtonCallback(window, mouseButtonCallback);
            glfwSetCursorPosCallback(window, cursorPosCallback);
#if TEST
            glfwSetWindowSizeCallback(window, windowSizeCallback_scene);
#else
            glfwSetWindowSizeCallback(window, windowSizeCallback);
#endif
            glfwSetWindowIconifyCallback(window, windowIconifyCallback);
            glfwSetKeyCallback(window, keyCallback);
            glfwSetScrollCallback(window, scrollCallback);
#if TEST
            glfwSetWindowUserPointer(window, &params);
#else
            glfwSetWindowUserPointer(window, &state.params);
#endif


            //
            // Render loop
            //
            {
#if TEST
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    width,
                    height
                );
#else
                sutil::CUDAOutputBuffer<uchar4> output_buffer(
                    output_buffer_type,
                    state.params.width,
                    state.params.height
                );

                output_buffer.setStream(state.stream);
#endif
                sutil::GLDisplay gl_display;

                std::chrono::duration<double> state_update_time(0.0);
                std::chrono::duration<double> render_time(0.0);
                std::chrono::duration<double> display_time(0.0);

                do
                {
                    auto t0 = std::chrono::steady_clock::now();
                    glfwPollEvents();
#if TEST
                    updateState_scene(output_buffer, params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe_scene(output_buffer, scene);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;

                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++params.subframe_index;
#else
                    updateState(output_buffer, state.params);
                    auto t1 = std::chrono::steady_clock::now();
                    state_update_time += t1 - t0;
                    t0 = t1;

                    launchSubframe(output_buffer, state);
                    t1 = std::chrono::steady_clock::now();
                    render_time += t1 - t0;
                    t0 = t1;

                    displaySubframe(output_buffer, gl_display, window);
                    t1 = std::chrono::steady_clock::now();
                    display_time += t1 - t0;
                    sutil::displayStats(state_update_time, render_time, display_time);

                    glfwSwapBuffers(window);

                    ++state.params.subframe_index;
#endif
                } while (!glfwWindowShouldClose(window));
                CUDA_SYNC_CHECK();
            }

            sutil::cleanupUI(window);
        }
        else
        {
            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                sutil::initGLFW();  // For GL context
                sutil::initGL();
            }
#if TEST
            sutil::CUDAOutputBuffer<uchar4> output_buffer(output_buffer_type, width, height);
            handleCameraUpdate_scene(params);
            handleResize_scene(output_buffer);
            launchSubframe_scene(output_buffer, scene);
#else
            sutil::CUDAOutputBuffer<uchar4> output_buffer(
                output_buffer_type,
                state.params.width,
                state.params.height
            );

            handleCameraUpdate(state.params);
            handleResize(output_buffer, state.params);
            launchSubframe(output_buffer, state);
#endif

            sutil::ImageBuffer buffer;
            buffer.data = output_buffer.getHostPointer();
            buffer.width = output_buffer.width();
            buffer.height = output_buffer.height();
            buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

            sutil::saveImage(outfile.c_str(), buffer, false);

            if (output_buffer_type == sutil::CUDAOutputBufferType::GL_INTEROP)
            {
                glfwTerminate();
            }
        }
#if TEST
        cleanup_scene();
#else
        cleanupState(state);
#endif
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
