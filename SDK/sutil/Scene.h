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

#pragma once

#include <cuda/BufferView.h>
#include <cuda/MaterialData.h>
#include <cuda/whitted.h>
#include <sutil/Aabb.h>
#include <sutil/Camera.h>
#include <sutil/Matrix.h>
#include <sutil/Preprocessor.h>
#include <sutil/sutilapi.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>


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

#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/Light.h>
#include <cuda/MaterialData.h>

#define TEST 1

enum RayType
{
    RAY_TYPE_RADIANCE = 0,
    RAY_TYPE_OCCLUSION = 1,
    RAY_TYPE_COUNT
};


#define PAYLOAD_TYPE_RADIANCE  OPTIX_PAYLOAD_TYPE_ID_0
#define PAYLOAD_TYPE_OCCLUSION OPTIX_PAYLOAD_TYPE_ID_1


struct RadiancePRD
{
    // these are produced by the caller, passed into trace, consumed/modified by CH and MS and consumed again by the caller after trace returned.
    float3       attenuation;
    unsigned int seed;
    int          depth;

    // these are produced by CH and MS, and consumed by the caller after trace returned.
    float3       emitted;
    float3       radiance;
    float3       origin;
    float3       direction;
    int          done;
};


const unsigned int radiancePayloadSemantics[18] =
{
    // RadiancePRD::attenuation
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::seed
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::depth
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE,
    // RadiancePRD::emitted
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::radiance
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE,
    // RadiancePRD::origin
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::direction
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE,
    // RadiancePRD::done
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};


const unsigned int occlusionPayloadSemantics[1]
{
    // occlduded
    OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
};


struct ParallelogramLight
{
    float3 corner;
    float3 v1, v2;
    float3 normal;
    float3 emission;
};

#if TEST
struct LaunchParams
{
    unsigned int             width;
    unsigned int             height;
    unsigned int             subframe_index;
    unsigned int            samples_per_launch;

    float4* accum_buffer;
    uchar4* frame_buffer;
    int                      max_depth;
    float                    scene_epsilon;

    float3                   eye;
    float3                   U;
    float3                   V;
    float3                   W;

    BufferView<ParallelogramLight>        lights;
    float3                   miss_color;
    OptixTraversableHandle   handle;
};
#else

struct Params
{
    unsigned int subframe_index;
    float4* accum_buffer;
    uchar4* frame_buffer;
    unsigned int width;
    unsigned int height;
    unsigned int samples_per_launch;

    float3       eye;
    float3       U;
    float3       V;
    float3       W;

    ParallelogramLight     light; // TODO: make light list
    OptixTraversableHandle handle;
};
#endif

struct RayGenData
{
};


struct MissData
{
    float4 bg_color;
};


struct HitGroupData
{
    GeometryData geometry_data;
    MaterialData material_data;
    float3  emission_color;
    float3  diffuse_color;
    float4* vertices;
};


namespace sutil
{


class Scene
{
public:
    SUTILAPI Scene();
    SUTILAPI ~Scene();

    struct Instance
    {
        Matrix4x4                         transform;
        Aabb                              world_aabb;

        int                               mesh_idx;
    };

    struct MeshGroup
    {
        std::string                       name;

        std::vector<GenericBufferView>    indices;
        std::vector<BufferView<float3> >  positions;
        std::vector<BufferView<float3> >  normals;
        std::vector<BufferView<Vec2f> >   texcoords[GeometryData::num_textcoords];
        std::vector<BufferView<Vec4f> >   colors;

        std::vector<int32_t>              material_idx;

        OptixTraversableHandle            gas_handle = 0;
        CUdeviceptr                       d_gas_output = 0;

        Aabb                              object_aabb;
    };


    SUTILAPI void addCamera  ( const Camera& camera            )    { m_cameras.push_back( camera );     }
    SUTILAPI void addInstance( std::shared_ptr<Instance> instance ) { m_instances.push_back( instance ); }
    SUTILAPI void addMesh    ( std::shared_ptr<MeshGroup> mesh )    { m_meshes.push_back( mesh );        }
    SUTILAPI void addMaterial( const MaterialData& mtl    )         { m_materials.push_back( mtl );      }
    SUTILAPI void addBuffer  ( const uint64_t buf_size, const void* data );
    SUTILAPI void addImage(
                const int32_t width,
                const int32_t height,
                const int32_t bits_per_component,
                const int32_t num_components,
                const void*   data
                );
    SUTILAPI void addSampler(
                cudaTextureAddressMode address_s,
                cudaTextureAddressMode address_t,
                cudaTextureFilterMode  filter_mode,
                const int32_t          image_idx
                );

    SUTILAPI CUdeviceptr                    getBuffer ( int32_t buffer_index  )const;
    SUTILAPI cudaArray_t                    getImage  ( int32_t image_index   )const;
    SUTILAPI cudaTextureObject_t            getSampler( int32_t sampler_index )const;

    SUTILAPI void                           finalize();
    SUTILAPI void                           cleanup();

    SUTILAPI Camera                                         camera()const;
    SUTILAPI OptixPipeline                                  pipeline()const           { return m_pipeline;   }
    SUTILAPI const OptixShaderBindingTable*                 sbt()const                { return &m_sbt;       }
    SUTILAPI OptixTraversableHandle                         traversableHandle() const { return m_ias_handle; }
    SUTILAPI sutil::Aabb                                    aabb() const              { return m_scene_aabb; }
    SUTILAPI OptixDeviceContext                             context() const           { return m_context;    }
    SUTILAPI const std::vector<MaterialData>&               materials() const         { return m_materials;  }
    SUTILAPI const std::vector<std::shared_ptr<MeshGroup>>& meshes() const            { return m_meshes;     }
    SUTILAPI const std::vector<std::shared_ptr<Instance>>&  instances() const         { return m_instances;  }

    SUTILAPI void createContext();
    SUTILAPI void buildMeshAccels();
    SUTILAPI void buildInstanceAccel( int rayTypeCount = whitted::RAY_TYPE_COUNT );

private:
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBT();

    // TODO: custom geometry support

    std::vector<Camera>                      m_cameras;
    std::vector<std::shared_ptr<Instance> >  m_instances;
    std::vector<std::shared_ptr<MeshGroup> > m_meshes;
    std::vector<MaterialData>                m_materials;
    std::vector<CUdeviceptr>                 m_buffers;
    std::vector<cudaTextureObject_t>         m_samplers;
    std::vector<cudaArray_t>                 m_images;
    sutil::Aabb                              m_scene_aabb;

    OptixDeviceContext                   m_context                  = 0;
    OptixShaderBindingTable              m_sbt                      = {};
    OptixPipelineCompileOptions          m_pipeline_compile_options = {};
    OptixPipeline                        m_pipeline                 = 0;
    OptixModule                          m_ptx_module               = 0;

    OptixProgramGroup                    m_raygen_prog_group        = 0;
    OptixProgramGroup                    m_radiance_miss_group      = 0;
    OptixProgramGroup                    m_occlusion_miss_group     = 0;
    OptixProgramGroup                    m_radiance_hit_group       = 0;
    OptixProgramGroup                    m_occlusion_hit_group      = 0;
    OptixTraversableHandle               m_ias_handle               = 0;
    CUdeviceptr                          m_d_ias_output_buffer      = 0;
};


SUTILAPI void loadScene( const std::string& filename, Scene& scene );

} // end namespace sutil

