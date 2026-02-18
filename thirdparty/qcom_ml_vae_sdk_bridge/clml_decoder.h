// Copyright (c) 2023-2024 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//******************************************************************************************************************************

/**
********************************************************************************************************************************
* @file
*     clml_decoder.h
* @brief
*     Header file for Image Decoder model class definition.
********************************************************************************************************************************
*/

#ifndef DECODER_H
#define DECODER_H

#include <memory>
#include <vector>

#include "nn_framework/clml_base_nn.h"

struct CpuAttentionHostOpData;

class Decoder:  public CLMLBaseNN
{
    public:
        Decoder(
            const NNModelDesc&                          desc,
            const CLEnvironment&                        cl_env);

        std::vector<MLTensor> create(
            const std::vector<MLTensor>&                inputs,
            const std::vector<MLTensor>&                targets = {});

        ~Decoder();

    private:
        Decoder();
        Decoder(const Decoder&);
        Decoder &operator=(const Decoder&);

        // Model descriptor
        NNModelDesc                                      m_desc;
        std::vector<std::unique_ptr<CpuAttentionHostOpData>> m_host_attn_ops;

        MLTensor createMidBlockOps(
            const MLTensor&                              input,
            cl_uint                                      channels,
            cl_uint                                      num_blocks,
            float                                        eps,
            const cl_activation_function_qcom&           act_fxn,
            cl_uint                                      norm_groups,
            cl_uint                                      num_attn_heads,
            const std::string&                           param_filename_prefix,
            bool                                         defer_output_mem = true);

        MLTensor createUpBlockOps(
            const MLTensor&                              input,
            cl_uint                                      in_channels,
            cl_uint                                      out_channels,
            cl_uint                                      num_blocks,
            float                                        eps,
            const cl_activation_function_qcom&           act_fxn,
            cl_uint                                      norm_groups,
            bool                                         add_up_sample,
            const std::string&                           param_filename_prefix,
            bool                                         defer_output_mem = true);

        MLTensor createResNetBlockOps(
            const MLTensor&                              input,
            cl_uint                                      in_channels,
            cl_uint                                      out_channels,
            float                                        eps,
            const cl_activation_function_qcom&           act_fxn,
            cl_uint                                      norm_groups,
            const std::string&                           param_filename_prefix,
            bool                                         defer_output_mem = true);

        MLTensor createAttentionBlockOps(
            const MLTensor&                              input,
            cl_uint                                      num_attn_heads,
            cl_uint                                      channels,
            float                                        eps,
            cl_uint                                      norm_groups,
            const std::string&                           param_filename_prefix,
            bool                                         defer_output_mem = true);

        MLTensor createUpSampleOp(
            const MLTensor&                              input,
            cl_uint                                      in_channels,
            cl_uint                                      out_channels,
            const std::string&                           param_filename_prefix,
            bool                                         defer_output_mem = true);

        void tearDown();

};

#endif ///> DECODER_H
