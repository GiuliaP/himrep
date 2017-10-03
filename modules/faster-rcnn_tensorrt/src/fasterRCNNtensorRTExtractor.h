#ifndef fasterRCNNtensorRTExtractorEXTRACTOR_H_
#define fasterRCNNtensorRTExtractorEXTRACTOR_H_

#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include <sys/stat.h>
#include <time.h>
#include <memory>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// OpenCV
#include <opencv2/opencv.hpp>

// CUDA-C includes
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>

// GIE
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvCaffeParser.h"
#include "cudaUtility.h"

using namespace cv;

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// Logger for tensorRT info/warning/errors
class Logger : public nvinfer1::ILogger			
{
    void log( Severity severity, const char* msg ) override
    {
       if ( severity != Severity::kINFO )
       {
         switch (severity)
         {
            case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
            case Severity::kERROR: std::cerr << "ERROR: "; break;
            case Severity::kWARNING: std::cerr << "WARNING: "; break;
            case Severity::kINFO: std::cerr << "INFO: "; break;
            default: std::cerr << "UNKNOWN: "; break;
         }
         std::cerr << msg << std::endl;
       }
    }
};

struct BBox
{
	float x1, y1, x2, y2;
};


class GIEFeatExtractor {

protected:
    
    bool cudaFreeMapped(void *cpuPtr);

    bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size );

    bool caffeToGIEModel( const std::string& deployFile,		    			// name for caffe prototxt
			  const std::string& modelFile,	            				// name for model
                          const std::string& binaryprotoFile,       				// name for .binaryproto
			  const std::vector<std::string>& outputs,  				// network outputs
			  unsigned int maxBatchSize,		        			// batch size - NB must be at least as large as the batch we want to run with)
                          nvcaffeparser1::IPluginFactory* pluginFactory,	                // factory for plugin layers
			  IHostMemory **gieModelStream);		    			// output stream for the GIE model

    bool init(string _caffemodel_file,
            string _binaryproto_meanfile, float meanR, float meanG, float meanB, 
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_name);

    nvinfer1::IRuntime* mInfer;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* mContext;

    //cv::Mat meanMat;
    float *meanData;
    vector<float> mean_values;

    nvinfer1::Dims4 resizeDims;

    uint32_t mWidth;
    uint32_t mHeight;
    uint32_t mInputSize;
    float*   mInputCPU;
    float*   mInputCUDA;
	
    uint32_t mOutputSize;
    uint32_t mOutputDims;
    float*   mOutputCPU;
    float*   mOutputCUDA;

    Logger  gLogger;

public:

    string prototxt_file;
    string caffemodel_file;
    string blob_name;
    string binaryproto_meanfile;

    bool timing;
    
    fasterRCNNtensorRTExtractor(string _caffemodel_file,
            string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_name,
            bool _timing );

    ~fasterRCNNtensorRTExtractor();

    bool extract(cv::Mat &image, float* outputBboxPred, float* outputClsProb, float *outputRois, float (&times)[2]);

};

#endif
