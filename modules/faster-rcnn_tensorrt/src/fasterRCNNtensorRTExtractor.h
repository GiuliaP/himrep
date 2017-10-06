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

// other classes
#include "cudaUtility.h"
#include "PluginFactory.h"

// boost
#include "boost/algorithm/string.hpp"
//#include "boost/make_shared.hpp"

//using namespace cv;
//using namespace std;

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

class fasterRCNNtensorRTExtractor {

protected:
    
    bool cudaFreeMapped(void *cpuPtr);
    bool cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size );
    
    bool caffeToGIEModel( const std::string& deployFile,		   // name for caffe prototxt
			  const std::string& modelFile,	            			// name for model
           const std::string& binaryprotoFile,       				// name for .binaryproto
			  const std::vector<std::string>& outputs,  				// network outputs
			  unsigned int maxBatchSize,		        			      // batch size - NB must be at least as large as the batch we want to run with)
           nvcaffeparser1::IPluginFactory* pluginFactory,	   // factory for plugin layers
			  IHostMemory **gieModelStream);		    			      // output stream for the GIE model

    bool init(std::string _caffemodel_file, 
      std::string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
      std::string _prototxt_file, 
      int _resizeWidth, int _resizeHeight,
      std::string in_blob_names, std::string out_blob_names,
      int _batchSize,
      int _poolingH, int _poolingW, int _featuresStride, int _preNmsTop, int _nmsMaxOut, float _iouThreshold, float _minBoxSize, float _spatialScale, 
      float* _anchorsRatios, int _anchorsRatioCount, float* _anchorsScales, int _anchorsScaleCount, 
      float _nms_threshold, float _score_threshold,
      int _nClasses);

    nvinfer1::IRuntime* mInfer;
    nvinfer1::ICudaEngine* mEngine;
    nvinfer1::IExecutionContext* mContext;
    PluginFactory mPluginFactory;

    /////////////////////////
   
    int poolingH;
    int poolingW;
   
    int featuresStride;
    int preNmsTop;
    int nmsMaxOut;
   
    int anchorsRatioCount;
    int anchorsScaleCount;
   
    float iouThreshold;
    float minBoxSize;
    float spatialScale;
   
    float *anchorsRatios;
    float *anchorsScales;
   
    /////////////////////////
   
    float nms_threshold;
	 float score_threshold;

    ///////////////////////// 

    int indices[5];
    
    int inputC;
    int inputH;
    int inputW;
    
    int resizeHeight;
    int resizeWidth;
    
    int OUTPUT_CLS_SIZE;
    int OUTPUT_BBOX_SIZE;

    std::vector<std::string> BLOB_NAMES;
    
    /////////////////////////
    
    void *buffers[5];
    float *data;
    float *imInfo;
    float *rois;
    float *bboxPreds;
    float *clsProbs;
    
    float* predBBoxes;
    
    /////////////////////////
    
    std::vector<float> mean_values;
    float *meanData;
    
    ////////////////////////
    
    int batchSize;
   
    ////////////////////////

    Logger  gLogger;

    void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo, const int N, const int nmsMaxOut, const int numCls);
    
    std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold);
	
public:

   std::string prototxt_file;
   std::string caffemodel_file;
 
   std::string binaryproto_meanfile;

   bool timing;

   fasterRCNNtensorRTExtractor(std::string _caffemodel_file,
            std::string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
            std::string _prototxt_file, int _resizeWidth, int _resizeHeight,
            bool _timing,
            std::string _in_blob_names, std::string out_blob_names,
            int _batchSize,
            int _poolingH, int _poolingW, int _featuresStride, int _preNmsTop, int _nmsMaxOut, float _iouThreshold, float _minBoxSize, float _spatialScale, 
            float* _anchorsRatios, int _anchorsRatioCount, float* _anchorsScales, int _anchorsScaleCount, 
            float _nms_threshold, float _score_threshold,
            int nClasses);

    ~fasterRCNNtensorRTExtractor();

    bool extract(cv::Mat &image, std::vector<int> &detectionClasses, std::vector< std::vector<float> > &detectionScores, 
                std::vector< std::vector< std::vector<float> > > &detectionBoxes, float (&times)[2]);
    
    bool draw_detection(cv::Mat &image, cv::Mat &outImage);

};

#endif
