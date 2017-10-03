
#include "fasterRCNNtensorRTExtractor.h"

#include <sstream>

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_C = 3;
static const int INPUT_H = 375;
static const int INPUT_W = 500;
static const int IM_INFO_SIZE = 3;
static const int OUTPUT_CLS_SIZE = 21;
static const int OUTPUT_BBOX_SIZE = OUTPUT_CLS_SIZE * 4;

const std::string CLASSES[OUTPUT_CLS_SIZE]{ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };

const char* INPUT_BLOB_NAME0 = "data";
const char* INPUT_BLOB_NAME1 = "im_info";
const char* OUTPUT_BLOB_NAME0 = "bbox_pred";
const char* OUTPUT_BLOB_NAME1 = "cls_prob";
const char* OUTPUT_BLOB_NAME2 = "rois";

const int poolingH = 7;
const int poolingW = 7;
const int featureStride = 16;
const int preNmsTop = 6000;
const int nmsMaxOut = 300;
const int anchorsRatioCount = 3;
const int anchorsScaleCount = 3;
const float iouThreshold = 0.7f;
const float minBoxSize = 16;
const float spatialScale = 0.0625f;
const float anchorsRatios[anchorsRatioCount] = { 0.5f, 1.0f, 2.0f };
const float anchorsScales[anchorsScaleCount] = { 8.0f, 16.0f, 32.0f };

struct BBox
{
	float x1, y1, x2, y2;
};

template<int OutC>
class Reshape : public IPlugin
{
public:
	Reshape() {}
	Reshape(const void* buffer, size_t size)
	{
		assert(size == sizeof(mCopySize));
		mCopySize = *reinterpret_cast<const size_t*>(buffer);
	}

	int getNbOutputs() const override
	{
		return 1;
	}
	Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
	{
		assert(nbInputDims == 1);
		assert(index == 0);
		assert(inputs[index].nbDims == 3);
		assert((inputs[0].d[0])*(inputs[0].d[1]) % OutC == 0);
		return DimsCHW(OutC, inputs[0].d[0] * inputs[0].d[1] / OutC, inputs[0].d[2]);
	}

	int initialize() override
	{
		return 0;
	}

	void terminate() override
	{
	}

	size_t getWorkspaceSize(int) const override
	{
		return 0;
	}

	// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
	int enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) override
	{
		CHECK(cudaMemcpyAsync(outputs[0], inputs[0], mCopySize * batchSize, cudaMemcpyDeviceToDevice, stream));
		return 0;
	}

	
	size_t getSerializationSize() override
	{
		return sizeof(mCopySize);
	}

	void serialize(void* buffer) override
	{
		*reinterpret_cast<size_t*>(buffer) = mCopySize;
	}

	void configure(const Dims*inputs, int nbInputs, const Dims* outputs, int nbOutputs, int)	override
	{
		mCopySize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2] * sizeof(float);
	}

protected:
	size_t mCopySize;
};


// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>());
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>());
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			assert(nbWeights == 0 && weights == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>
				(createFasterRCNNPlugin(featureStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale,
					DimsHW(poolingH, poolingW), Weights{ nvinfer1::DataType::kFLOAT, anchorsRatios, anchorsRatioCount },
					Weights{ nvinfer1::DataType::kFLOAT, anchorsScales, anchorsScaleCount }), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
		assert(isPlugin(layerName));
		if (!strcmp(layerName, "ReshapeCTo2"))
		{
			assert(mPluginRshp2 == nullptr);
			mPluginRshp2 = std::unique_ptr<Reshape<2>>(new Reshape<2>(serialData, serialLength));
			return mPluginRshp2.get();
		}
		else if (!strcmp(layerName, "ReshapeCTo18"))
		{
			assert(mPluginRshp18 == nullptr);
			mPluginRshp18 = std::unique_ptr<Reshape<18>>(new Reshape<18>(serialData, serialLength));
			return mPluginRshp18.get();
		}
		else if (!strcmp(layerName, "RPROIFused"))
		{
			assert(mPluginRPROI == nullptr);
			mPluginRPROI = std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)>(createFasterRCNNPlugin(serialData, serialLength), nvPluginDeleter);
			return mPluginRPROI.get();
		}
		else
		{
			assert(0);
			return nullptr;
		}
	}

	// caffe parser plugin implementation
	bool isPlugin(const char* name) override
	{
		return (!strcmp(name, "ReshapeCTo2")
			|| !strcmp(name, "ReshapeCTo18")
			|| !strcmp(name, "RPROIFused"));
	}

	// the application has to destroy the plugin when it knows it's safe to do so
	void destroyPlugin()
	{
		mPluginRshp2.release();		mPluginRshp2 = nullptr;
		mPluginRshp18.release();	mPluginRshp18 = nullptr;
		mPluginRPROI.release();		mPluginRPROI = nullptr;
	}


	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
};

void bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
	const int N, const int nmsMaxOut, const int numCls)
{
	float width, height, ctr_x, ctr_y;
	float dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	float *deltas_offset, *predBBoxes_offset, *imInfo_offset;
	for (int i = 0; i < N * nmsMaxOut; ++i)
	{
		width = rois[i * 4 + 2] - rois[i * 4] + 1;
		height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		ctr_x = rois[i * 4] + 0.5f * width;
		ctr_y = rois[i * 4 + 1] + 0.5f * height;
		deltas_offset = deltas + i * numCls * 4;
		predBBoxes_offset = predBBoxes + i * numCls * 4;
		imInfo_offset = imInfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; ++j)
		{
			dx = deltas_offset[j * 4];
			dy = deltas_offset[j * 4 + 1];
			dw = deltas_offset[j * 4 + 2];
			dh = deltas_offset[j * 4 + 3];
			pred_ctr_x = dx * width + ctr_x;
			pred_ctr_y = dy * height + ctr_y;
			pred_w = exp(dw) * width;
			pred_h = exp(dh) * height;
			predBBoxes_offset[j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, imInfo_offset[1] - 1.f), 0.f);
			predBBoxes_offset[j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, imInfo_offset[0] - 1.f), 0.f);
		}
	}
}

std::vector<int> nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
{
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max) -> float {
		if (x1min > x2min) {
			std::swap(x1min, x2min);
			std::swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2) -> float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
		float overlap2D = overlapX * overlapY;
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};

	std::vector<int> indices;
	for (auto i : score_index)
	{
		const int idx = i.second;
		bool keep = true;
		for (unsigned k = 0; k < indices.size(); ++k)
		{
			if (keep)
			{
				const int kept_idx = indices[k];
				float overlap = computeIoU(&bbox[(idx*numClasses + classNum) * 4],
					&bbox[(kept_idx*numClasses + classNum) * 4]);
				keep = overlap <= nms_threshold;
			}
			else
				break;
		}
		if (keep) indices.push_back(idx);
	}
	return indices;
}


// Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
bool GIEFeatExtractor::cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
{
	if( !cpuPtr || !gpuPtr || size == 0 )
		return false;

	//CUDA(cudaSetDeviceFlags(cudaDeviceMapHost));

	if( CUDA_FAILED(cudaHostAlloc(cpuPtr, size, cudaHostAllocMapped)) )
		return false;

	if( CUDA_FAILED(cudaHostGetDevicePointer(gpuPtr, *cpuPtr, 0)) )
		return false;

	memset(*cpuPtr, 0, size);
	std::cout << "cudaAllocMapped : " << size << " bytes" << std::endl;
	return true;
}

bool GIEFeatExtractor::cudaFreeMapped(void *cpuPtr)
{
    if ( CUDA_FAILED( cudaFreeHost(cpuPtr) ) )
        return false;
    std::cout << "cudaFreeMapped: OK" << std::endl;
}

bool GIEFeatExtractor::caffeToGIEModel( const std::string& deployFile,			// name for .prototxt
					const std::string& modelFile,	                // name for .caffemodel
                                        const std::string& binaryprotoFile,             // name for .binaryproto
					const std::vector<std::string>& outputs,        // network outputs
					unsigned int maxBatchSize,			// batch size - NB must be at least as large as the batch we want to run with)
                                        nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
					IHostMemory **gieModelStream)		        // output stream for the GIE model
{
    
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);

    //builder->setMinFindIterations(3); // allow time for TX1 GPU to spin up
    //builder->setAverageFindIterations(2);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    parser->setPluginFactory(pluginFactory);

    const bool useFp16 = builder->platformHasFastFp16(); //getHalf2Mode();
    std::cout << "Platform FP16 support: " << useFp16 << std::endl;
    std::cout << "Loading: " << deployFile << ", " << modelFile << std::endl;

    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT; // create a 16-bit model if it's natively supported

    std::cout << "Begin parsing model..." << std::endl;
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, DataType::kFLOAT);
    std::cout << "End parsing model..." << std::endl;

    if( !blobNameToTensor )
    {
	std::cout << "Failed to parse caffe network." << std::endl;
	return false;
    }

    if (binaryprotoFile!="")
    {
        // Parse the mean image if it is needed

        IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(binaryprotoFile.c_str());
        resizeDims = meanBlob->getDimensions();

        const float *meanDataConst = reinterpret_cast<const float*>(meanBlob->getData());  // expected to be float* (c,h,w)
        
        meanData = (float *) malloc(resizeDims.w*resizeDims.h*resizeDims.c*resizeDims.n*sizeof(float));
        memcpy(meanData, meanDataConst, resizeDims.w*resizeDims.h*resizeDims.c*resizeDims.n*sizeof(float) );

        //cv::Mat tmpMat(resizeDims.h, resizeDims.w, CV_8UC3, meanDataChangeable);

        //cv::cvtColor(tmpMat, tmpMat, CV_RGB2BGR);
        //std::cout << "converted" << std::endl;

        //tmpMat.copyTo(meanMat);

        meanBlob->destroy();
        //free(meanDataChangeable);

    }

    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate
    const size_t num_outputs = outputs.size();
	
    for( size_t n=0; n < num_outputs; n++ )
	network->markOutput(*blobNameToTensor->find(outputs[n].c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(10 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5

    // set up the network for paired-fp16 format, only on DriveCX
    if (useFp16)
	builder->setHalf2Mode(true);

    std::cout << "Begin building engine..." << std::endl;
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);
    if( !engine )
    {
        std::cout << "Failed to build CUDA engine." << std::endl;
	return false;
    }
    std::cout << "End building engine..." << std::endl;

    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    (*gieModelStream) = engine->serialize();

    engine->destroy();
    builder->destroy();

    shutdownProtobufLibrary();

    return true;

}

GIEFeatExtractor::GIEFeatExtractor(string _caffemodel_file,
            string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
            string _prototxt_file, int _resizeWidth, int _resizeHeight,
            string _blob_name,
            bool _timing ) {

    mEngine  = NULL;
    mInfer   = NULL;
    mContext = NULL;

    resizeDims.n = -1;
    resizeDims.c = -1;
    resizeDims.w = -1;
    resizeDims.h = -1;

    mWidth     = 0;
    mHeight    = 0;
    mInputSize = 0;

    mInputCPU  = NULL;
    mInputCUDA = NULL;

    mOutputSize = 0;
    mOutputDims = 0;

    mOutputCPU     = NULL;
    mOutputCUDA    = NULL;

    prototxt_file = "";
    caffemodel_file = "";
    blob_name = "";
    binaryproto_meanfile = "";

    timing = false;
    

    if( !init(_caffemodel_file, _binaryproto_meanfile, _meanR, _meanG, _meanB, _prototxt_file, _resizeWidth, _resizeHeight) )
    {
        std::cout << "GIEFeatExtractor: init() failed." << std::endl;
    }

    // Initialize timing flag
    timing = _timing;

}

bool GIEFeatExtractor::init(string _caffemodel_file, string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, string _prototxt_file, int _resizeWidth, int _resizeHeight)
{

    cudaDeviceProp prop;
    int whichDevice;

    if ( CUDA_FAILED( cudaGetDevice(&whichDevice)) )
       return false;
 
    if ( CUDA_FAILED( cudaGetDeviceProperties(&prop, whichDevice)) )
       return false;

    if (prop.canMapHostMemory != 1)
    {
       std::cout << "Device cannot map memory!" << std::endl;
       return false;
    }

    //if ( CUDA_FAILED( cudaSetDeviceFlags(cudaDeviceMapHost)) )
    //    return false;

    // Assign specified .caffemodel, .binaryproto, .prototxt files     
    caffemodel_file  = _caffemodel_file;
    binaryproto_meanfile = _binaryproto_meanfile;
    
    mean_values.push_back(_meanB);
    mean_values.push_back(_meanG);
    mean_values.push_back(_meanR);

    prototxt_file = _prototxt_file;
    
    //Assign blob to be extracted
    blob_name = _blob_name;
    

    // create a GIE model from the caffe model and serialize it to a stream
    PluginFactory pluginFactory;
    IHostMemory *gieModelStream{ nullptr };
	
    if( !caffeToGIEModel( prototxt_file, caffemodel_file, binaryproto_meanfile, std::vector< std::string > {  OUTPUT_BLOB_NAME0, OUTPUT_BLOB_NAME1, OUTPUT_BLOB_NAME2 }, 1, &pluginFactory, &gieModelStream )
    {
       std::cout << "Failed to load: " << caffemodel_file << std::endl;
    }

    pluginFactory.destroyPlugin();

    std::cout << caffemodel_file << ": loaded." << std::endl;

    // Create runtime inference engine execution context
    IRuntime* infer = createInferRuntime(gLogger);
    if( !infer )
    {
        std::cout << "Failed to create InferRuntime." << std::endl;
    }
	
    ICudaEngine* engine = infer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &pluginFactory);
    if( !engine )
    {
	std::cout << "Failed to create CUDA engine." << std::endl;
    }
	
    IExecutionContext* context = engine->createExecutionContext();
    if( !context )
    {
	std::cout << "failed to create execution context." << std::endl;
    }

    std::cout << "CUDA engine context initialized with " << engine->getNbBindings() << " bindings." << std::endl;
	
    mInfer   = infer;
    mEngine  = engine;
    mContext = context;

    
	
    std::cout << caffemodel_file << ": initialized." << std::endl;

    if (binaryproto_meanfile=="")
    {
        // Set input size if the mean pixel is used
        resizeDims.h = _resizeHeight;
        resizeDims.w = _resizeWidth;
        resizeDims.c = 3;
        resizeDims.n = 1;
    }

    return true;

}

GIEFeatExtractor::~GIEFeatExtractor()
{
    if( mEngine != NULL )
    {
        mEngine->destroy();
        mEngine = NULL;
    }
		
    if( mInfer != NULL )
    {
        mInfer->destroy();
        mInfer = NULL;
    }

    cudaFreeMapped(mOutputCPU);
    cudaFreeMapped(mInputCPU);

    if (mean_values[0]==-1)
        free(meanData);
}

bool GIEFeatExtractor::extract(cv::Mat &imMat, float* outputBboxPred, float* outputClsProb, float *outputRois, float (&times)[2])
{

    times[0] = 0.0f;
    times[1] = 0.0f;


const ICudaEngine& engine = context.getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly 2 inputs and 3 outputs.
	assert(engine.getNbBindings() == 5);
	void* buffers[5];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex0 = engine.getBindingIndex(INPUT_BLOB_NAME0),
		inputIndex1 = engine.getBindingIndex(INPUT_BLOB_NAME1),
		outputIndex0 = engine.getBindingIndex(OUTPUT_BLOB_NAME0),
		outputIndex1 = engine.getBindingIndex(OUTPUT_BLOB_NAME1),
		outputIndex2 = engine.getBindingIndex(OUTPUT_BLOB_NAME2);


	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex0], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));   // data
	CHECK(cudaMalloc(&buffers[inputIndex1], batchSize * IM_INFO_SIZE * sizeof(float)));                  // im_info
	CHECK(cudaMalloc(&buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float))); // bbox_pred
	CHECK(cudaMalloc(&buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)));  // cls_prob
	CHECK(cudaMalloc(&buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float)));                // rois

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex0], inputData, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[inputIndex1], inputImInfo, batchSize * IM_INFO_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
	
        context.enqueue(batchSize, buffers, stream, nullptr);
	
        CHECK(cudaMemcpyAsync(outputBboxPred, buffers[outputIndex0], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputClsProb, buffers[outputIndex1], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputRois, buffers[outputIndex2], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream));
	
        cudaStreamSynchronize(stream);


	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex0]));
	CHECK(cudaFree(buffers[inputIndex1]));
	CHECK(cudaFree(buffers[outputIndex0]));
	CHECK(cudaFree(buffers[outputIndex1]));
	CHECK(cudaFree(buffers[outputIndex2]));
























    // Check input image 
    if (imMat.empty())
    {
        std::cout << "GIEFeatExtractor::extract_singleFeat_1D(): empty imMat!" << std::endl;
        return false;
    }

    // Start timing
    cudaEvent_t startPrep, stopPrep, startNet, stopNet;
    if (timing)
    {
        cudaEventCreate(&startPrep);
        cudaEventCreate(&startNet);
        cudaEventCreate(&stopPrep);
        cudaEventCreate(&stopNet);
        cudaEventRecord(startPrep, NULL);
        cudaEventRecord(startNet, NULL);
    }

    // Image preprocessing
 
    // resize (to 256x256 or to the size of the mean mean image)
    if (imMat.rows != resizeDims.h || imMat.cols != resizeDims.w)
    {
       if (imMat.rows > resizeDims.h || imMat.cols > resizeDims.w)
       {
           cv::resize(imMat, imMat, cv::Size(resizeDims.h, resizeDims.w), 0, 0, CV_INTER_LANCZOS4);
       }
       else
       {
           cv::resize(imMat, imMat, cv::Size(resizeDims.h, resizeDims.w), 0, 0, CV_INTER_LINEAR);
       }
    }

    // crop and subtract the mean image or the mean pixel
    int h_off = (imMat.rows - mHeight) / 2;
    int w_off = (imMat.cols - mWidth) / 2;

    cv::Mat cv_cropped_img = imMat;
    cv::Rect roi(w_off, h_off, mWidth, mHeight);
    cv_cropped_img = imMat(roi);

    int top_index;
    for (int h = 0; h < mHeight; ++h)
    {
       const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
       int img_index = 0;
       for (int w = 0; w < mWidth; ++w)
       {
           for (int c = 0; c < imMat.channels(); ++c)
           {
               top_index = (c * mHeight + h) * mWidth + w;
               float pixel = static_cast<float>(ptr[img_index++]);
               if (mean_values[0]==-1)
               {
                   int mean_index = (c * imMat.rows + h_off + h) * imMat.cols + w_off + w;
                   mInputCPU[top_index] = pixel - meanData[mean_index];
                }
                else
                {
                    mInputCPU[top_index] = pixel - mean_values[c]; 
                }
            }
         }
      }

    /*

    // subtract mean 
    if (meanR==-1)
    {
        if (!meanMat.empty() && imMat.rows==meanMat.rows && imMat.cols==meanMat.cols && imMat.channels()==meanMat.channels() && imMat.type()==meanMat.type())
        {
            imMat = imMat - meanMat;
        }
        else
        {
            std::cout << "GIEFeatExtractor::extract_singleFeat_1D(): cannot subtract mean image!" << std::endl;
            return false;
        }
    }
    else
    {  
        imMat = imMat - cv::Scalar(meanB, meanG, meanR);
    }

    // crop to input dimension (central crop)
    if (imMat.cols>=mWidth && imMat.rows>=mHeight)
    {
        cv::Rect imROI(floor((imMat.cols-mWidth)*0.5f), floor((imMat.rows-mHeight)*0.5f), mWidth, mHeight);
        imMat(imROI).copyTo(imMat);
    } 
        else
    {
        cv::resize(imMat, imMat, cv::Size(mHeight, mWidth), 0, 0, CV_INTER_LINEAR);
    }

    // convert to float (with range 0-255)
    imMat.convertTo(imMat, CV_32FC3);

    if ( !imMat.isContinuous() ) 
          imMat = imMat.clone();*/

    // copy 
    //CUDA( cudaMemcpy(mInputCPU, imMat.data, mInputSize, cudaMemcpyDefault) );
    //memcpy(mInputCPU, imMat.data, mInputSize);

    void* inferenceBuffers[] = { mInputCUDA, mOutputCUDA };

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopPrep, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopPrep);

        cudaEventElapsedTime(times, startPrep, stopPrep);

    }

     mContext->execute(1, inferenceBuffers);
     //CUDA(cudaDeviceSynchronize());\

    features.insert(features.end(), &mOutputCPU[0], &mOutputCPU[mOutputDims]);

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopNet, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopNet);

        cudaEventElapsedTime(times+1, startNet, stopNet);

    }

    return true;

}

