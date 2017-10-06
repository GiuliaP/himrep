
#include "fasterRCNNtensorRTExtractor.h"

std::vector<int> fasterRCNNtensorRTExtractor::nms(std::vector<std::pair<float, int> >& score_index, float* bbox, const int classNum, const int numClasses, const float nms_threshold)
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


void fasterRCNNtensorRTExtractor::bboxTransformInvAndClip(float* rois, float* deltas, float* predBBoxes, float* imInfo,
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


// Allocate ZeroCopy mapped memory, shared between CUDA and CPU.
bool fasterRCNNtensorRTExtractor::cudaAllocMapped( void** cpuPtr, void** gpuPtr, size_t size )
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

bool fasterRCNNtensorRTExtractor::cudaFreeMapped(void *cpuPtr)
{
    if ( CUDA_FAILED( cudaFreeHost(cpuPtr) ) )
        return false;
    std::cout << "cudaFreeMapped: OK" << std::endl;
}

bool fasterRCNNtensorRTExtractor::caffeToGIEModel( const std::string& deployFile,			         // name for .prototxt
					                                    const std::string& modelFile,	                  // name for .caffemodel
                                                   const std::string& binaryprotoFile,             // name for .binaryproto
					                                    const std::vector<std::string>& outputs,        // network outputs
					                                    unsigned int maxBatchSize,			               // batch size - NB must be at least as large as the batch we want to run with)
                                                   nvcaffeparser1::IPluginFactory* pluginFactory,	// factory for plugin layers
					                                    IHostMemory **gieModelStream)		               // output stream for the GIE model
{
    
    // create API root class - must span the lifetime of the engine usage
    IBuilder* builder = createInferBuilder(gLogger);

    //builder->setMinFindIterations(3); // allow time for TX1 GPU to spin up
    //builder->setAverageFindIterations(2);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    parser->setPluginFactory(pluginFactory);

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
        
        nvinfer1::Dims resizeDims = meanBlob->getDimensions();
        resizeHeight = resizeDims.d[1];
        resizeWidth = resizeDims.d[0];

        const float *meanDataConst = reinterpret_cast<const float*>(meanBlob->getData());  // expected to be float* (c,h,w)
        
        meanData = (float *) malloc(resizeDims.d[0]*resizeDims.d[1]*resizeDims.d[2]*resizeDims.d[3]*sizeof(float));
        memcpy(meanData, meanDataConst, resizeDims.d[0]*resizeDims.d[1]*resizeDims.d[2]*resizeDims.d[3]*sizeof(float) );

        //cv::Mat tmpMat(resizeDims.h, resizeDims.w, CV_8UC3, meanDataChangeable);

        //cv::cvtColor(tmpMat, tmpMat, CV_RGB2BGR);
        //std::cout << "converted" << std::endl;

        //tmpMat.copyTo(meanMat);

        meanBlob->destroy();
        //free(meanDataChangeable);

    }

    for (auto& s : outputs)
		network->markOutput(*blobNameToTensor->find(s.c_str()));
		
    // Build the engine
    
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(10 << 20);	// we need about 6MB of scratch space for the plugin layer for batch size 5

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

fasterRCNNtensorRTExtractor::fasterRCNNtensorRTExtractor(std::string _caffemodel_file,
            std::string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
            std::string _prototxt_file, int _resizeWidth, int _resizeHeight,
            bool _timing,
            std::string _in_blob_names, std::string _out_blob_names,
            int _batchSize,
            int _poolingH, int _poolingW, int _featuresStride, int _preNmsTop, int _nmsMaxOut, float _iouThreshold, float _minBoxSize, float _spatialScale, 
            float* _anchorsRatios, int _anchorsRatioCount, float* _anchorsScales, int _anchorsScaleCount, 
            float _nms_threshold, float _score_threshold,
            int _nClasses) {

    mEngine  = NULL;
    mInfer   = NULL;
    mContext = NULL;

    resizeWidth = -1;
    resizeHeight = -1;

    //buffers  = NULL;
    
    predBBoxes = NULL;
    
    prototxt_file = "";
    caffemodel_file = "";
    binaryproto_meanfile = "";

    timing = _timing;

    if( !init(_caffemodel_file, 
               _binaryproto_meanfile, _meanR, _meanG, _meanB, 
               _prototxt_file, _resizeWidth, _resizeHeight, 
               _in_blob_names, _out_blob_names, 
               _batchSize, 
               _poolingH, _poolingW, _featuresStride, _preNmsTop, _nmsMaxOut, _iouThreshold, _minBoxSize, _spatialScale, 
               _anchorsRatios, _anchorsRatioCount, _anchorsScales, _anchorsScaleCount, 
               _nms_threshold, _score_threshold,
               _nClasses) )
    {
        std::cout << "GIEFeatExtractor: init() failed." << std::endl;
    }

}

bool fasterRCNNtensorRTExtractor::init(std::string _caffemodel_file, 
      std::string _binaryproto_meanfile, float _meanR, float _meanG, float _meanB, 
      std::string _prototxt_file, 
      int _resizeWidth, int _resizeHeight,
      std::string in_blob_names, std::string out_blob_names,
      int _batchSize,
      int _poolingH, int _poolingW, int _featuresStride, int _preNmsTop, int _nmsMaxOut, float _iouThreshold, float _minBoxSize, float _spatialScale, 
      float* _anchorsRatios, int _anchorsRatioCount, float* _anchorsScales, int _anchorsScaleCount, 
      float _nms_threshold, float _score_threshold,
      int _nClasses)
{

    cudaDeviceProp prop;
    int whichDevice;

    if ( CUDA_FAILED( cudaGetDevice(&whichDevice)) )
       return false;
 
    if ( CUDA_FAILED( cudaGetDeviceProperties(&prop, whichDevice)) )
       return false;

    // Assign specified .caffemodel, .binaryproto, .prototxt files     
    caffemodel_file  = _caffemodel_file;
    binaryproto_meanfile = _binaryproto_meanfile;
    
    mean_values.push_back(_meanB);
    mean_values.push_back(_meanG);
    mean_values.push_back(_meanR);

    prototxt_file = _prototxt_file;

    //
    
    batchSize = _batchSize;
    
    std::vector<std::string> tmp;
    boost::split(BLOB_NAMES, in_blob_names, boost::is_any_of(","));
    boost::split(tmp, out_blob_names, boost::is_any_of(","));
    
    BLOB_NAMES.insert(BLOB_NAMES.end(), tmp.begin(), tmp.end());
    
    for (int i=0; i<BLOB_NAMES.size(); i++) 
    {
      std::cout << BLOB_NAMES[i] << std::endl;
    
    }
    //
    
    featuresStride = _featuresStride;
    preNmsTop = _preNmsTop;
    nmsMaxOut = _nmsMaxOut;
    
    iouThreshold = _iouThreshold;
    minBoxSize = _minBoxSize;
    spatialScale = _spatialScale;
    
    poolingH = _poolingH;
    poolingW = _poolingW;
    
    anchorsRatioCount = _anchorsRatioCount;
    anchorsScaleCount = _anchorsScaleCount;
    
    anchorsRatios = new float[_anchorsRatioCount];
    anchorsRatios[0] = _anchorsRatios[0];
    anchorsRatios[1] = _anchorsRatios[1];
    anchorsRatios[2] = _anchorsRatios[2];
    
    anchorsScales = new float[_anchorsScaleCount];
    anchorsScales[0] = _anchorsScales[0];
    anchorsScales[1] = _anchorsScales[1];
    anchorsScales[2] = _anchorsScales[2];

    nms_threshold = _nms_threshold;
	 score_threshold = _score_threshold;
	
    /////////////////////////

    // create a GIE model from the caffe model and serialize it to a stream
    mPluginFactory.assignParameters(featuresStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale, poolingH, poolingW, 
                                   anchorsRatios, anchorsRatioCount, anchorsScales, anchorsScaleCount);
    
    IHostMemory *gieModelStream{ nullptr };
	
    if( !caffeToGIEModel( prototxt_file, caffemodel_file, binaryproto_meanfile, std::vector< std::string > {  BLOB_NAMES[2], BLOB_NAMES[3], BLOB_NAMES[4] }, batchSize, &mPluginFactory, &gieModelStream ) )
    {
       return false;
    }

    mPluginFactory.destroyPlugin();

    std::cout << caffemodel_file << ": loaded." << std::endl;

    // Create runtime inference engine execution context
    mInfer = createInferRuntime(gLogger);
    if( !mInfer )
    {
        std::cout << "Failed to create InferRuntime." << std::endl;
        return false;
    }
	
    mEngine = mInfer->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), &mPluginFactory);
    if( !mEngine )
    {
	 std::cout << "Failed to create CUDA engine." << std::endl;
    }
	
    mContext = mEngine->createExecutionContext();
    if( !mContext )
    {
	 std::cout << "failed to create execution context." << std::endl;
    }

    std::cout << "CUDA engine context initialized with " << mEngine->getNbBindings() << " bindings." << std::endl;

    ///////////////////////////////
    
    OUTPUT_CLS_SIZE = _nClasses;
    OUTPUT_BBOX_SIZE = _nClasses * 4;
    
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	 // of these, but in this case we know that there is exactly 2 inputs and 3 outputs.
	 assert(mEngine->getNbBindings() == 5);
	 
	 // in order to bind the buffers, we need to know the names of the input and output tensors
	 // note that indices are guaranteed to be less than IEngine::getNbBindings()
	 for (int i=0; i<5; i++)
	 {
	   indices[i] = mEngine->getBindingIndex(BLOB_NAMES[i].c_str());
	   std::cout << indices[i] << std::endl;
	 }
	 
    nvinfer1::Dims d = mEngine->getBindingDimensions(indices[0]);
    
    inputW = d.d[2];
    inputH = d.d[1];
    inputC = d.d[0];

    std::cout << "C: " << inputC << " H: " << inputH << " W: " << inputW<< std::endl;
    
    if (binaryproto_meanfile=="")
    {
       // Set input size if the mean pixel is used
       resizeHeight = _resizeHeight;
       resizeWidth = _resizeWidth;
    }
    
   // host memory for outputs 
   data = new float[batchSize*inputC*inputH*inputW];
   imInfo = new float[batchSize * 3];	
	bboxPreds = new float[batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE];
	clsProbs = new float[batchSize * nmsMaxOut * OUTPUT_CLS_SIZE];
	rois = new float[batchSize * nmsMaxOut * 4];

	 if (CUDA_FAILED(cudaMalloc(&buffers[indices[0]], batchSize * inputC * inputH * inputW * sizeof(float)) ) ) 
    {
       return false;
    }
       
    if (CUDA_FAILED(cudaMalloc(&buffers[indices[1]], batchSize * 3 * sizeof(float)) ) )
    {
       return false;
    } 
      
    if (CUDA_FAILED(cudaMalloc(&buffers[indices[2]], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float)) ) )
    {
       return false;
    } 
    
    if (CUDA_FAILED(cudaMalloc(&buffers[indices[3]], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float)) ) )
    {
       return false;
    } 
    
    if (CUDA_FAILED(cudaMalloc(&buffers[indices[4]], batchSize * nmsMaxOut * 4 * sizeof(float)) ) )
    {
       return false;
    } 
      
    predBBoxes = new float[nmsMaxOut * OUTPUT_BBOX_SIZE];

   //////////////////////////////

   std::cout << caffemodel_file << ": initialized." << std::endl;

   return true;

}

fasterRCNNtensorRTExtractor::~fasterRCNNtensorRTExtractor()
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

   mPluginFactory.destroyPlugin();
   
   delete[] data;
   delete[] imInfo;
   delete[] bboxPreds;
   delete[] clsProbs;
   delete[] rois;
	
   if (CUDA_FAILED(cudaFree(buffers[indices[0]])) )
      std::cout << " failed CUDA deallocation "<< std::endl;
      
	if (CUDA_FAILED(cudaFree(buffers[indices[1]])) )
	   std::cout << " failed CUDA deallocation "<< std::endl;
	   
	if (CUDA_FAILED(cudaFree(buffers[indices[2]])) )
	   std::cout << " failed CUDA deallocation "<< std::endl;
	   
	if (CUDA_FAILED(cudaFree(buffers[indices[3]])) )
	   std::cout << " failed CUDA deallocation "<< std::endl;
	   
	if (CUDA_FAILED(cudaFree(buffers[indices[4]])) )
	   std::cout << " failed CUDA deallocation "<< std::endl;

    delete[] predBBoxes;
    
    if (mean_values[0]==-1)
        free(meanData);
        
    delete[] anchorsRatios;
    delete[] anchorsScales;
}

bool fasterRCNNtensorRTExtractor::extract(cv::Mat &imMat, std::vector<int> &detectionClasses, std::vector< std::vector<float> > &detectionScores, std::vector< std::vector< std::vector<float> > > &detectionBoxes, float (&times)[2])
{

    times[0] = 0.0f;
    times[1] = 0.0f;

    // Check input image 
    if (imMat.empty())
    {
        std::cout << "fasterRCNNtensorRTExtractor::extract(): empty imMat!" << std::endl;
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
 
    for (int i = 0; i < batchSize; ++i)
	 {
		
		imInfo[i * 3] = float(imMat.rows);                 // number of rows
		imInfo[i * 3 + 1] = float(imMat.cols);             // number of columns
		imInfo[i * 3 + 2] = float(inputH) / float(imMat.rows);   // image scale
	}
	
	
	//std::cout << imMat.rows << " - " << imMat.cols << std::endl;
	//std::cout << resizeHeight << " - " << resizeWidth << std::endl;
	
    // resize (to ... or to the size of the mean image)
    if (imMat.rows != resizeHeight || imMat.cols != resizeWidth)
    {
       if (imMat.rows > resizeHeight || imMat.cols > resizeWidth)
       {
           cv::resize(imMat, imMat, cv::Size(resizeWidth, resizeHeight), 0, 0, CV_INTER_LANCZOS4);
       }
       else
       {
           cv::resize(imMat, imMat, cv::Size(resizeWidth, resizeHeight), 0, 0, CV_INTER_LINEAR);
       }
    }

    // crop and subtract the mean image or the mean pixel
    int h_off = (imMat.rows - inputH) / 2;
    int w_off = (imMat.cols - inputW) / 2;

    //std::cout << h_off << " - " << w_off << std::endl;
    //std::cout << inputH << " - " << inputW << std::endl;
    
    cv::Mat cv_cropped_img = imMat;
    cv::Rect roi(w_off, h_off, inputW, inputH);
    cv_cropped_img = imMat(roi);

    int top_index;
    for (int h = 0; h < inputH; ++h)
    {
       const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
       int img_index = 0;
       for (int w = 0; w < inputW; ++w)
       {
           for (int c = 0; c < imMat.channels(); ++c)
           {
               top_index = (c * inputH + h) * inputW + w;
               float pixel = static_cast<float>(ptr[img_index++]);
               if (mean_values[0]==-1)
               {
                   int mean_index = (c * imMat.rows + h_off + h) * imMat.cols + w_off + w;
                   data[top_index] = pixel - meanData[mean_index];
                }
                else
                {
                    data[top_index] = pixel - mean_values[c]; 
                }
            }
         }
      }
 
    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopPrep, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopPrep);

        cudaEventElapsedTime(times, startPrep, stopPrep);

    }

	cudaStream_t stream;
	if (CUDA_FAILED(cudaStreamCreate(&stream)))
	   return false;

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	if (CUDA_FAILED(cudaMemcpyAsync(buffers[indices[0]], data, batchSize *inputC * inputH * inputW * sizeof(float), cudaMemcpyHostToDevice, stream)) )
	   return false;
	   
	if (CUDA_FAILED(cudaMemcpyAsync(buffers[indices[1]], imInfo, batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice, stream)) )
	   return false;
	
	mContext->enqueue(batchSize, buffers, stream, nullptr);
	
	if (CUDA_FAILED(cudaMemcpyAsync(bboxPreds, buffers[indices[2]], batchSize * nmsMaxOut * OUTPUT_BBOX_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream)) )
	   return false;
	   
	if (CUDA_FAILED(cudaMemcpyAsync(clsProbs, buffers[indices[3]], batchSize * nmsMaxOut * OUTPUT_CLS_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream)) )
	   return false;
	   
	if (CUDA_FAILED(cudaMemcpyAsync(rois, buffers[indices[4]], batchSize * nmsMaxOut * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream)) )
	   return false;
	
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);

    if (timing)
    {
        // Record the stop event
        cudaEventRecord(stopNet, NULL);

        // Wait for the stop event to complete
        cudaEventSynchronize(stopNet);

        cudaEventElapsedTime(times+1, startNet, stopNet);

    }
    
   // unscale back to raw image space
	for (int j=0; j<nmsMaxOut*4 && imInfo[2]!=1; ++j)
	    rois[j] /= imInfo[2];

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, batchSize, nmsMaxOut, OUTPUT_CLS_SIZE);

   for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
	{
	   
		std::vector<std::pair<float, int> > score_index;
		for (int r = 0; r < nmsMaxOut; ++r)
		{
			if (clsProbs[r*OUTPUT_CLS_SIZE + c] > score_threshold)
			{
				score_index.push_back(std::make_pair(clsProbs[r*OUTPUT_CLS_SIZE + c], r));
				std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
			}
		}

		// apply NMS algorithm
		std::vector<int> indices = nms(score_index, predBBoxes, c, OUTPUT_CLS_SIZE, nms_threshold);
			
		// Show results

		if (!indices.empty()) 
	   {
	      std::vector< std::vector<float> > detectionBoxes_c;
		   std::vector<float> detectionScores_c;
	   
		   for (unsigned k = 0; k < indices.size(); ++k)
		   {
			   int idx = indices[k];
		
		      float score = clsProbs[idx*OUTPUT_CLS_SIZE + c];
	
			   std::vector<float> box;
			   box.push_back(predBBoxes[idx*OUTPUT_BBOX_SIZE + c * 4]);
			   box.push_back(predBBoxes[idx*OUTPUT_BBOX_SIZE + c * 4 + 1]);
			   box.push_back(predBBoxes[idx*OUTPUT_BBOX_SIZE + c * 4 + 2]);
			   box.push_back(predBBoxes[idx*OUTPUT_BBOX_SIZE + c * 4 + 3]);
			
			   detectionScores_c.push_back(score);
			   detectionBoxes_c.push_back(box);

	      }
	   
	      detectionClasses.push_back(c);
	      detectionScores.push_back(detectionScores_c);
	      detectionBoxes.push_back(detectionBoxes_c);
	         
	      std::cout << detectionClasses[c] << std::endl;
	      std::cout << detectionScores_c.size() << std::endl;
         std::cout << detectionBoxes_c.size() << std::endl;
         
      }

	}

    return true;

}

bool fasterRCNNtensorRTExtractor::draw_detection(cv::Mat &image, cv::Mat &outImage)
{
   return true;
}
