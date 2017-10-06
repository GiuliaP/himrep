#include "Reshape.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{

   int featureStride;
   int preNmsTop;
   int nmsMaxOut;
   
	float iouThreshold;
   float minBoxSize;
   float spatialScale;
	
   int poolingH;
   int poolingW;

   int anchorsRatioCount;
   int anchorsScaleCount;
   
   float* anchorsRatios;
   float* anchorsScales;
   
public:

	// deserialization plugin implementation
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{
	   
	   std::cout << "one " << std::endl;
	   
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
		std::cout << "two " << std::endl;
		
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
		
		delete[] anchorsRatios;
		delete[] anchorsScales;
		
	}

   void assignParameters(int _featureStride, int _preNmsTop, int _nmsMaxOut, float _iouThreshold, float _minBoxSize, float _spatialScale, int _poolingH, int _poolingW, float *_anchorsRatios, int _anchorsRatioCount, float *_anchorsScales, int _anchorsScaleCount)
   {
      featureStride = _featureStride;
      std::cout << "featureStride " << featureStride << std::endl;
      
      preNmsTop = _preNmsTop;
      std::cout << "preNmsTop " << preNmsTop << std::endl;
      
      nmsMaxOut = _nmsMaxOut;
      std::cout << "nmsMaxOut " << nmsMaxOut << std::endl;
    
      iouThreshold = _iouThreshold;
      std::cout << "iouThreshold " << iouThreshold << std::endl;
      
      minBoxSize = _minBoxSize;
      std::cout << "minBoxSize " << minBoxSize << std::endl;
      
      spatialScale = _spatialScale;
      std::cout << "spatialScale " << spatialScale << std::endl;
    
      poolingH = _poolingH;
      std::cout << "poolingH " << poolingH << std::endl;
      poolingW = _poolingW;
      std::cout << "poolingW " << poolingW << std::endl;
    
      anchorsRatioCount = _anchorsRatioCount;
      std::cout << "anchorsRatioCount " << anchorsRatioCount << std::endl;
      anchorsScaleCount = _anchorsScaleCount;
      std::cout << "anchorsScaleCount " << anchorsScaleCount << std::endl;
    
      anchorsRatios = new float[_anchorsRatioCount];
      anchorsRatios[0] = _anchorsRatios[0];
      std::cout << "anchorsRatios[0] " << anchorsRatios[0] << std::endl;
      anchorsRatios[1] = _anchorsRatios[1];
      std::cout << "anchorsRatios[1] " << anchorsRatios[1] << std::endl;
      anchorsRatios[2] = _anchorsRatios[2];
      std::cout << "anchorsRatios[2] " << anchorsRatios[2] << std::endl;
    
      anchorsScales = new float[_anchorsScaleCount];
      anchorsScales[0] = _anchorsScales[0];
      std::cout << "anchorsScales[0] " << anchorsScales[0] << std::endl;
      anchorsScales[1] = _anchorsScales[1];
      std::cout << "anchorsScales[1] " << anchorsScales[1] << std::endl;
      anchorsScales[2] = _anchorsScales[2];
      std::cout << "anchorsScales[2] " << anchorsScales[2] << std::endl;
    
   }
   
	std::unique_ptr<Reshape<2>> mPluginRshp2{ nullptr };
	std::unique_ptr<Reshape<18>> mPluginRshp18{ nullptr };
	void(*nvPluginDeleter)(INvPlugin*) { [](INvPlugin* ptr) {ptr->destroy(); } };
	std::unique_ptr<INvPlugin, decltype(nvPluginDeleter)> mPluginRPROI{ nullptr, nvPluginDeleter };
   			
};
