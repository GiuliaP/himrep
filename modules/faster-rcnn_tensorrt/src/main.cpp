/*
 * Copyright (C) 2017 iCub Facility - Istituto Italiano di Tecnologia
 * Author: Giulia Pasquale
 * email:  giulia.pasquale@iit.it
 * Permission is granted to copy, distribute, and/or modify this program
 * under the terms of the GNU General Public License, version 2 or any
 * later version published by the Free Software Foundation.
 *
 * A copy of the license can be found at
 * http://www.robotcub.org/icub/license/gpl.txt
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details
 */

// General includes
#include <cstdio>
#include <cstdlib> // getenv
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

// OpenCV
#include <opencv2/opencv.hpp>

#include <yarp/os/Network.h>
#include <yarp/os/RFModule.h>
#include <yarp/os/Time.h>
#include <yarp/os/BufferedPort.h>
#include <yarp/os/Semaphore.h>
#include <yarp/os/RpcClient.h>
#include <yarp/os/PortReport.h>
#include <yarp/os/Stamp.h>

#include <yarp/sig/Vector.h>
#include <yarp/sig/Image.h>

#include <yarp/math/Math.h>

#include "fasterRCNNtensorRTExtractor.h"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;

#define CMD_HELP                    VOCAB4('h','e','l','p')
#define DUMP_CODE                   VOCAB4('d','u','m','p')
#define DUMP_STOP                   VOCAB4('s','t','o','p')

class fasterRCNNtensorRTPort: public BufferedPort<Image>
{
private:

    // Resource Finder and module options

    ResourceFinder                &rf;

    string                        contextPath;

    bool                          dump_code;

    double                        rate;
    double                        last_read;

    // Data (common to all methods)

    cv::Mat                       matImg;

    Port                          port_out_img;
    Port                          port_out_code;

    FILE                          *fout_code;

    Semaphore                     mutex;

    // Data (specific for each method - instantiate only those are needed)

    fasterRCNNtensorRTExtractor              *rt_extractor;

    void onRead(Image &img)
    {

    	// Read at specified rate
        if (Time::now() - last_read < rate)
            return;

        mutex.wait();

        // If something arrived...
        if (img.width()>0 && img.height()>0)
        {

            // Convert the image and check that it is continuous

            cv::Mat tmp_mat = cv::cvarrToMat((IplImage*)img.getIplImage());
            cv::cvtColor(tmp_mat, matImg, CV_RGB2BGR);

	    // input image
            float* data = new float[INPUT_C*INPUT_H*INPUT_W];

            // outputs
            float* rois = new float[nmsMaxOut * 4];
            float* bboxPreds = new float[nmsMaxOut * OUTPUT_BBOX_SIZE];
            float* clsProbs = new float[nmsMaxOut * OUTPUT_CLS_SIZE];

            // predicted bounding boxes
            float* predBBoxes = new float[nmsMaxOut * OUTPUT_BBOX_SIZE];

            // Extract the feature vector
            float times[2];
            if (!rt_extractor->extract(data, imInfo, bboxPreds, clsProbs, rois, times))
            {
                std::cout << "fasterRCNNtensorRTExtractor::extract(): failed..." << std::endl;
                return;
            }

            if (rt_extractor->timing)
            {
                std::cout << times[0] << ": PREP " << times[1] << ": NET" << std::endl;
            }

           // unscale back to raw image space
	for (int i = 0; i < N; ++i)
	{
		float * rois_offset = rois + i * nmsMaxOut * 4;
		for (int j = 0; j < nmsMaxOut * 4 && imInfo[i * 3 + 2] != 1; ++j)
			rois_offset[j] /= imInfo[i * 3 + 2];
	}

	bboxTransformInvAndClip(rois, bboxPreds, predBBoxes, imInfo, N, nmsMaxOut, OUTPUT_CLS_SIZE);

	const float nms_threshold = 0.3f;
	const float score_threshold = 0.8f;

	for (int i = 0; i < N; ++i)
	{
		float *bbox = predBBoxes + i * nmsMaxOut * OUTPUT_BBOX_SIZE;
		float *scores = clsProbs + i * nmsMaxOut * OUTPUT_CLS_SIZE;
		for (int c = 1; c < OUTPUT_CLS_SIZE; ++c) // skip the background
		{
			std::vector<std::pair<float, int> > score_index;
			for (int r = 0; r < nmsMaxOut; ++r)
			{
				if (scores[r*OUTPUT_CLS_SIZE + c] > score_threshold)
				{
					score_index.push_back(std::make_pair(scores[r*OUTPUT_CLS_SIZE + c], r));
					std::stable_sort(score_index.begin(), score_index.end(),
						[](const std::pair<float, int>& pair1,
							const std::pair<float, int>& pair2) {
						return pair1.first > pair2.first;
					});
				}
			}

			// apply NMS algorithm
			std::vector<int> indices = nms(score_index, bbox, c, OUTPUT_CLS_SIZE, nms_threshold);
			// Show results
			for (unsigned k = 0; k < indices.size(); ++k)
			{
				int idx = indices[k];
				std::string storeName = CLASSES[c] + "-" + std::to_string(scores[idx*OUTPUT_CLS_SIZE + c]) + ".ppm";
				std::cout << "Detected " << CLASSES[c] << " in " << ppms[i].fileName << " with confidence " << scores[idx*OUTPUT_CLS_SIZE + c] * 100.0f << "% "
					<< " (Result stored in " << storeName << ")." << std::endl;

				BBox b{ bbox[idx*OUTPUT_BBOX_SIZE + c * 4], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 1], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 2], bbox[idx*OUTPUT_BBOX_SIZE + c * 4 + 3] };
				writePPMFileWithBBox(storeName, ppms[i], b);
			}
		}
	}

            Stamp stamp;
            this->getEnvelope(stamp);

            if (port_out_code.getOutputCount())
            {
                port_out_code.setEnvelope(stamp);
                yarp::sig::Vector codingYarpVec(codingVec.size(), &codingVec[0]);
                port_out_code.write(codingYarpVec);
            }

            if (port_out_img.getOutputCount())
            {
                port_out_img.write(img);
            }
        }

delete[] data;
	delete[] rois;
	delete[] bboxPreds;
	delete[] clsProbs;
	delete[] predBBoxes;

        mutex.post();

    }

public:

    fasterRCNNtensorRTPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
    {

        // Resource Finder and module options

        contextPath = rf.getHomeContextPath().c_str();

        // Data initialization (specific for Caffe method)

        // Binary file (.caffemodel) containing the network's weights
        string caffemodel_file = rf.check("caffemodel_file", Value("bvlc_googlenet.caffemodel")).asString().c_str();
        cout << "Setting .caffemodel file to " << caffemodel_file << endl;

        // Text file (.prototxt) defining the network structure
        string prototxt_file = rf.check("prototxt_file", Value("deploy.prototxt")).asString().c_str();
        cout << "Setting .prototxt file to " << prototxt_file << endl;

        // Name of blobs to be extracted
        string blob_name = rf.check("blob_name", Value("pool5/7x7_s1")).asString().c_str();

        // Boolean flag for timing or not the feature extraction
        bool timing = rf.check("timing",Value(false)).asBool();

        string  binaryproto_meanfile = "";
        float meanR = -1, meanG = -1, meanB = -1;
        int resizeWidth = -1, resizeHeight = -1;
        if (rf.find("binaryproto_meanfile").isNull() && rf.find("meanR").isNull())
        {
            cout << "ERROR: missing mean info!!!!!" << endl;
        }
        else if (rf.find("binaryproto_meanfile").isNull())
        {
            meanR = rf.check("meanR", Value(123)).asDouble();
            meanG = rf.check("meanG", Value(117)).asDouble();
            meanB = rf.check("meanB", Value(104)).asDouble();
            resizeWidth = rf.check("resizeWidth", Value(256)).asDouble();
            resizeHeight = rf.check("resizeHeight", Value(256)).asDouble();
            std::cout << "Setting mean to " << " R: " << meanR << " G: " << meanG << " B: " << meanB << std::endl;
            std::cout << "Resizing anysotropically to " << " W: " << resizeWidth << " H: " << resizeHeight << std::endl;

        }
        else if (rf.find("meanR").isNull())
        {
            binaryproto_meanfile = rf.check("binaryproto_meanfile", Value("imagenet_mean.binaryproto")).asString().c_str();
            cout << "Setting .binaryproto file to " << binaryproto_meanfile << endl;
        }
        else
        {
            std::cout << "ERROR: need EITHER mean file (.binaryproto) OR mean pixel values!" << std::endl;
        }

        rt_extractor = new fasterRCNNtensorRTExtractor(caffemodel_file, binaryproto_meanfile, meanR, meanG, meanB,
                prototxt_file, resizeWidth, resizeHeight,
                blob_name,
                timing);
	    if( !rt_extractor )
	    {
		    cout << "Failed to initialize fasterRCNNtensorRTExtractor" << endl;
	    }

        // Data (common to all methods)

        string name = rf.find("name").asString().c_str();

        port_out_img.open(("/"+name+"/img:o").c_str());
        port_out_code.open(("/"+name+"/code:o").c_str());

        BufferedPort<Image>::useCallback();

        rate = rf.check("rate",Value(0.0)).asDouble();
        last_read = 0.0;

        dump_code = rf.check("dump_code");
        if(dump_code)
        {
            string code_path = rf.check("dump_code",Value("codes.bin")).asString().c_str();
            code_path = contextPath + "/" + code_path;
            string code_write_mode = rf.check("append")?"wb+":"wb";

            fout_code = fopen(code_path.c_str(),code_write_mode.c_str());
        }

    }

    void interrupt()
    {
        mutex.wait();

        port_out_code.interrupt();
        port_out_img.interrupt();

        BufferedPort<Image>::interrupt();

        mutex.post();
    }

    void resume()
    {
        mutex.wait();

        port_out_code.resume();
        port_out_img.resume();

        BufferedPort<Image>::resume();

        mutex.post();
    }

    void close()
    {
        mutex.wait();

        if (dump_code)
        {
            fclose(fout_code);
        }

        port_out_code.close();
        port_out_img.close();

        delete gie_extractor;

        BufferedPort<Image>::close();

        mutex.post();
    }

    bool execReq(const Bottle &command, Bottle &reply)
    {
        switch(command.get(0).asVocab())
        {
        case(CMD_HELP):
            {
            reply.clear();
            reply.add(Value::makeVocab("many"));
            reply.addString("[dump] [path-to-file] [a] to start dumping the codes in the context directory. Use 'a' for appending.");
            reply.addString("[stop] to stop dumping.");
            return true;
            }

        case(DUMP_CODE):
            {
            mutex.wait();

            dump_code = true;
            string code_path;
            string code_write_mode;

            if (command.size()==1)
            {
                code_path = contextPath + "/codes.bin";
                code_write_mode="wb";
            }
            else if (command.size()==2)
            {
                if (strcmp(command.get(1).asString().c_str(),"a")==0)
                {
                    code_write_mode="wb+";
                    code_path = contextPath + "/codes.bin";
                } else
                {
                    code_write_mode="wb";
                    code_path = command.get(1).asString().c_str();
                }
            } else if (command.size()==3)
            {
                code_write_mode="wb+";
                code_path = command.get(2).asString().c_str();
            }

            fout_code = fopen(code_path.c_str(),code_write_mode.c_str());
            reply.addString("Start dumping codes...");

            mutex.post();
            return true;
            }

        case(DUMP_STOP):
            {
            mutex.wait();

            dump_code = false;
            fclose(fout_code);
            reply.addString("Stopped code dump.");

            mutex.post();

            return true;
            }

        default:
            return false;
        }
    }

};


class fasterRCNNtensorRTModule: public RFModule
{
protected:
    fasterRCNNtensorRTPort         *RTPort;
    Port                           rpcPort;

public:

    fasterRCNNtensorRTModule()
{
        RTPort = NULL;
}

    bool configure(ResourceFinder &rf)
    {

        string name = rf.find("name").asString().c_str();

        Time::turboBoost();

        RTPort = new fasterRCNNtensorRTPort(rf);

        RTPort->open(("/"+name+"/img:i").c_str());

        rpcPort.open(("/"+name+"/rpc").c_str());
        attach(rpcPort);

        return true;
    }

    bool interruptModule()
    {
        if (RTPort!=NULL)
            RTPort->interrupt();

        rpcPort.interrupt();

        return true;
    }

    bool close()
    {
        if(RTPort!=NULL)
        {
            RTPort->close();
            delete RTPort;
        }

        rpcPort.close();

        return true;
    }

    bool respond(const Bottle &command, Bottle &reply)
    {
        if (RTPort->execReq(command,reply))
            return true;
        else
            return RFModule::respond(command,reply);
    }

    double getPeriod()    { return 1.0;  }

    bool updateModule()
    {
        //RTPort->update();

        return true;
    }

};


int main(int argc, char *argv[])
{
    Network yarp;

    if (!yarp.checkNetwork())
        return 1;

    ResourceFinder rf;

    rf.setVerbose(true);

    rf.setDefaultContext("himrep");
    rf.setDefaultConfigFile("faster-rcnn_tensorrt.ini");

    rf.configure(argc,argv);

    rf.setDefault("name","faster-rcnn_tensorrt");

    fasterRCNNtensorRTModule mod;

    return mod.runModule(rf);
}
