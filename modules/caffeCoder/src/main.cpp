/*
 * Copyright (C) 2016 iCub Facility - Istituto Italiano di Tecnologia
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

#include "CaffeFeatExtractor.hpp"

using namespace std;
using namespace yarp;
using namespace yarp::os;
using namespace yarp::sig;
using namespace yarp::math;

#define CMD_HELP                    VOCAB4('h','e','l','p')
#define DUMP_CODE                   VOCAB4('d','u','m','p')
#define DUMP_STOP                   VOCAB4('s','t','o','p')

class CaffeCoderPort: public BufferedPort<Image>
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

    CaffeFeatExtractor<float>    *caffe_extractor;

    void onRead(Image &img)
    {

    	// Read at specified rate
        if (Time::now() - last_read < rate)
            return;

        mutex.wait();

        // If something arrived...
        if (img.width()>0 && img.height()>0)
        {

            // Convert the image

            cv::Mat tmp_mat=cv::cvarrToMat((IplImage*)img.getIplImage());
            cv::cvtColor(tmp_mat, matImg, CV_RGB2BGR);

            // Extract the feature vector

            std::vector<float> codingVecFloat;
            float times[2];
            caffe_extractor->extract_singleFeat_1D(matImg, codingVecFloat, times);
            if (!caffe_extractor->extract_singleFeat_1D(matImg, codingVecFloat, times))
            {
                std::cout << "CaffeFeatExtractor::extract_singleFeat_1D(): failed..." << std::endl;
                return;
            }
            std::vector<double> codingVec(codingVecFloat.begin(), codingVecFloat.end());

            if (caffe_extractor->timing)
            {
                std::cout << times[0] << ": PREP " << times[1] << ": NET" << std::endl;
            }

            // Dump if required
            if (dump_code)
            {
                fwrite (&codingVec[0], sizeof(double), codingVec.size(), fout_code);
            }

            Stamp stamp;
            this->getEnvelope(stamp);

            if(port_out_code.getOutputCount())
            {
                port_out_code.setEnvelope(stamp);
                Vector codingYarpVec(codingVec.size(), &codingVec[0]);
                port_out_code.write(codingYarpVec);
            }

            if(port_out_img.getOutputCount())
            {
                port_out_img.write(img);
            }
        }

        mutex.post();

    }

public:

    CaffeCoderPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
    {

        // Resource Finder and module options

        contextPath = rf.getHomeContextPath().c_str();

        // Data initialization (specific for Caffe method)

        // Binary file (.caffemodel) containing the network's weights
        string caffemodel_file = rf.check("caffemodel_file", Value("/usr/local/src/robot/caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel")).asString().c_str();
        cout << "Setting .caffemodel file to " << caffemodel_file << endl;

        // Text file (.prototxt) defining the network structure
        string prototxt_file = rf.check("prototxt_file", Value(contextPath + "/bvlc_googlenet_val_cutpool5.prototxt")).asString().c_str();
        cout << "Setting .prototxt file to " << prototxt_file << endl;

        // Name of blobs to be extracted
        string blob_name = rf.check("blob_name", Value("pool5/7x7_s1")).asString().c_str();
        cout << "Setting blob_names to " << blob_name << endl;

        // Boolean flag for timing or not the feature extraction
        bool timing = rf.check("timing",Value(false)).asBool();

        // Compute mode and eventually GPU ID to be used
        string compute_mode = rf.check("compute_mode", Value("GPU")).asString();
        int device_id = rf.check("device_id", Value(0)).asInt();

        int resizeWidth = rf.check("resizeWidth", Value(256)).asDouble();
        int resizeHeight = rf.check("resizeHeight", Value(256)).asDouble();

        caffe_extractor = NULL;
        caffe_extractor = new CaffeFeatExtractor<float>(caffemodel_file,
                prototxt_file, resizeWidth, resizeHeight,
                blob_name,
                compute_mode,
                device_id,
                timing);

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


class CaffeCoderModule: public RFModule
{
protected:
    CaffeCoderPort         *caffePort;
    Port                   rpcPort;

public:

    CaffeCoderModule()
{
        caffePort=NULL;
}

    bool configure(ResourceFinder &rf)
    {

        string name = rf.find("name").asString().c_str();

        Time::turboBoost();

        caffePort = new CaffeCoderPort(rf);

        caffePort->open(("/"+name+"/img:i").c_str());

        rpcPort.open(("/"+name+"/rpc").c_str());
        attach(rpcPort);

        return true;
    }

    bool interruptModule()
    {
        if (caffePort!=NULL)
            caffePort->interrupt();

        rpcPort.interrupt();

        return true;
    }

    bool close()
    {
        if(caffePort!=NULL)
        {
            caffePort->close();
            delete caffePort;
        }

        rpcPort.close();

        return true;
    }

    bool respond(const Bottle &command, Bottle &reply)
    {
        if (caffePort->execReq(command,reply))
            return true;
        else
            return RFModule::respond(command,reply);
    }

    double getPeriod()    { return 1.0;  }

    bool updateModule()
    {
        //caffePort->update();

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
    rf.setDefaultConfigFile("caffeCoder.ini");

    rf.configure(argc,argv);

    rf.setDefault("name","caffeCoder");

    CaffeCoderModule mod;

    return mod.runModule(rf);
}

