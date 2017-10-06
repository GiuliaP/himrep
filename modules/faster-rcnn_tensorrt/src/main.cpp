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

class fasterRCNNtensorRTPort: public BufferedPort<Image>
{
private:

    // Resource Finder and module options

    ResourceFinder                &rf;

    string                        contextPath;

    double                        rate;
    double                        last_read;

    // Data (common to all methods)

    cv::Mat                       matImg;

    string*                       CLASSES;
    int                           nClasses;
                            
    Port                          port_out_img;
    Port 			                port_out_detection;

    Semaphore                     mutex;

    // Data (specific for each method - instantiate only those are needed)

    fasterRCNNtensorRTExtractor   *rt_extractor;

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

            std::vector< std::vector<float> > detectionScores;
            std::vector< std::vector< std::vector<float> > > detectionBoxes;
            std::vector<int> detectionClasses;
            
            // Extract the feature vector
            float times[2];
            if (!rt_extractor->extract(matImg, detectionClasses, detectionScores, detectionBoxes, times))
            {
                std::cout << "fasterRCNNtensorRTExtractor::extract(): failed..." << std::endl;
                return;
            }

            if (rt_extractor->timing)
            {
                std::cout << "PREP: " << times[0] << "  -  NET: " << times[1] << std::endl;
            }

            Stamp stamp;
            this->getEnvelope(stamp);

            if (port_out_detection.getOutputCount())
            {
               
               if (!detectionScores.empty() && !detectionBoxes.empty() && !detectionClasses.empty())
               { 
                  std::cout << detectionClasses.size() << std::endl;
                  std::cout << detectionScores.size() << std::endl;
                  std::cout << detectionBoxes.size() << std::endl;
                 
                  Bottle allScores;
                  for (int c=0; c<detectionClasses.size(); c++) // skip the background
                  {
                     if (!detectionScores[c].empty() && !detectionBoxes[c].empty())
                     {  
                        Bottle &b = allScores.addList();
                     
                        b.addString(CLASSES[detectionClasses[c]]);
                  
                        Bottle &bb = b.addList();
                        for (int k=0; k<detectionScores[c].size(); k++)
                        {
                           std::cout << CLASSES[detectionClasses[c]] << ": " 
                           << detectionScores[c][k] << "in (" << detectionBoxes[c][k][0] << ", "<< detectionBoxes[c][k][1] << ") - (" 
                           << detectionBoxes[c][k][2] << ", " << detectionBoxes[c][k][3] << ")" << std::endl;

                           bb.addDouble(detectionScores[c][k]);
                        
                           bb.addDouble(detectionBoxes[c][k][0]);
                           bb.addDouble(detectionBoxes[c][k][1]);
                           bb.addDouble(detectionBoxes[c][k][2]);
                           bb.addDouble(detectionBoxes[c][k][3]);
                        }
                     }
                  }
               
                  if (allScores.size()>0)
                  {
                     port_out_detection.setEnvelope(stamp);
                     port_out_detection.write(allScores);
                  }
              }
            }

            if (port_out_img.getOutputCount())
            {
               port_out_img.setEnvelope(stamp);
                
               for (int c=0; c<detectionClasses.size(); c++) // skip the background
               {
                  if (!detectionScores[c].empty() && !detectionBoxes[c].empty())
                  {  
                     for (int k=0; k<detectionScores[c].size(); k++)
                     {

                        int tlx = (int)detectionBoxes[c][k][0];
                        int tly = (int)detectionBoxes[c][k][1];
                        int brx = (int)detectionBoxes[c][k][2];
                        int bry = (int)detectionBoxes[c][k][3];
                        
                        int y_text, x_text;
			               y_text = tly-10;
			               x_text = tlx;
			               if (y_text<5)
				               y_text = bry+2;
				            
				            std::string text_string = CLASSES[detectionClasses[c]] + ": " + std::to_string(detectionScores[c][k]);
				            
				            std::cout << text_string << std::endl;
                        
				            cv::rectangle(tmp_mat,cv::Point(tlx,tly),cv::Point(brx,bry),cv::Scalar(255,255,255),2);
			               cv::putText(tmp_mat,text_string.c_str(),cv::Point(x_text,y_text), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2);
			      
                     }
                  }
               }
  
               port_out_img.write(img);
            }
        }
        
        mutex.post();

    }

public:

    fasterRCNNtensorRTPort(ResourceFinder &_rf) :BufferedPort<Image>(),rf(_rf)
    {

        // Resource Finder and module options

        contextPath = rf.getHomeContextPath().c_str();

        // Data initialization (specific for Caffe method)

        // Binary file (.caffemodel) containing the network's weights
        string caffemodel_file = rf.check("caffemodel_file", Value("VGG16_faster_rcnn_final.caffemodel")).asString().c_str();
        cout << "Setting .caffemodel file to " << caffemodel_file << endl;

        // Text file (.prototxt) defining the network structure
        string prototxt_file = rf.check("prototxt_file", Value("faster_rcnn_test_iplugin.prototxt")).asString().c_str();
        cout << "Setting .prototxt file to " << prototxt_file << endl;

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
            meanG = rf.check("meanG", Value(116)).asDouble();
            meanB = rf.check("meanB", Value(103)).asDouble();
            
            resizeWidth = rf.check("resizeWidth", Value(500)).asDouble();
            resizeHeight = rf.check("resizeHeight", Value(375)).asDouble();
            std::cout << "Setting mean to " << " R: " << meanR << " G: " << meanG << " B: " << meanB << std::endl;
            std::cout << "Resizing anysotropically to " << " W: " << resizeWidth << " H: " << resizeHeight << std::endl;

        }
        else if (rf.find("meanR").isNull())
        {
            binaryproto_meanfile = rf.check("binaryproto_meanfile", Value("mean.binaryproto")).asString().c_str();
            cout << "Setting .binaryproto file to " << binaryproto_meanfile << endl;
        }
        else
        {
            std::cout << "ERROR: need EITHER mean file (.binaryproto) OR mean pixel values!" << std::endl;
        }

        string in_blob_names = rf.check("in_blob_names", Value("data,iminfo")).asString().c_str();
        string out_blob_names = rf.check("out_blob_names", Value("bbox_pred,cls_prob,rois")).asString().c_str();
        
        int batchSize = rf.check("batch_size", Value(1)).asInt();
        
        ///////////////////////////////
        
         int poolingH = rf.check("poolingH", Value(7)).asInt();
         int poolingW = rf.check("poolingW", Value(7)).asInt();
   
         int featuresStride = rf.check("featuresStride", Value(16)).asInt();
         int preNmsTop = rf.check("preNmsTop", Value(6000)).asInt();
         int nmsMaxOut = rf.check("nmsMaxOut", Value(300)).asInt();
   
         int anchorsRatioCount = rf.check("anchorsRatioCount", Value(3)).asInt();
         int anchorsScaleCount = rf.check("anchorsScaleCount", Value(3)).asInt();
   
         float iouThreshold = rf.check("iouThreshold", Value(0.7)).asDouble();
         float minBoxSize = rf.check("minBoxSize", Value(16)).asDouble();
         float spatialScale = rf.check("spatialScale", Value(0.0625)).asDouble();
   
         float *anchorsRatios = new float[anchorsRatioCount];
         float *anchorsScales = new float[anchorsScaleCount];

         Bottle *r = rf.find("anchorsRatios").asList();
         Bottle *s = rf.find("anchorsScales").asList();
         
         for (int i=0; i<r->size(); i++)
         {
            anchorsRatios[i] = r->get(i).asDouble();
         }
         
         for (int i=0; i<s->size(); i++)
         {
            anchorsScales[i] = s->get(i).asDouble();
         }
         
         float nms_threshold = rf.check("nms_threshold", Value(0.3)).asDouble();
	      float score_threshold = rf.check("score_threshold", Value(0.8)).asDouble();
        
        
        //////////////////////////////
        
        if (rf.check("label_file"))
        {
            string label_file = rf.check("label_file", Value("")).asString().c_str() ;
            cout << "Setting labels to: " << label_file << endl;
            
            ifstream infile;
            
            string obj_name;
            vector<string> obj_names;
            int obj_idx;
            vector<int> obj_idxs;
            
            infile.open(label_file.c_str());
            infile>>obj_name;
            infile>>obj_idx;
            
            while (!infile.eof()) {
            
               std::cout << obj_name << " --> " << obj_idx <<std::endl;
               obj_names.push_back(obj_name);
               obj_idxs.push_back(obj_idx);
               
               infile>> obj_name;
               infile>>obj_idx;
            
            }
            infile.close();

            if (obj_names.size()!=obj_idxs.size())
            {
               std::cout << "label file wrongly formatted!" << std::endl;
            } 
        
            nClasses = obj_names.size();
            CLASSES = new string[nClasses];
            for (int i=0; i<nClasses; i++)
            {
               CLASSES[obj_idxs[i]] = obj_names[i];
            }
        }
        else
        {
               std::cout << "missing label file!" << std::endl;
        }
        
        //////////////////////////////
        
        std::cout << "caffemodel_file " << caffemodel_file << std::endl;
        std::cout << "binaryproto_meanfile " << binaryproto_meanfile << std::endl;
        std::cout << "meanR " << meanR << std::endl;
        std::cout << "meanG " << meanG << std::endl;
        std::cout << "meanB " << meanB << std::endl;
        
        std::cout << "prototxt_file " << prototxt_file << std::endl;
        std::cout << "resizeWidth " << resizeWidth << std::endl;
        std::cout << "resizeHeight " << resizeHeight << std::endl;
        std::cout << "timing " << timing << std::endl;

        std::cout << "in_blob_names " << in_blob_names << std::endl;
        std::cout << "out_blob_names " << out_blob_names << std::endl;
        std::cout << "batchSize " << batchSize << std::endl;
        
        std::cout << "nms_threshold " << nms_threshold << std::endl;
        std::cout << "score_threshold " << score_threshold << std::endl;
        std::cout << "nClasses " << nClasses << std::endl;
        
        rt_extractor = new fasterRCNNtensorRTExtractor(caffemodel_file, binaryproto_meanfile, meanR, meanG, meanB,
                prototxt_file, resizeWidth, resizeHeight, 
                timing,
                in_blob_names, out_blob_names,
                batchSize, 
                poolingH, poolingW, featuresStride, preNmsTop, nmsMaxOut, iouThreshold, minBoxSize, spatialScale, 
                anchorsRatios, anchorsRatioCount, anchorsScales, anchorsScaleCount, 
                nms_threshold, score_threshold,
                nClasses);
     
	     if ( !rt_extractor )
	     {
		     std::cout << "Failed to initialize fasterRCNNtensorRTExtractor" << std::endl;
	     }

        // Data (common to all methods)

        string name = rf.find("name").asString().c_str();

        port_out_img.open(("/"+name+"/img:o").c_str());
        port_out_detection.open(("/"+name+"/detection:o").c_str());

        BufferedPort<Image>::useCallback();

        rate = rf.check("rate",Value(0.0)).asDouble();
        last_read = 0.0;

    }

    void interrupt()
    {
        mutex.wait();

        port_out_detection.interrupt();
        port_out_img.interrupt();

        BufferedPort<Image>::interrupt();

        mutex.post();
    }

    void resume()
    {
        mutex.wait();

        port_out_detection.resume();
        port_out_img.resume();

        BufferedPort<Image>::resume();

        mutex.post();
    }

    void close()
    {
        mutex.wait();

        port_out_detection.close();
        port_out_img.close();

        delete rt_extractor;

        BufferedPort<Image>::close();

        delete[] CLASSES;
        
        mutex.post();
    }

    bool execReq(const Bottle &command, Bottle &reply)
    {
    
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

