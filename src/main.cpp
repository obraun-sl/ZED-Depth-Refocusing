///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2021, .
// Author: Olivier Braun
//////////////////////////////////////////////////////////////////////////

/****************************************************************************************************
 ** This sample demonstrates how to grab and process images/depth on a CUDA kernel                 **
 ** This sample creates a simple layered depth-of-filed rendering based on CUDAconvolution sample  **
 ****************************************************************************************************/



 
// ZED SDK include
#include <sl/Camera.hpp>

// OpenGL extensions
#include <GL/glew.h>
#include <GL/freeglut.h>

// CUDA specific for OpenGL interoperability
#include <cuda_gl_interop.h>

// OpenCV include (to create gaussien kernel)
#include "opencv2/opencv.hpp"

// CUDA functions 
#include "dof_gpu.h"

using namespace sl;
using namespace std;

 
// Declare some resources (GL texture ID, GL shader ID...)
GLuint imageTex;
cudaGraphicsResource* pcuImageRes;

// ZED Camera object
sl::Camera zed;

// sl::Mat ressources
sl::Mat gpuImageLeft;
sl::Mat gpuImageOutput;
sl::Mat gpuDepthMap;
sl::Mat gpuDepthMapNorm;

// tmp buffer for convolution
::uchar4* d_buffer_image;


// Focus point detected in pixels (X,Y) when mouse click event
int x_focus_point;
int y_focus_point;
float depth_focus_point = 0.f;
float norm_depth_focus_point = 0.f;

float min_depth_focus_mm = 500.f;//Minimum depth focus range in mm
float max_depth_focus_mm = 15000.f;//Maximum depth focus range in mm


inline float clamp(float v, float v_min, float v_max) {
	return v < v_min ? v_min : (v > v_max ? v_max : v);
}


void mouseButtonCallback(int button, int state, int x, int y) {
 
	if (button==0) {
 		x_focus_point = x;
		y_focus_point = y;
		//--> get the depth at the mouse click point
        gpuDepthMap.getValue<sl::float1>(x_focus_point, y_focus_point, &depth_focus_point, sl::MEM::GPU);


		//--> check that the value is a number...
		if (std::isfinite(depth_focus_point))
		{
			std::cout << " Focus point set at : " << depth_focus_point << " mm at " << x_focus_point << "," << y_focus_point << std::endl;
            norm_depth_focus_point = (max_depth_focus_mm - depth_focus_point) / (max_depth_focus_mm - min_depth_focus_mm);
			clamp(norm_depth_focus_point, 0.f, 1.f);
		}
	}
}
 


void close() {

    //#Cleaning
    gpuImageLeft.free();
    gpuImageOutput.free();
    gpuDepthMap.free();
    gpuDepthMapNorm.free();
    zed.close();
    glBindTexture(GL_TEXTURE_2D, 0);
    glutDestroyWindow(1);
}

void draw() {

	sl::RuntimeParameters params;
    params.sensing_mode = sl::SENSING_MODE::FILL;

    sl::ERROR_CODE res = zed.grab(params);

    if (res == sl::ERROR_CODE::SUCCESS) {


		/// Retrieve Image and Depth
        zed.retrieveImage(gpuImageLeft,sl::VIEW::LEFT,sl::MEM::GPU);
        zed.retrieveMeasure(gpuDepthMap, sl::MEASURE::DEPTH, sl::MEM::GPU);

 		/// Process Image with CUDA
        ///--> normalize the depth map and make separable convolution
        normalizeDepth(gpuDepthMap.getPtr<float>(sl::MEM::GPU), gpuDepthMapNorm.getPtr<float>(sl::MEM::GPU), gpuDepthMap.getStep(sl::MEM::GPU),  min_depth_focus_mm, max_depth_focus_mm, gpuDepthMap.getWidth(), gpuDepthMap.getHeight());
        convolutionRowsGPU((::uchar4*)d_buffer_image,(::uchar4*)gpuImageLeft.getPtr<sl::uchar4>(sl::MEM::GPU), gpuDepthMapNorm.getPtr<float>(sl::MEM::GPU), gpuImageLeft.getWidth(), gpuImageLeft.getHeight(), gpuDepthMapNorm.getStep(sl::MEM::GPU), norm_depth_focus_point);
        convolutionColumnsGPU((::uchar4*)gpuImageOutput.getPtr<sl::uchar4>(sl::MEM::GPU), (::uchar4*)d_buffer_image, gpuDepthMapNorm.getPtr<float>(sl::MEM::GPU), gpuImageLeft.getWidth(), gpuImageLeft.getHeight(), gpuDepthMapNorm.getStep(sl::MEM::GPU), norm_depth_focus_point);
	
  
		/// Map to OpenGL and display
		cudaArray_t ArrIm;
        cudaGraphicsMapResources(1, &pcuImageRes, 0);
        cudaGraphicsSubResourceGetMappedArray(&ArrIm, pcuImageRes, 0, 0);
        cudaMemcpy2DToArray(ArrIm, 0, 0, gpuImageOutput.getPtr<sl::uchar4>(sl::MEM::GPU), gpuImageOutput.getStepBytes(sl::MEM::GPU), gpuImageOutput.getWidth() * sizeof(sl::uchar4), gpuImageOutput.getHeight(), cudaMemcpyDeviceToDevice);
        cudaGraphicsUnmapResources(1, &pcuImageRes, 0);

       		

		//OpenGL Part
        glDrawBuffer(GL_BACK); //write to both BACK_LEFT & BACK_RIGHT
        glLoadIdentity();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        //Draw Image Texture in Left Part of Side by Side
        glBindTexture(GL_TEXTURE_2D, imageTex);


		glBegin(GL_QUADS);        
		glTexCoord2f(0.0, 1.0); 
		glVertex2f(-1.0, -1.0);
		glTexCoord2f(1.0, 1.0);  
		glVertex2f(1.0, -1.0);   
		glTexCoord2f(1.0, 0.0);    
		glVertex2f(1.0, 1.0);    
		glTexCoord2f(0.0, 0.0);  
		glVertex2f(-1.0, 1.0);   
		glEnd();


	    //swap.
        glutSwapBuffers();

    }

    glutPostRedisplay();

}

int main(int argc, char **argv) {

    if (argc > 2) {
        std::cout << "Only the path of a SVO can be passed in arg" << std::endl;
        return -1;
    }
    //init glut
    glutInit(&argc, argv);

    /*Setting up  The Display  */
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    //Configure Window Postion
    glutInitWindowPosition(50, 25);

    //Configure Window Size
    glutInitWindowSize(1280, 720);

    //Create Window
    glutCreateWindow("ZED Depth Refocusing");

    //init GLEW Library
    glewInit();

    sl::InitParameters parameters;
    parameters.depth_mode = sl::DEPTH_MODE::QUALITY;
    parameters.camera_resolution = sl::RESOLUTION::HD720;
    parameters.coordinate_units = sl::UNIT::MILLIMETER;
	parameters.depth_minimum_distance = 50.0;



    sl::ERROR_CODE err = zed.open(parameters);
    // ERRCODE display
    std::cout << "ZED Init Err : " << sl::toString(err) << std::endl;
    if (err != sl::ERROR_CODE::SUCCESS) {
        zed.close();
        return -1;
    }


    // Get Image Size
    sl::Resolution cam_resolution_ = zed.getCameraInformation().camera_configuration.resolution;
    int image_width_ = cam_resolution_.width;
    int image_height_ = cam_resolution_.height;

    cudaError_t err1;

    // Create and Register OpenGL Texture for Image (RGBA -- 4channels)
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &imageTex);
    glBindTexture(GL_TEXTURE_2D, imageTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width_, image_height_, 0, GL_BGRA_EXT, GL_UNSIGNED_BYTE, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    err1 = cudaGraphicsGLRegisterImage(&pcuImageRes, imageTex, GL_TEXTURE_2D, cudaGraphicsMapFlagsNone);
    if (err1 != 0 ) return -1;

	

	// Alloc sl::Mat and tmp buffer
    gpuImageLeft.alloc(cam_resolution_, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
    gpuImageOutput.alloc(cam_resolution_, sl::MAT_TYPE::U8_C4, sl::MEM::GPU);
    gpuDepthMap.alloc(cam_resolution_, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);
    gpuDepthMapNorm.alloc(cam_resolution_, sl::MAT_TYPE::F32_C1, sl::MEM::GPU);
	cudaMalloc((void **)&d_buffer_image, image_width_*image_height_ * 4);


	// Create all the gaussien kernel for different radius and copy them to GPU
	vector<cv::Mat> gaussianKernel;
	vector<float*> h_kernel;
	vector<int> filter_sizes_;
	for (int radius = 1; radius <= KERNEL_RADIUS; ++radius)
		filter_sizes_.push_back(2 * radius + 1);

	for (int i = 0; i < filter_sizes_.size(); ++i) {
		gaussianKernel.push_back(cv::getGaussianKernel(filter_sizes_[i],-1, CV_32F));
		h_kernel.push_back(gaussianKernel[i].ptr<float>(0));
		copyKernel(h_kernel[i], i);

	}

	x_focus_point = image_width_ / 2;
	y_focus_point = image_height_ / 2;

	std::cout << "** Click on the image to set the focus distance **" << std::endl;

    //Set Draw Loop
    glutDisplayFunc(draw);
	glutMouseFunc(mouseButtonCallback);
    glutCloseFunc(close);
    glutMainLoop();

    return 0;
}



