#include <pybind11/pybind11.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>

#include "opencv2/objdetect.hpp"
//Convert
#include <pybind11/stl.h>
#include "conversions.h"
#include <iostream>
#include "laplacian_foveation.hpp"
namespace py = pybind11;

cv::Mat read_image(std::string image_name)
{
        cv::Mat image = cv::imread(image_name, CV_LOAD_IMAGE_COLOR);
            return image;
}

void receiveVec6f(cv::Vec6f v)
{
  std::cout << "Point: " << v << std::endl;
}


cv::Scalar test_scalar(cv::Scalar s)
{
    return s;
}

cv::Point3f test_point3f(cv::Point3f p)
{
    return p;
}

cv::Vec6f sendVec6f()
{
    cv::Vec6f v(1, 2, 3, 4, 5, 6);
    return v;
}

PYBIND11_MODULE(pysmooth_foveation, m){
    
    //General initializing functions
     
    m.doc() = "pysmooth_foveation python wrapper";
    NDArrayConverter::init_numpy();
    
    m.def("read_image", &read_image, "A function that read an image", py::arg("image"));
    m.def("receiveVec6f", &receiveVec6f);
    m.def("test_scalar", &test_scalar);
    m.def("test_point3f", &test_point3f);
    m.def("sendVec6f", &sendVec6f);


    py::class_<LaplacianBlending>(m, "LaplacianBlending")
	/*Construtor: py::init*/
        .def(py::init<const int &, const int &, const int &, const int &, const int &, const int &>())
	/*.def -> metodos*/        
	.def("Foveate", &LaplacianBlending::Foveate)
        .def("update_fovea", &LaplacianBlending::CreateFilterPyr);
}
