#include <boost/python.hpp>
#include "laplacian_foveation.hpp"

using namespace boost::python;
BOOST_PYTHON_MODULE(yolt_python)
{
    class_< LaplacianBlending >("LaplacianBlending", init<const cv::Mat,const int,const int >())
      .def("foveate", &LaplacianBlending::foveate);
}


