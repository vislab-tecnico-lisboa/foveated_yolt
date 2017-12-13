#include <boost/python.hpp>
#include "laplacian_foveation.hpp"


using namespace boost::python;
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(X_returnsum_overloads, LaplacianBlending::foveate, 1, 1)

BOOST_PYTHON_MODULE(yolt_python)
{
    class_< LaplacianBlending >("LaplacianBlending", init<const cv::Mat,const int,const int >())
      .def("foveate", &LaplacianBlending::foveate, X_returnsum_overloads( ));//args("center"), "foveation function"));
}


