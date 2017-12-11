#include <boost/python.hpp>
#include "laplacian_foveation.hpp"


char const* greet()
{
   return "hello, world";
}

cv::Mat foveate(const cv::Mat & img, const int & size_map, const int & levels, const int & sigma, const cv::Mat & fixation_point)
{
    // Foveate images
    int m = floor(4*img.size().height);
    int n = floor(4*img.size().width);

    cv::Mat image;
    img.convertTo(image, CV_64F);

    // Compute kernels
    std::vector<Mat> kernels = createFilterPyr(m, n, levels, sigma);

    // Construct Pyramid
    LaplacianBlending pyramid(image,levels, kernels);

    // Foveate
    cv::Mat foveated_image = pyramid.foveate(fixation_point);

    foveated_image.convertTo(foveated_image,CV_8UC3);
    cv::resize(foveated_image,foveated_image,Size(size_map,size_map));

    return foveated_image;
}


BOOST_PYTHON_MODULE(pylib)
{
    using namespace boost::python;
    /*class_< Bonjour >("Bonjour", init<std::string>())
      .def("greet", &Bonjour::greet)
      .add_property("msg", &Bonjour::get_msg, &Bonjour::set_msg);*/

    def("foveate", foveate);
}

int main(int argc, char** argv) {

}
