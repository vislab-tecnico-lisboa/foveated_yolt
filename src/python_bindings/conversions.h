// Adapted from https://github.com/edmBernard/pybind11_opencv_numpy
// Changes by Joao Avelino, 2019

# ifndef __NDARRAY_CONVERTER_H__
# define __NDARRAY_CONVERTER_H__

#include <Python.h>
#include <opencv2/core/core.hpp>


class NDArrayConverter {
public:
    // must call this first, or the other routines don't work!
    static bool init_numpy();
    
    static bool toMat(PyObject* o, cv::Mat &m);
    static PyObject* toNDArray(const cv::Mat& mat);
};

//
// Define the type converter
//

#include <pybind11/pybind11.h>
#include <iostream>

namespace pybind11 { namespace detail {
    
template <> struct type_caster<cv::Mat> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};


//cv::Mat convertions. Needs to be checked but appears to be working.
//
template <> struct type_caster<cv::Mat_<float>> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat_<float>, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};
    
template <> struct type_caster<cv::Mat_<u_char>> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat_<uchar>, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};

template <> struct type_caster<cv::Mat_<double>> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat_<double>, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};

template <> struct type_caster<cv::Mat_<int>> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Mat_<int>, _("numpy.ndarray"));
    
    bool load(handle src, bool) {
        return NDArrayConverter::toMat(src.ptr(), value);
    }
    
    static handle cast(const cv::Mat &m, return_value_policy, handle defval) {
        return handle(NDArrayConverter::toNDArray(m));
    }
};

//cv::Rect conversions: converts to/from python tuple!
template <> struct type_caster<cv::Rect_<float>> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Rect_<float>, _("cvrect"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        return PyArg_ParseTuple(obj, "ffff", &value.x, &value.y, &value.width, &value.height) > 0;
    }
    
    //From c++ to python 
    static handle cast(const cv::Rect_<float> &r, return_value_policy, handle defval) {
        return Py_BuildValue("(dddd)", r.x, r.y, r.width, r.height);
    }
};

template <> struct type_caster<cv::Rect> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Rect, _("cvrect"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        return PyArg_ParseTuple(obj, "ffff", &value.x, &value.y, &value.width, &value.height) > 0;
    }
    
    //From c++ to python 
    static handle cast(const cv::Rect &r, return_value_policy, handle defval) {
        return Py_BuildValue("(dddd)", r.x, r.y, r.width, r.height);
    }
};


//cv::Point conversions: convert to a python tuple (x, y)
template <> struct type_caster<cv::Point_<double>> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Point_<double>, _("cvpoint"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        return PyArg_ParseTuple(obj, "dd", &value.x, &value.y) > 0;
    }
    
    //From c++ to python 
    static handle cast(const cv::Point_<double> &p, return_value_policy, handle defval) {
        return Py_BuildValue("(dd)", p.x, p.y);
    }
};

template <> struct type_caster<cv::Point> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Point, _("cvpoint"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        return PyArg_ParseTuple(obj, "dd", &value.x, &value.y) > 0;
    }
    
    //From c++ to python 
    static handle cast(const cv::Point &p, return_value_policy, handle defval) {
        
        return Py_BuildValue("(dd)", double(p.x), double(p.y));
    }
};

template <> struct type_caster<cv::Point2f> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Point2f, _("cvpoint2f"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        double x, y; 
        bool test = PyArg_ParseTuple(obj, "dd", &x, &y) > 0;
        value.x = x;
        value.y = y;


        return test;
    }
    
    //From c++ to python 
    static handle cast(const cv::Point2f &p, return_value_policy, handle defval) {
        
        return Py_BuildValue("(dd)", double(p.x), double(p.y));
    }
};

template <> struct type_caster<cv::Point3f> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Point3f, _("cvpoint3f"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        double x, y, z; 
        bool test = PyArg_ParseTuple(obj, "ddd", &x, &y, &z) > 0;
        value.x = x;
        value.y = y;
        value.z = z;


        return test;
    }
    
    //From c++ to python 
    static handle cast(const cv::Point3f &p, return_value_policy, handle defval) {
        
        return Py_BuildValue("(ddd)", double(p.x), double(p.y), double(p.z));
    }
};

//cv::Vec6f conversion
template <> struct type_caster<cv::Vec6f> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Vec6f, _("cvvec6f"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        double v[6];
        int test = PyArg_ParseTuple(obj, "dddddd", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]);
        if(test <=0)
            return false;

        cv::Vec6f out(v[0], v[1], v[2], v[3], v[4], v[5]);

        value[0] = v[0];
        value[1] = v[1];
        value[2] = v[2];
        value[3] = v[3];
        value[4] = v[4];
        value[5] = v[5];

        return true;
    }
    
    //From c++ to python 
    static handle cast(const cv::Vec6f &v, return_value_policy, handle defval) {
        
        return Py_BuildValue("(dddddd)", double(v[0]),double(v[1]), double(v[2]),double(v[3]),double(v[4]),double(v[5]));
    }
};

template <> struct type_caster<cv::Vec3d> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Vec3d, _("cvvec3d"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        double v[3];
        bool test = PyArg_ParseTuple(obj, "ddd", &v[0], &v[1], &v[2]) > 0;

        cv::Vec3d out(v[0], v[1], v[2]);

        value[0] = v[0];
        value[1] = v[1];
        value[2] = v[2];

        return true;
    }
    
    //From c++ to python 
    static handle cast(const cv::Vec3d &v, return_value_policy, handle defval) {
        
        return Py_BuildValue("(ddd)", double(v[0]),double(v[1]), double(v[2]));
    }
};


template <> struct type_caster<cv::Vec2f> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Vec2f, _("cvvec2f"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        double v[2];
        int test = PyArg_ParseTuple(obj, "dd", &v[0], &v[1]);

        cv::Vec2f out(v[0], v[1]);

        value[0] = v[0];
        value[1] = v[1];

        return true;
    }
    
    //From c++ to python 
    static handle cast(const cv::Vec2f &v, return_value_policy, handle defval) {
        
        return Py_BuildValue("(dd)", double(v[0]),double(v[1]));
    }
};

//cv::Scalar
template <> struct type_caster<cv::Scalar> {
public:
    
    PYBIND11_TYPE_CASTER(cv::Scalar, _("cvscalar"));
    
    //From python to c++
    bool load(handle src, bool) {
        PyObject *obj = src.ptr();

        if(!obj || obj == Py_None)
            return false;
        
        int v[3];
        int test = PyArg_ParseTuple(obj, "iii", &v[0], &v[1], &v[2]);

        cv::Scalar out(v[0], v[1], v[2]);

        value[0] = v[0];
        value[1] = v[1];
        value[2] = v[2];

        return true;
    }
    
    //From c++ to python 
    static handle cast(const cv::Scalar &v, return_value_policy, handle defval) {
        
        return Py_BuildValue("(iii)", int(v[0]), int(v[1]), int(v[2]));
    }
};



}} // namespace pybind11::detail

# endif
