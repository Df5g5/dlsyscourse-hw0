#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

#include <memory>

namespace py = pybind11;

struct IMat{
    std::pair<size_t, size_t> shape;

    virtual float* operator[](size_t){ return nullptr;};
    virtual const float* operator[](size_t) const = 0;
    virtual void applyFunc(std::function<float(float)>) = 0;
    virtual void normalize() = 0;
    virtual void subtractIy(const unsigned char *y) { assert(false); }
    virtual void product(float) { assert(false); }
    virtual void subtractMat(IMat*) {assert(false);}
};
struct ConstMat;
struct Mat : public IMat{

    float* data;

    Mat(std::pair<size_t, size_t> _shape, float* _data = nullptr){
        shape = _shape;
        if(_data){
            data=_data;
        }else{
            auto __data=new float[shape.first * shape.second];
            
            data=__data;
        }
    }
    const float *operator[](size_t i) const {
        return data + shape.second*i;
    }
    float *operator[](size_t i){
        return data + shape.second*i;
    }
    void applyFunc(std::function<float(float)> func){
        for(int i=0;i<shape.first * shape.second;i++){
            data[i] = func(data[i]);
        }
    }
    void product(float x){
        for(int i=0;i<shape.first * shape.second;i++){
            data[i] *= x;
        }
    }
    void normalize(){
        for(int i=0;i<shape.first;i++){
            float sum=0;
            for(int j=0;j<shape.second;j++){
                sum+=data[i * shape.second + j];
            }
            for(int j=0;j<shape.second;j++){
                data[i * shape.second + j] /= sum;
            }
        }
    }
    void subtractIy(const unsigned char *y) {
        for(int i=0;i<shape.first;i++){
            data[i*shape.second + *y] -= 1;
            y++;
        }
    }
    void subtractMat(IMat *x){
        assert(shape.second==x->shape.second);

        for(int i=0;i<x->shape.first;i++){
            for(int j=0;j<x->shape.second;j++){
                (*this)[i][j] -= (*x)[i][j]; 
            }
        }

            // for(int i=0;i<x->shape.first * x->shape.second;i++){
            //     data[i] -= x->data[i];
            // }
        
    }
};

struct ConstMat : public IMat{

    const float* data;
    
    ConstMat(std::pair<size_t, size_t> _shape, const float* _data){
        shape = _shape;
        if(_data){
            data=_data;
        }
    }
    const float *operator[](size_t i) const {
        return data + shape.second*i;
    }
    void applyFunc(std::function<float(float)>){ assert(false); }
    virtual void normalize(){assert(false); }
};

namespace MatO{
    IMat* matmul(const IMat *x, const IMat *y){
        
        assert(x->shape.second == y->shape.first);

        IMat *result = new Mat({x->shape.first, y->shape.second});
        
        for(int i=0;i<x->shape.first;i++){
            for(int j=0;j<y->shape.second;j++){
                (*result)[i][j]=0;
                for(int k=0;k< x->shape.second;k++)
                    (*result)[i][j] += (*x)[i][k] * (*y)[k][j];
            }
        }
        return result;
    }
    IMat* matmul_transposed(const IMat *x, const IMat *y, int transpose =0){
        assert(transpose==1); //TODO transpose second matrix too

        IMat *result = new Mat({x->shape.second, y->shape.second});
        for(int i=0;i<x->shape.second;i++){
            for(int j=0;j<y->shape.second;j++){
                 (*result)[i][j]=0;
                for(int k=0;k< x->shape.first;k++)
                    (*result)[i][j] += (*x)[k][i] * (*y)[k][j];
            }
        }
        return result;
    
    }
}


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /**
     *     n_iterations = int(X.shape[0] / batch)
        for i in range(n_iterations):
          X_batch = X[i * batch:(i + 1) * batch] # (batch * input_dim)
          y_batch = y[i * batch:(i + 1) * batch] # (batch * input_dim)
          exp_X_theta = np.exp(np.matmul(X_batch, theta)) # (batch x num_classes)
          Z = exp_X_theta / np.sum(exp_X_theta, axis=1)[:,None] # (batch x num_classes)
          I_y = np.zeros_like(Z)
          np.put_along_axis(I_y, y_batch[:,None], 1, axis=1)  # (batch x num_classes)
          grad_softmax = np.matmul(X_batch.T, (Z - I_y)) / batch
          theta -= lr * grad_softmax
     * 
     */
    /// BEGIN YOUR CODE
        size_t n_iterations = m/batch;
        for(int i=0;i<n_iterations;i++){
            IMat *X_batch = new ConstMat({batch, n}, X+i*batch*n);
            auto y_batch = y + batch*i;
            IMat *mat_theta = new Mat({n, k}, theta);
            auto Z = MatO::matmul(X_batch, mat_theta);
            Z->applyFunc(expf);
            Z->normalize();
            Z->subtractIy(y_batch);
            Z->product(1.f / batch);
            
            auto grad = MatO::matmul_transposed(X_batch, Z, 1);
            grad->product(lr);

            mat_theta->subtractMat(grad);

        }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
