#ifndef TENSOR_H
#define TENSOR_H

#include <nan.h>
#include <torch/torch.h>

namespace torchjs
{
class Tensor : public Nan::ObjectWrap
{
public:
  static NAN_MODULE_INIT(Init);
  void setTensor(at::Tensor tensor);
  torch::Tensor getTensor();
  static v8::Local<v8::Object> NewInstance();

private:
  explicit Tensor();
  ~Tensor();
  static NAN_METHOD(New);
  static NAN_METHOD(toString);

  static NAN_GETTER(HandleGetters);
  static NAN_SETTER(HandleSetters);

  static Nan::Persistent<v8::FunctionTemplate> constructor;

private:
  torch::Tensor mTensor;
};

NAN_METHOD(ones);
NAN_METHOD(zeros);

} // namespace torchjs
#endif