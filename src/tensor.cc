#include "tensor.h"

using std::cout;
using std::endl;

namespace torchjs
{
Nan::Persistent<v8::FunctionTemplate> Tensor::constructor;

Tensor::Tensor(){};
Tensor::~Tensor(){};

NAN_MODULE_INIT(Tensor::Init)
{
  v8::Local<v8::FunctionTemplate> ctor = Nan::New<v8::FunctionTemplate>(Tensor::New);
  Tensor::constructor.Reset(ctor);
  ctor->InstanceTemplate()->SetInternalFieldCount(1);
  ctor->SetClassName(Nan::New("Tensor").ToLocalChecked());

  // link our getters and setter to the object property
  Nan::SetAccessor(ctor->InstanceTemplate(), Nan::New("tensor").ToLocalChecked(), Tensor::HandleGetters, Tensor::HandleSetters);
  // Nan::SetAccessor(ctor->InstanceTemplate(), Nan::New("y").ToLocalChecked(), Tensor::HandleGetters, Tensor::HandleSetters);
  // Nan::SetAccessor(ctor->InstanceTemplate(), Nan::New("z").ToLocalChecked(), Tensor::HandleGetters, Tensor::HandleSetters);

  // Nan::SetPrototypeMethod(tpl, "setTensor", setTensor);
  // Nan::SetPrototypeMethod(tpl, "getTensor", getTensor);

  target->Set(Nan::New("Tensor").ToLocalChecked(), ctor->GetFunction());
}

void Tensor::setTensor(at::Tensor tensor)
{
  this->mTensor = tensor;
}
torch::Tensor Tensor::getTensor()
{
  return this->mTensor;
}

v8::Local<v8::Object> Tensor::NewInstance()
{
  v8::Local<v8::Function> constructorFunc = Nan::New(Tensor::constructor)->GetFunction();
  const int argc = 0;
  v8::Local<v8::Value> argv[] = {};
  return Nan::NewInstance(constructorFunc, argc, argv).ToLocalChecked();
  // return Nan::NewInstance(cons, 0, {}).ToLocalChecked();
}

NAN_METHOD(Tensor::toString)
{
  Tensor *obj = ObjectWrap::Unwrap<Tensor>(info.Holder());
  std::stringstream ss;
  at::IntList sizes = obj->mTensor.sizes();
  ss << "Tensor[Type=" << obj->mTensor.type() << ", ";
  ss << "Size=" << sizes << std::endl;
  info.GetReturnValue().Set(Nan::New(ss.str()).ToLocalChecked());
}

NAN_METHOD(Tensor::New)
{
  if (!info.IsConstructCall())
  {
    return Nan::ThrowError(Nan::New("Tensor::New - called without new keyword").ToLocalChecked());
  }

  Tensor *obj = new Tensor();
  obj->Wrap(info.Holder());
  info.GetReturnValue().Set(info.Holder());
}

NAN_METHOD(ones)
{
  // Sanity checking of the arguments
  if (info.Length() < 2)
    return Nan::ThrowError(Nan::New("Wrong number of arguments").ToLocalChecked());
  if (!info[0]->IsArray() || !info[1]->IsBoolean())
    return Nan::ThrowError(Nan::New("Wrong argument types").ToLocalChecked());
  // Retrieving parameters (require_grad and tensor shape)
  const bool require_grad = info[1]->BooleanValue();
  const v8::Local<v8::Array> array = info[0].As<v8::Array>();
  const uint32_t length = array->Length();
  // Convert from v8::Array to std::vector
  std::vector<int64_t> dims;
  for (int i = 0; i < length; i++)
  {
    v8::Local<v8::Value> v;
    int d = array->Get(i)->NumberValue();
    dims.push_back(d);
  }
  // Call the libtorch and create a new torchjs::Tensor object
  // wrapping the new torch::Tensor that was created by torch::ones
  at::Tensor v = torch::ones(dims, torch::requires_grad(require_grad));
  auto newinst = Tensor::NewInstance();
  Tensor *obj = Nan::ObjectWrap::Unwrap<Tensor>(newinst);
  obj->setTensor(v);
  info.GetReturnValue().Set(newinst);
}

NAN_METHOD(zeros)
{
  // Sanity checking of the arguments
  if (info.Length() < 2)
    return Nan::ThrowError(Nan::New("Wrong number of arguments").ToLocalChecked());
  if (!info[0]->IsArray() || !info[1]->IsBoolean())
    return Nan::ThrowError(Nan::New("Wrong argument types").ToLocalChecked());
  // Retrieving parameters (require_grad and tensor shape)
  const bool require_grad = info[1]->BooleanValue();
  const v8::Local<v8::Array> array = info[0].As<v8::Array>();
  const uint32_t length = array->Length();
  // Convert from v8::Array to std::vector
  std::vector<int64_t> dims;
  for (int i = 0; i < length; i++)
  {
    v8::Local<v8::Value> v;
    int d = array->Get(i)->NumberValue();
    dims.push_back(d);
  }
  // Call the libtorch and create a new torchjs::Tensor object
  // wrapping the new torch::Tensor that was created by torch::ones
  at::Tensor v = torch::zeros(dims, torch::requires_grad(require_grad));
  auto newinst = Tensor::NewInstance();
  Tensor *obj = Nan::ObjectWrap::Unwrap<Tensor>(newinst);
  obj->setTensor(v);
  info.GetReturnValue().Set(newinst);
}

template <typename T>
struct V8TypedArrayTraits; // no generic case
template <>
struct V8TypedArrayTraits<v8::Float32Array>
{
  typedef float value_type;
};
template <>
struct V8TypedArrayTraits<v8::Float64Array>
{
  typedef double value_type;
};
// etc. v8 doesn't export anything to make this nice.

template <typename T>
v8::Local<T> createTypedArray(size_t size)
{
  size_t byteLength = size * sizeof(typename V8TypedArrayTraits<T>::value_type);
  v8::Local<v8::ArrayBuffer> buffer = v8::ArrayBuffer::New(v8::Isolate::GetCurrent(), byteLength);
  v8::Local<T> result = T::New(buffer, 0, size);
  return result;
};

NAN_GETTER(Tensor::HandleGetters)
{
  Tensor *self = Nan::ObjectWrap::Unwrap<Tensor>(info.This());

  std::string propertyName = std::string(*Nan::Utf8String(property));
  if (propertyName == "tensor")
  {
    uint64_t numel = self->mTensor.numel();
    v8::Local<v8::Float32Array> array = createTypedArray<v8::Float32Array>(numel);

    Nan::TypedArrayContents<float> typedArrayContents(array);
    float *dst = *typedArrayContents;

    torch::Tensor x = self->mTensor.contiguous();
    auto x_p = x.data<float>();
    int64_t size = x.numel();

    // cout << "size" << size << endl;
    for (size_t i = 0; i < size; i++)
    {
      dst[i] = x_p[i];
    }

    info.GetReturnValue().Set(array);
  }
  else
  {
    info.GetReturnValue().Set(Nan::Undefined());
  }
}

NAN_SETTER(Tensor::HandleSetters)
{
  Tensor *self = Nan::ObjectWrap::Unwrap<Tensor>(info.This());

  if (!value->IsTypedArray())
  {
    return Nan::ThrowError(Nan::New("expected value to be a TypedArray").ToLocalChecked());
  }

  v8::Local<v8::Float32Array> array = value.As<v8::Float32Array>();
  Nan::TypedArrayContents<float> typedArrayContents(array);
  float *dst = *typedArrayContents;
  // printf("numel %d", self->mTensor.numel());

  torch::Tensor x = self->mTensor.contiguous();
  auto x_p = x.data<float>();
  int64_t size = x.numel();

  // cout << "size" << size << endl;
  for (size_t i = 0; i < size; i++)
  {
    x_p[i] = dst[i];
  }
}

} // namespace torchjs