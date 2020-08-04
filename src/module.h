#ifndef MODULE_H
#define MODULE_H

#include <nan.h>
#include <string>
#include <torch/script.h>

namespace torchjs
{
class ScriptModule : public Nan::ObjectWrap
{
public:
  static NAN_MODULE_INIT(Init);
  // static v8::Local<v8::Object> NewInstance();

private:
  explicit ScriptModule(const std::string filename);
  ~ScriptModule();
  static NAN_METHOD(New);
  static NAN_METHOD(forward);
  static NAN_METHOD(is_cuda_available);
  static NAN_METHOD(cuda);
  static NAN_METHOD(cpu);
  static inline Nan::Persistent<v8::Function> &constructor()
  {
    static Nan::Persistent<v8::Function> my_constructor;
    return my_constructor;
  }

private:
  torch::jit::Module mModule;
  bool is_cuda;
};
} // namespace torchjs
#endif