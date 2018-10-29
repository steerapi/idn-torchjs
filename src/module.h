#ifndef MODULE_H
#define MODULE_H

#include <nan.h>
#include <torch/script.h>
#include <string>

namespace torchjs
{
class ScriptModule : public Nan::ObjectWrap
{
public:
  static NAN_MODULE_INIT(Init)
  {
    v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
    tpl->SetClassName(Nan::New("ScriptModule").ToLocalChecked());
    tpl->InstanceTemplate()->SetInternalFieldCount(1);

    Nan::SetPrototypeMethod(tpl, "New", New);
    Nan::SetPrototypeMethod(tpl, "forward", forward);
    Nan::SetPrototypeMethod(tpl, "cuda", cuda);
    Nan::SetPrototypeMethod(tpl, "cpu", cpu);

    constructor().Reset(Nan::GetFunction(tpl).ToLocalChecked());
    Nan::Set(target, Nan::New("ScriptModule").ToLocalChecked(),
             Nan::GetFunction(tpl).ToLocalChecked());
  }
  // static v8::Local<v8::Object> NewInstance();

private:
  explicit ScriptModule(const std::string filename) : is_cuda(false)
  {
    // Load the traced network from the file
    this->mModule = torch::jit::load(filename);
  }
  ~ScriptModule() {}
  static NAN_METHOD(New);
  static NAN_METHOD(forward);
  static NAN_METHOD(cuda);
  static NAN_METHOD(cpu);
  static inline Nan::Persistent<v8::Function> &constructor()
  {
    static Nan::Persistent<v8::Function> my_constructor;
    return my_constructor;
  }

private:
  std::shared_ptr<torch::jit::script::Module> mModule;
  bool is_cuda;
};
} // namespace torchjs
#endif