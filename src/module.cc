#include "module.h"
#include "tensor.h"

namespace torchjs
{
// Class constructor
ScriptModule::ScriptModule(const std::string filename) : is_cuda(false)
{
  // Load the traced network from the file
  this->mModule = torch::jit::load(filename);
}

// Class deconstructor
ScriptModule::~ScriptModule()
{
}

NAN_MODULE_INIT(ScriptModule::Init)
{
  v8::Local<v8::FunctionTemplate> tpl = Nan::New<v8::FunctionTemplate>(New);
  tpl->SetClassName(Nan::New("ScriptModule").ToLocalChecked());
  tpl->InstanceTemplate()->SetInternalFieldCount(1);

  Nan::SetPrototypeMethod(tpl, "New", New);
  Nan::SetPrototypeMethod(tpl, "forward", forward);
  Nan::SetPrototypeMethod(tpl, "is_cuda_available", is_cuda_available);
  Nan::SetPrototypeMethod(tpl, "cuda", cuda);
  Nan::SetPrototypeMethod(tpl, "cpu", cpu);

  ScriptModule::constructor().Reset(Nan::GetFunction(tpl).ToLocalChecked());
  Nan::Set(target, Nan::New("ScriptModule").ToLocalChecked(),
           Nan::GetFunction(tpl).ToLocalChecked());
}

// JavaScript object creation
NAN_METHOD(ScriptModule::New)
{
  if (info.IsConstructCall())
  {
    // Get the filename parameter
    v8::String::Utf8Value param_filename(info[0]->ToString());
    const std::string filename = std::string(*param_filename);
    // Create a new script module using that file name
    ScriptModule *obj = new ScriptModule(filename);
    obj->Wrap(info.This());
    info.GetReturnValue().Set(info.This());
  }
  else
  {
    return Nan::ThrowError(
        Nan::New("ScriptModule::New - called without new keyword")
            .ToLocalChecked());
  }
}

NAN_METHOD(ScriptModule::forward)
{
  ScriptModule *script_module = ObjectWrap::Unwrap<ScriptModule>(info.Holder());
  Nan::MaybeLocal<v8::Object> maybe = Nan::To<v8::Object>(info[0]);
  Tensor *tensor = Nan::ObjectWrap::Unwrap<Tensor>(maybe.ToLocalChecked());

  torch::Tensor torch_tensor = tensor->getTensor();
  if (script_module->is_cuda)
  {
    torch_tensor = torch_tensor.cuda();
  }
  auto result =
      script_module->mModule.forward({torch_tensor});
  if(result.isTuple()){
    auto tuple = result.toTuple();
    v8::Local<v8::Array> arr = Nan::New<v8::Array>(tuple->elements().size());
    for (size_t i = 0; i < tuple->elements().size(); i++)
    {
      torch::Tensor o1 = tuple->elements()[i].toTensor();
      if (script_module->is_cuda)
      {
        o1 = o1.cpu();
      }
      auto newinst = Tensor::NewInstance();
      Tensor *obj = Nan::ObjectWrap::Unwrap<Tensor>(newinst);
      obj->setTensor(o1);
      Nan::Set(arr, i, newinst);
    }
    info.GetReturnValue().Set(arr);
  }else{
    torch::Tensor output = result.toTensor();
    if (script_module->is_cuda)
    {
      output = output.cpu();
    }
    auto newtensorinst = Tensor::NewInstance();
    Tensor *obj = Nan::ObjectWrap::Unwrap<Tensor>(newtensorinst);
    obj->setTensor(output);
    info.GetReturnValue().Set(newtensorinst);
  }  
}

NAN_METHOD(ScriptModule::cuda)
{
  ScriptModule *script_module = ObjectWrap::Unwrap<ScriptModule>(info.Holder());
  if(torch::cuda::is_available()){
    script_module->is_cuda = true;
    script_module->mModule.to(at::kCUDA);
  }
}

NAN_METHOD(ScriptModule::is_cuda_available)
{
  ScriptModule *script_module = ObjectWrap::Unwrap<ScriptModule>(info.Holder());
  if(torch::cuda::is_available()){
    info.GetReturnValue().Set(true);
  }else{
    info.GetReturnValue().Set(false);
  }
}

NAN_METHOD(ScriptModule::cpu)
{
  ScriptModule *script_module = ObjectWrap::Unwrap<ScriptModule>(info.Holder());
  script_module->is_cuda = false;
  script_module->mModule.to(at::kCPU);
}

} // namespace torchjs