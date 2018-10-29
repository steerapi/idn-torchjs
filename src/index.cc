/*********************************************************************
 * NAN - Native Abstractions for Node.js
 *
 * Copyright (c) 2015 NAN contributors
 *
 * MIT License <https://github.com/nodejs/nan/blob/master/LICENSE.md>
 ********************************************************************/

#include <nan.h>
#include "tensor.h" // NOLINT(build/include)
#include "module.h" // NOLINT(build/include)

using Nan::GetFunction;
using Nan::New;
using Nan::Set;
using v8::FunctionTemplate;
using v8::Handle;
using v8::Object;
using v8::String;

// Expose synchronous and asynchronous access to our
// Estimate() function
NAN_MODULE_INIT(InitModule)
{

  torchjs::ScriptModule::Init(target);
  torchjs::Tensor::Init(target);

  // Set(target, New<String>("calculateSync").ToLocalChecked(),
  //     GetFunction(New<FunctionTemplate>(CalculateSync)).ToLocalChecked());

  // Set(target, New<String>("calculateAsync").ToLocalChecked(),
  //     GetFunction(New<FunctionTemplate>(CalculateAsync)).ToLocalChecked());

  Set(target, New<String>("ones").ToLocalChecked(),
      GetFunction(New<FunctionTemplate>(torchjs::ones)).ToLocalChecked());
  Set(target, New<String>("zeros").ToLocalChecked(),
      GetFunction(New<FunctionTemplate>(torchjs::zeros)).ToLocalChecked());

  // Set(target, New<String>("ScriptModule").ToLocalChecked(),
  //     GetFunction(New<FunctionTemplate>(torchjs::ScriptModule::New)).ToLocalChecked());
}

NODE_MODULE(torchjs, InitModule)
