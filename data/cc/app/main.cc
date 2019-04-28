#include <gflags/gflags.h>
#include <glog/logging.h>
#include <paddle_inference_api.h>
#include "debug_reader.h"

DEFINE_string(model_dir, "", "model dir");
DEFINE_string(prog_file, "", "program filename");
DEFINE_string(params_file, "", "params filename");
DEFINE_string(data, "", "input data file");
DEFINE_int32(batch_size, 1, "batch size");
DEFINE_bool(enable_ir_optim, true, "enable ir optim");
DEFINE_bool(use_gpu, false, "use gpu");
DEFINE_int32(gpu_memory, 1000, "gpu memory(MB)");
DEFINE_int32(gpu_id, 0, "gpu device id");

using paddle::AnalysisConfig;

namespace paddle {

namespace debug {
namespace app {

void SetConfig(AnalysisConfig *config) {
  // LOG_IF(FATAL, FLAGS_model_dir.empty()) << "Model dir is empty";
  CHECK_NE(FLAGS_model_dir.empty(), false) << "Model dir is empty";

  if (FLAGS_prog_file.empty() || FLAGS_params_file.empty()) {
    config->SetModel(FLAGS_model_dir);
  } else {
    config->SetModel(FLAGS_prog_file, FLAGS_params_file);
  }

  config->SwitchIrOptim(FLAGS_enable_ir_optim);
  if (FLAGS_use_gpu) {
    config->EnableUseGpu(FLAGS_gpu_memory, FLAGS_gpu_id);
  }
}

std::unique_ptr<paddle::debug::Reader> CreateDataReader() {
  std::unique_ptr<paddle::debug::Reader> reader(
      new paddle::debug::Reader(FLAGS_data));
  reader->SetBatchSize(FLAGS_batch_size);
  return reader;
}

}  // namespace app
}  // namespace debug
}  // namespace paddle

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(*argv);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  AnalysisConfig config;
  paddle::debug::app::SetConfig(&config);
  auto predictor = paddle::CreatePaddlePredictor(config);
  CHECK_EQ(FLAGS_data.empty(), false);
  auto reader = paddle::debug::app::CreateDataReader();

  const auto &inputs = reader->data();

  reader->NextBatch();
  std::vector<paddle::PaddleTensor> outputs;
  predictor->Run(inputs, &outputs);

  return 0;
}
